"""
Evaluate LLM on Sudoku puzzles using an API.

We call the LLM repeatedly:
  1) Provide an initial puzzle prompt.
  2) LLM responds with a single forced placement (e.g., <ANSWER>\nr3c6: 5\n</ANSWER>).
  3) We check if that placement is valid and correct based on the puzzle's known solution.
  4) If correct, we update the board and continue; if incorrect, we stop.
  5) Continue until the puzzle is solved or we reach a maximum number of steps.

Example Usage:
--------------
export OPENAI_API_KEY="your_openai_api_key"
export DATASET="challenge_100"
export API="openai"
export MODEL="gpt-4o-mini-2024-07-18"
python -m eval.run \
    --dataset ${DATASET} \
    --output_csv ../data/benchmark_results/${DATASET}/${MODEL}.csv \
    --api ${API} \
    --model ${MODEL} \
    --batch_size 20

Output:
-------
A CSV file with columns:
[
    "data_source",
    "puzzle_id",
    "model",
    "num_empty_cells",
    "shuffle_seed",
    "n_response_idx",
    "n_history_turns",
    "setting",
    "conversation",
    "num_rounds",
    "num_correct_placements",
    "final_solved",
    "final_board",
]

Plus a summary of average correctness/final-solved rates in stdout.
"""

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional, Union
import uuid

# import aiohttp
# import anthropic
import datasets
import jinja2
# import openai
import pandas as pd
from tqdm import tqdm
# from transformers import AutoTokenizer
# try:
#     from vllm import AsyncLLMEngine, SamplingParams
#     from vllm.engine.arg_utils import AsyncEngineArgs
# except ImportError:
#     print("vllm not installed. Please install vllm to use it.")
#     AsyncLLMEngine = None
#     SamplingParams = None
#     AsyncEngineArgs = None

# Add smolagents imports
try:
    from smolagents import CodeAgent, LiteLLMModel
except ImportError:
    print("smolagents not installed. Please run `uv add smolagents litellm`")
    CodeAgent = None
    LiteLLMModel = None

from eval.prompts import (
    BOARD_PROMPT,
    PREFILLED_ASSISTANT_RESPONSE,
    RULE_PROMPT,
)
from eval.utils import (
    extract_action_from_response,
    pretty_print_visual_elements,
    random_fill_hints,
    smolagents_output_to_string,
)
from sudoku_ds import (
    SudokuAction,
    SudokuBoard,
)


async def process_one(
    args: argparse.Namespace,
    request: Dict,
    model: str,
) -> Dict:
    # Load data
    rules = request["rules"]
    current_board_ascii = request["initial_board"]
    solution_ascii = request["solution"]
    rows = request["rows"]
    cols = request["cols"]
    visual_elements = request["visual_elements"]
    if pd.isna(visual_elements) or visual_elements == "":
        visual_elements = None
    n_history_turns = request["n_history_turns"]

    # Construct setting string
    settings = []
    if n_history_turns == -1:
        settings.append("full-history")
    else:
        assert n_history_turns >= 0
        settings.append(f"{n_history_turns}-history-turns")
    if len(settings) == 0:
        setting = "default"
    else:
        setting = "_".join(settings)

    # Pretty print visual elements
    if visual_elements is None:
        pretty_visual_elements = None
    else:
        visual_elements = json.loads(visual_elements)
        pretty_visual_elements = pretty_print_visual_elements(visual_elements)

    # Construct boards
    solution_board = SudokuBoard.from_ascii(solution_ascii, rows, cols)
    current_board = SudokuBoard.from_ascii(current_board_ascii, rows, cols)
    max_rounds = current_board.to_ascii(unfilled=".").count(".")

    # Initial conversation
    rule_prompt = jinja2.Template(RULE_PROMPT).render(
        rows=rows,
        cols=cols,
        rules=rules,
        pretty_visual_elements=pretty_visual_elements,
    )
    # `history_conversation`` is for recording
    # Actual input conversation will be constructed before calling API
    history_conversation = [
        {"role": "user", "content": rule_prompt},
        {"role": "assistant", "content": PREFILLED_ASSISTANT_RESPONSE}
    ]

    # Initialize smolagents CodeAgent if api is smolagents
    agent = None
    if args.agent_framework == "smolagents":
        if CodeAgent is None or LiteLLMModel is None:
            raise ImportError("smolagents is not installed. Please run `pip install -r requirements.txt`")
        # Use args.model as model_id for LiteLLMModel
        llm_model = LiteLLMModel(
            model_id=model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            # top_k is not directly supported by LiteLLMModel, configure via model params if needed
        )
        agent = CodeAgent(
            tools=[], # No tools needed for Sudoku
            model=llm_model,
        )
        

    num_correct_placements = 0
    assistant_response = None # Initialize assistant_response
    for round_idx in range(max_rounds):
        round_str = f"Round {round_idx + 1} / {max_rounds}"

        ##################
        ## Get response ##
        ################## 

        # Construct user prompt describing the current board
        board_prompt = jinja2.Template(BOARD_PROMPT).render(
            current_board=current_board.to_spaced_ascii(unfilled="."),
        )
        history_conversation.append({"role": "user", "content": board_prompt})

        # Construct input conversation
        # If full history, include all history turns
        if n_history_turns == -1:
            input_conversation = [
                {"role": message["role"], "content": message["content"]}
                for message in history_conversation
            ]
        # Otherwise
        # - First two prompts are fixed (rule prompt and prefilled assistant response)
        # - Last prompt is the current board
        # - In between, we add the most recent history turns
        else:
            input_conversation = [
                {"role": message["role"], "content": message["content"]}
                for message in \
                    history_conversation[:2] \
                    + history_conversation[2:-1][-2*n_history_turns:] \
                    + history_conversation[-1:]
            ]

        # Call agent
        if args.agent_framework == "smolagents":
            if agent is None:
                 print(f"[Fail] {round_str}. Smolagent not initialized.")
                 break
            try:
                # Format the input conversation history into a single string prompt
                # TODO: Consider a more robust formatting function if needed
                prompt_string = ""
                for i, msg in enumerate(input_conversation):
                    role = msg["role"].capitalize()
                    content = msg["content"]
                    # Simple formatting, add newlines between turns
                    if i > 0:
                        prompt_string += "\n\n"
                    prompt_string += f"{role}: {content}"

                # Call agent.run with the full formatted history and reset=True
                result = await asyncio.to_thread(
                    agent.run,
                    prompt_string,
                    reset=True # Treat each call as stateless from the agent's perspective
                )

                assistant_response = smolagents_output_to_string(result)
            except Exception as e:
                # TODO: Implement retry logic similar to the original call_api if needed
                print(f"[Fail] {round_str}. Error calling smolagent: {e}")
                # Use the previous response if available, otherwise break
                if assistant_response is None:
                    break
                # If there was a previous response, try to reuse it or handle error
                print(f"Using previous response due to error.")

        # --- Placeholder for other API/Agent Frameworks ---
        # elif args.api == "openai":
        #     # Original call_api logic or a refactored version for OpenAI
        #     pass # Replace with actual call
        # elif args.api == "anthropic":
        #     # Original call_api logic or a refactored version for Anthropic
        #     pass # Replace with actual call
        # ... etc.
        else:
             print(f"[Fail] {round_str}. Unsupported Agent Framework: {args.agent_framework}")
             break

        # Teriminate if no response
        if not assistant_response:
            print(f"{round_str}. No response from server.")
            break

        # Update conversation
        history_conversation.append({"role": "assistant", "content": assistant_response})

        #################################
        ## Solution-independent checks ##
        ################################# 

        # Extract action from response
        action = extract_action_from_response(assistant_response)
        # Terminate if no action found
        if not action:
            print(f"[Fail] {round_str}. No valid action found in response.")
            break

        # Convert to SudokuAction
        try:
            r_str, c_str, val_str = action
            sudoku_action = SudokuAction.from_tokens([
                "<vl>", f"<value{val_str}>", f"<r{r_str}>", f"<c{c_str}>"
            ])
        # Terminate if action parsing fails
        except Exception as e:
            print(f"[Fail] {round_str}. Error parsing action: {e}.")
            break

        # Update board state
        try:
            current_board.execute_action(sudoku_action)
        # Terminate if action execution fails
        except Exception as e:
            print(f"[Fail] {round_str}. Error executing action: {e}")
            break

        ###############################
        ## Solution-dependent checks ##
        ###############################

        # Check correctness
        action_row, action_col = sudoku_action.coordinates[0]
        ref = solution_board.get_cell(action_row, action_col).value.value
        hyp = sudoku_action.value.value 
        if hyp == ref:
            print(f"[Pass] {round_str}.")
            num_correct_placements += 1
        # Terminate if incorrect placement
        else:
            print(f"[Fail] {round_str}. Incorrect placement at {action_row}, {action_col}.")
            break

        # Teriminate if all cells are filled
        if '.' not in current_board.to_ascii(unfilled="."):
            print(f"[Pass] {round_str}. All cells filled.")
            break

    ##########################
    ## Final solution match ##
    ##########################

    # Check if solution is correct
    final_board_ascii = current_board.to_ascii(unfilled=".")
    final_solved = 1 if (final_board_ascii == solution_ascii) else 0

    return {
        # From input
        "data_source": args.dataset,
        "puzzle_id": request["puzzle_id"],
        "model": args.model_save_name if args.model_save_name else model,
        "num_empty_cells": request["num_empty_cells"],
        "shuffle_seed": request["shuffle_seed"],
        "n_response_idx": request["n_response_idx"],
        "n_history_turns": n_history_turns,
        "setting": setting,
        "initial_board": request["initial_board"],
        # From output
        "conversation": json.dumps(history_conversation),
        "num_rounds": round_idx + 1,
        "num_correct_placements": num_correct_placements,
        "final_solved": final_solved,
        "final_board": final_board_ascii,
    }


async def process_batch(
    args: argparse.Namespace,
    requests: List[Dict],
    model: str,
    batch_size: int = 1
) -> List[Dict]:
    semaphore = asyncio.Semaphore(batch_size)
    async def process_with_semaphore(request):
        async with semaphore:
            return await process_one(
                args=args,
                request=request,
                model=model,
            )
    
    tasks = [process_with_semaphore(request) for request in requests]
    outputs = []
    
    # Process requests with progress bar
    with tqdm(total=len(tasks), desc="Processing requests") as pbar:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            outputs.append(result)
            pbar.update(1)
    
    return outputs


def construct_request(
    puzzle_id: str,
    author: str,
    rules: str,
    visual_elements: Optional[str],
    initial_board: str,
    solution: str,
    rows: int,
    cols: int,
    num_empty_cells: int,
    shuffle_seed: Optional[int],
    n_response_idx: int,
    n_history_turns: int,
) -> Optional[Dict]:
    # Fill hints if needed
    if num_empty_cells > 0:
        initial_board = random_fill_hints(
            initial_board,
            solution,
            num_empty_cells,
            shuffle_seed,
        )
        if initial_board is None:
            return None
    return {
        "puzzle_id": puzzle_id,
        "author": author,
        "rules": rules,
        "visual_elements": visual_elements,
        "initial_board": initial_board,
        "solution": solution,
        "rows": rows,
        "cols": cols,
        "num_empty_cells": num_empty_cells,
        "shuffle_seed": shuffle_seed,
        "n_response_idx": n_response_idx,
        "n_history_turns": n_history_turns,
    }
    

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on Sudoku puzzles in a multi-round manner.")

    # Filepaths
    parser.add_argument("--dataset", type=str, required=True, choices=["challenge_100", "nikoli_100", "ctc"],
                        help="Dataset to evaluate on.")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Output CSV path.")
    
    # Subset of puzzles to evaluate
    parser.add_argument("--iloc_start", type=int, default=0,
                        help="Start index of puzzles to evaluate.")
    parser.add_argument("--iloc_end", type=int, default=None,
                        help="End index of puzzles to evaluate (exclusive).")
    parser.add_argument("--ilocs", type=int, nargs="+",
                        help="Specific puzzle indices to evaluate. Overrides start/end.")

    # Eval setting
    parser.add_argument("--puzzle_size", type=int, default=None,
                        help="Filter puzzles by size (e.g., 4 for 4x4). If None, use all sizes.")
    # The number of evaluations for each puzzle is the product of the following four arguments.
    parser.add_argument("--num_empty_cells", type=int, nargs="+", default=[0, 10, 20],
                        help="Number of empty cells in the intial board after hint fill in random cells. "
                             "0 means the original board.")
    parser.add_argument("--shuffle_seeds", type=int, nargs="+", default=[0],
                        help="Shuffle seeds for the random hint fill. Only used if num_empty_cells > 0.")
    parser.add_argument("--n_response_idxs", type=int, nargs="+", default=[0],
                        help="If you want to run multiple trials per puzzle/hint/seed. E.g., [0,1,2,3,4] for 5 runs.")
    parser.add_argument("--n_history_turns", type=int, nargs="+", default=[5],
                        help="Number of history turns to include in each LLM prompt. -1 means full history.")

    # Model
    parser.add_argument("--agent_framework", type=str, default="smolagents",
                        choices=["smolagents"],
                        help="Agent Framework or direct API to use for evaluation.")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name or path.")
    parser.add_argument("--model_save_name", type=str,
                        help="Model name in saved result. If not provided, use --model.")
    parser.add_argument("--max_tokens", type=int, default=8192,
                        help="Max tokens in each LLM response.")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="LLM temperature.")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p sampling probability.")
    parser.add_argument("--top_k", type=int, default=40,
                        help="Top-k sampling.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for parallel processing.")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Max retries for API calls.")
    parser.add_argument("--retry_delay", type=float, default=5.0,
                        help="Delay (in second) between retries.")

    # vLLM specific
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Tensor parallel size for vLLM.")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1,
                        help="Pipeline parallel size for vLLM.")
    parser.add_argument("--draft_model", type=str,
                        help="Use the draft model.")
    
    args = parser.parse_args()

    # Sanity check
    assert args.num_empty_cells != [0] or len(args.shuffle_seeds) == 1, \
        "shuffle_seed is only used when providing hints (i.e. num_empty_cells > 0)."

    # Load puzzle
    dataset = datasets.load_dataset("SakanaAI/Sudoku-Bench", args.dataset, split="test")
    
    # Filter by puzzle size if specified
    if args.puzzle_size is not None:
        print(f"Filtering dataset for puzzle size: {args.puzzle_size}x{args.puzzle_size}")
        original_count = len(dataset)
        dataset = dataset.filter(lambda example: example.get('rows') == args.puzzle_size and example.get('cols') == args.puzzle_size)
        filtered_count = len(dataset)
        print(f"Filtered dataset from {original_count} to {filtered_count} puzzles.")
        if filtered_count == 0:
            print(f"Warning: No puzzles found for size {args.puzzle_size}x{args.puzzle_size} in dataset {args.dataset}. Exiting.")
            return

    # Use a subset of puzzles if specified
    if args.ilocs is not None:
        ilocs = args.ilocs
    else:
        end_idx = args.iloc_end if args.iloc_end is not None else len(dataset)
        ilocs = range(args.iloc_start, end_idx)
    puzzle_rows = [dataset[i] for i in ilocs]
    print(f"Number of puzzles to evaluate: {len(puzzle_rows)}")

    # Construct requests
    requests = []
    for puzzle_row in puzzle_rows:
        for nhist in args.n_history_turns:
            for ne in args.num_empty_cells:
                for sseed in args.shuffle_seeds:
                    for nr_idx in args.n_response_idxs:
                        request = construct_request(
                            puzzle_id=puzzle_row["puzzle_id"],
                            author=puzzle_row["author"],
                            rules=puzzle_row["rules"],
                            visual_elements=puzzle_row["visual_elements"],
                            initial_board=puzzle_row["initial_board"],
                            solution=puzzle_row["solution"],
                            rows=puzzle_row["rows"],
                            cols=puzzle_row["cols"],
                            num_empty_cells=ne,
                            shuffle_seed=sseed,
                            n_response_idx=nr_idx,
                            n_history_turns=nhist,
                        )
                        if request is not None:
                            requests.append(request)
    print(f"Number of requests to process: {len(requests)}")

    # Process batch
    all_results = asyncio.run(process_batch(
        args=args,
        batch_size=args.batch_size,
        requests=requests,
        model=args.model
    ))

    # Convert results to DataFrame
    res_df = pd.DataFrame(all_results)
    if len(res_df) == 0:
        print("No results to save. Possibly no puzzles or an error occurred.")
        return

    # Print summary
    # We'll measure average number of correct placements and fraction of puzzles solved.
    group_cols = ["num_empty_cells", "setting", "model"]
    summary = (
        res_df
        .groupby(group_cols)
        .agg({
            "num_correct_placements": "mean",
            "final_solved": "mean"
        })
        .reset_index()
    )
    with pd.option_context("display.max_rows", None, "display.precision", 2):
        print(summary)

    # Save results to CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    res_df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to {args.output_csv}")


if __name__ == "__main__":
    main()