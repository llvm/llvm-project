#!/usr/bin/env python3
"""
Run an automatic search to find the fastest paramater values for LLVM
optimization passes.
"""

import argparse
import csv
import datetime
import json
import sys
import time
from statistics import median
from typing import Any, Never
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from hyperopt import hp, fmin, Trials, STATUS_OK, STATUS_FAIL
import run

bitcode_file_name: str = ""
REPETITIONS: int = 25
arch:str = ""
persistent_llc_flags: list[str] = []

def json_to_space(json_path: str) -> dict[str, Any] | Never:
    """
    Parses the JSON file at JSON_PATH and transforms it into a space
    dict used by hyperopt.fmin
    """
    space = {}
    with open(json_path, 'r') as json_file:
        config = json.load(json_file)
        # Populate the persistent flags global
        persistent_llc_flags.extend(config['persistent_llc_flags'])
        for flag in config['llc_flags']:
            name = flag['flag']
            match flag['type']:
                case "uint" | "int" | "number":
                    value_range = flag['range']
                    distribution = value_range['distribution']
                    if distribution == "logarithmic":
                        low = value_range['low']
                        high = value_range['high']
                        step = value_range['step']
                        if low == 0 or high == 0:
                            print("0 is an invalid number for a log distribution",
                                  file=sys.stderr)
                            print(f"Found in {name}", file=sys.stderr)
                            sys.exit(1)
                        space[name] = hp.qloguniform(name,
                                                     np.log(low),
                                                     np.log(high),
                                                     step)
                    elif distribution == "uniform":
                        low = value_range['low']
                        high = value_range['high']
                        step = value_range['step']
                        space[name] = hp.quniform(name, low, high, step)
                    # maybe: support normal distributions too
                    else:
                        print(f"Distribution {distribution} not supported!",
                              file=sys.stderr)
                        sys.exit(1)
                case "value" | "string":
                    value_range = flag['range']
                    choice = value_range['choice']
                    space[name] = hp.choice(name, choice)
                case "boolean":
                    space[name] = hp.choice(name, [True, False])
    return space

def objective(params: dict[str, Any]) -> dict[str, Any]:
    """
    The objective function. Returns an object describing the loss
    (median kernel runtime), status, and flags.
    """
    llc_args: list[str] = []
    for k, v in params.items():
        if isinstance(v, bool):
            # Turn pythonic True/False into true/false
            llc_args.append(f"--{k}={str(v).lower()}")
        elif isinstance(v, float):
            # Make sure that floats that don't have decimal parts get
            # turned into ints first
            if v - int(v) == 0:
                llc_args.append(f"--{k}={str(int(v))}")
            else :
                llc_args.append(f"--{k}={v}")
        else:
            llc_args.append(f"--{k}={v}")

    # Append persistent flags
    llc_args.extend(persistent_llc_flags)

    config = {"llc_args": llc_args,
              "replay_args": [f"--repetitions={REPETITIONS}"],
              "opt_passes": [],
              "opt_args": ["--O3"]}

    # NOTE: Since we're already invoking `opt --O3` within the
    # baseline run, we don't need to repeatedly invoke opt when
    # calling llc.
    code = run.run_llc(bitcode_file_name,
                       # Dummy argument
                       "AUTORUN",
                       config,
                       False,  # dry_run
                       False,  # also_opt
                       arch,
                       False)  # verbose
    if code != 0:
        sys.exit(code)
    json_path = run.check_record_json_file(bitcode_file_name)
    output_dir = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
    runtimes = run.replay_and_measure_kernel(json_path,
                                             config,
                                             output_dir,
                                             False,  # run_bitcode
                                             False)  # dry_run
    if runtimes:
        return {
            'loss': median(runtimes),
            'eval_time': time.time(),
            'baseline': "false",
            'config': config,
            'status': STATUS_OK
        }
    return {
        'loss': float("NaN"),
        'eval_time': time.time(),
        'baseline': "false",
        'config': config,
        'status': STATUS_FAIL
    }

def write_trials_csv(baseline: dict[str, Any],
                     trial_results: list[dict[str, Any]]) -> None:
    """
    Write the trial results out as a CSV file.
    """
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"autosearch-results-{now}.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['loss', 'eval_time', 'baseline', 'config', 'status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames,
                                quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        # Write baseline results
        writer.writerow(baseline)
        # Write trial results
        for row in trial_results:
            writer.writerow(row)

def save_loss_history_plot(baseline_loss: float, trials: Trials) -> None:
    """
    Save the loss history + the baseline as a scatterplot. Make a
    horizontal line from the baseline for easy visual comparison.
    """
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    losses = list(trials.losses())
    x = range(len(losses) + 1)
    y = [ baseline_loss ] + losses
    plt.scatter(x, y)
    plt.axhline(baseline_loss, c="g")
    plt.xlabel("Trial iteration")
    plt.ylabel("Median runtime (ns)")
    plt.savefig(f"loss_history-{now}.png", dpi=300, bbox_inches="tight")
    plt.close()

def main():
    global bitcode_file_name, arch
    parser = argparse.ArgumentParser(
        description="Automatically run parameter search for llc or opt flags."
    )
    parser.add_argument(
        "bc_path",
        help="Path to the recorded device IR bitcode (.bc) file"
    )
    parser.add_argument(
        "json_path",
        help="Path to the JSON file containing settings for desired flags"
    )
    parser.add_argument(
        "--arch",
        default="gfx90a",
        help="GPU Architecture (default: gfx90a)"
    )
    parser.add_argument(
        "--max-trials",
        default="100",
        help="Maximum number of trials (defualt: 100)"
    )
    parser.add_argument(
        "--no-plot",
        default=False,
        help="Don't save the median runtime history plot as a PNG image",
        action='store_true'
    )

    args = parser.parse_args()
    trials = Trials()

    arch = args.arch
    bc_path = args.bc_path
    json_path = args.json_path
    max_trials = int(args.max_trials)

    # Back up original bc and image files
    bitcode_file_name = run.get_original_bitcode(bc_path)
    run.backup_image(run.image_output_file(bc_path))

    # Run the baseline
    print("=" * 80)
    print("Replaying the baseline, unmodified, kernel:")
    print(f"{'=' * 80}")
    kernel_json_path = run.check_record_json_file(bitcode_file_name)
    output_dir = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
    baseline_config = {
        "opt_passes": [],
        "opt_args": ["--O3"],
        "llc_args": ["-O3"],
        "replay_args": [f"--repetitions={REPETITIONS}"]
    }
    status = run.run_llc(bitcode_file_name,
                         "BASELINE",
                         baseline_config,
                         False, # dry_run
                         True,  # also_opt
                         arch,
                         False) # verbose
    if status != 0:
        sys.exit(status)
    baseline_runtimes = run.replay_and_measure_kernel(kernel_json_path,
                                                      baseline_config,
                                                      output_dir,
                                                      False, # run_bitcode
                                                      False) # dry_run
    if baseline_runtimes:
        baseline = {
            "loss": median(baseline_runtimes),
            "eval_time": time.time(),
            "baseline": "true",
            "config": baseline_config,
            "status": STATUS_OK
        }
    else:
        print("Could not measure baseline runtimes", file=sys.stderr)
        sys.exit(1)

    # Initialize search space
    space = json_to_space(json_path)

    # Run search
    best = fmin(
        fn=objective,         # Objective Function to optimize
        space=space,          # Hyperparameter Search Space
        max_evals=max_trials, # Number of optimization attempts
        trials=trials
    )

    if not args.no_plot:
        save_loss_history_plot(baseline["loss"], trials)

    write_trials_csv(baseline, trials.results)

if __name__ == "__main__":
    main()
