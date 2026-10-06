#!/usr/bin/env python3

"""
Run opt or llc on a recorded, unoptimized, kernel IR bitcode (bc) file
to transform it using the pipeline definitions and opt/llc commandline
flags read from a user-defined JSON file (see example-opt-passes.json
in the repo). Replay the transformed kernel and measure its
performance for each transformation using rocprof, output a CSV file
and a boxplot of the results.
"""

import argparse
import datetime
import json
import subprocess
import sys
import shutil
import os
import csv

from collections import defaultdict
from typing import Generator, Never, TypeAlias

import matplotlib.pyplot as plt

PipelineConfType: TypeAlias = dict[str, list[str]]
PipelinesJsonType: TypeAlias = dict[str, PipelineConfType]
# Timeout for running rocprof
TIMEOUT: int = 90


def load_pipelines(pipelines_file: str) -> PipelinesJsonType | Never:
    """
    Load the pipeline definitions from a JSON file path.
    """
    if not os.path.exists(pipelines_file):
        print(
            f"Error: '{pipelines_file}' not found in current directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        with open(pipelines_file, "r") as f:
            pipelines = json.load(f)

        if not isinstance(pipelines, dict):
            print(
                f"Error: '{pipelines_file}' must contain a JSON object.",
                file=sys.stderr,
            )
            sys.exit(1)

        if not pipelines:
            print(f"Error: '{pipelines_file}' is empty.", file=sys.stderr)
            sys.exit(1)

        return pipelines

    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse '{pipelines_file}': {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to read '{pipelines_file}': {e}", file=sys.stderr)
        sys.exit(1)


def backup_bitcode(bc_path: str) -> str:
    """
    Make sure that the bitcode file at bc_path is backed up. A backup is a copy
    of the .bc file with '.original' appended at the end of the file name. If
    the backup file already exists or the user already invokes the script with
    the .bc.original file, do nothing.
    """
    if bc_path.endswith(".original"):
        return bc_path
    bc_backup_path = bc_path + ".original"
    if not os.path.exists(bc_backup_path):
        print(f"Backing up the bitcode file to {bc_backup_path}")
        shutil.copyfile(bc_path, bc_backup_path)
        return bc_backup_path
    return bc_backup_path


def backup_image(image_path: str) -> None:
    """
    Like backup_bitcode() but for the image file.
    """
    image_backup_path = image_path + ".original"
    if os.path.exists(image_backup_path):
        return
    print(f"Backing up the image file to {image_backup_path}")
    shutil.copyfile(image_path, image_backup_path)


def get_original_bitcode(bc_path: str) -> str | Never:
    """
    Always return the path to the original bitcode file, calling
    backup_bitcode() if it doesn't yet exist. Warning: we at the moment
    cannot verify that the file ending in '.original' is, in fact, the
    original.
    """
    if not os.path.exists(bc_path):
        print("Error: the specified bitcode file could not be found", file=sys.stderr)
        sys.exit(1)
    if bc_path.endswith(".original"):
        return bc_path
    return backup_bitcode(bc_path)


def check_record_json_file(bitcode_file: str) -> str | Never:
    """
    Return the path to the JSON file corresponding to the kernel's
    recording or exit with an error.
    """
    json_path = bitcode_file.replace(".bc", ".json").replace(".original", "")
    if not os.path.exists(json_path):
        print(
            f"Error: could not find a JSON file corresponding to the specified bitcode file path: {json_path}",
            file=sys.stderr,
        )
        sys.exit(1)
    return json_path


def opt_output_file(bitcode_file: str) -> str:
    """
    Strip the '.original' suffix from the IR bitcode file path if it
    exists. Necessary due to the fact that the replay tool expects the
    original '.bc' file extension instead of '.bc.original'.
    """
    return (
        bitcode_file.replace(".original", "")
        if bitcode_file.endswith(".original")
        else bitcode_file
    )


def llc_output_file(bitcode_file: str) -> str:
    """
    Strip the '.original' suffix from the IR bitcode file path if it
    exists and replace it with '.o'.
    """
    output_file = bitcode_file
    if output_file.endswith(".original"):
        output_file = output_file.replace(".original", "")
    return output_file.replace(".bc", ".o")


def image_output_file(bitcode_file: str) -> str:
    """
    Strip the '.original' suffix from the IR bitcode file path if it
    exists and replace it with '.image'.
    """
    output_file = bitcode_file
    if output_file.endswith(".original"):
        output_file = output_file.replace(".original", "")
    return output_file.replace(".bc", ".image")


def cleanup_modified_files(bc_path: str) -> None:
    """
    Restore the original .image and .bc files. If kernel recording
    files ending with 'bc.original' and '.image.original' exist,
    overwrite the
    """
    if bc_path.endswith(".bc.original"):
        bc_original_path = bc_path
        bc_originalless_path = bc_path.replace(".original", "")
    elif bc_path.endswith(".bc"):
        bc_original_path = bc_path + ".original"
        bc_originalless_path = bc_path
    else:
        print(
            f"Error: path must end with .bc(.original). Path: {bc_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    image_originalless_path = image_output_file(bc_path)
    image_original_path = image_originalless_path + ".original"

    if os.path.exists(bc_original_path):
        print(
            f"Cleaning up the file at {bc_originalless_path}, restoring from {bc_original_path}"
        )
        os.rename(bc_original_path, bc_originalless_path)
    else:
        print(
            f"Warning: couldn't find the file {bc_original_path}. If this is not your first run, make sure that your bitcode file is original.",
            file=sys.stderr,
        )

    if os.path.exists(image_original_path):
        os.rename(image_original_path, image_originalless_path)
        print(
            f"Cleaning up the file at {image_originalless_path}, restoring from {image_original_path}"
        )
    else:
        print(
            f"Warning: couldn't find the file {bc_original_path}. If this is not your first run, make sure that your image file is original.",
            file=sys.stderr,
        )


def run_opt(
    bitcode_file: str,
    pipeline_name: str,
    pipeline_config: PipelineConfType,
    dry_run: bool,
    verbose: bool,
) -> int:
    """
    Run `opt` on bitcode_file, reading the flags and custom pipeline
    data from pipeline_config. If dry_run is True, output a stub
    message and exit. If verbose is True, pass '-debug-pass-manager'
    to `opt`.
    """
    if dry_run:
        print(
            f"Simulating run of pipeline {pipeline_name} for bitcode file {bitcode_file}."
        )
        return 0

    print(f"{'=' * 80}")
    print(f"Running pipeline: {pipeline_name}")
    print(f"{'=' * 80}")

    output_name = opt_output_file(bitcode_file)
    opt_passes = pipeline_config["opt_passes"]
    opt_args = pipeline_config["opt_args"]

    # Using any() here makes sure that arrays with a single empty
    # string are treated as empty.
    if not any(opt_passes) and not any(opt_args):
        print(
            f"Opt passes and args not specified for pipeline {pipeline_name}!",
            file=sys.stderr,
        )
        return 2

    if not bitcode_file.endswith(".original"):
        print(
            f"Bitcode file {bitcode_file} does not have the .original extension, exiting to prevent accidental reuse of an already optimized bitcode",
            file=sys.stderr,
        )
        return 3

    cmd = ["opt"]
    if opt_passes:
        cmd.append(f"-passes={opt_passes}")
    if verbose:
        cmd.append("-debug-pass-manager")
    cmd.append("-o")
    cmd.append(output_name)
    cmd = cmd + opt_args
    cmd.append(bitcode_file)

    print("Running " + " ".join(cmd))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return result.returncode
    except FileNotFoundError:
        print(
            "Error: 'opt' command not found. Make sure LLVM is installed and in PATH.",
            file=sys.stderr,
        )
        return 1
    except Exception as e:
        print(f"Error running pipeline: {e}", file=sys.stderr)
        return 1


def run_llc(
    bitcode_file: str,
    pipeline_name: str,
    pipeline_config: PipelineConfType,
    dry_run: bool,
    also_opt: bool,
    arch: str,
    verbose: bool,
) -> int:
    """
    Run `llc` on bitcode_file, reading the flags from pipeline_config.
    If dry_run is True, output a stub message and exit. If also_opt is
    True, pass the bitcode file through `opt` first by invoking
    run_opt(). The arch argument is necessary. If verbose is True,
    pass '-debug-pass-manager' to `opt`.
    """
    if dry_run:
        print(
            f"Simulating run of backend pipeline {pipeline_name} for bitcode file {bitcode_file}."
        )
        return 0
    if also_opt:
        print(f"Running backend pipeline: {pipeline_name} with opt first")
    else:
        print(f"{'=' * 80}")
        print(f"Running backend pipeline: {pipeline_name}")
        print(f"{'=' * 80}")
    if not bitcode_file.endswith(".original"):
        print(
            f"Bitcode file {bitcode_file} does not have the .original extension, use the --llc-also-use-opt flag if you want to pass an already modified bitcode file to llc"
        )
        return 3
    if also_opt:
        status = run_opt(bitcode_file, pipeline_name, pipeline_config, dry_run, verbose)
        if status > 0:
            return status

    llc_output_name = llc_output_file(bitcode_file)
    image_output_name = image_output_file(bitcode_file)
    llc_args = pipeline_config["llc_args"]

    if not any(llc_args):
        print(f"llc args not specified for pipeline {pipeline_name}!", file=sys.stderr)
        return 2

    cmd = ["llc", "-mtriple=amdgcn-amd-amdhsa", "-filetype=obj"]
    if verbose:
        cmd.append("-debug-pass-manager")
    cmd.append(f"-mcpu={arch}")
    cmd.append("-o")
    cmd.append(llc_output_name)
    cmd = cmd + llc_args
    if also_opt:
        cmd.append(opt_output_file(bitcode_file))
    else:
        cmd.append(bitcode_file)

    link_cmd = ["ld.lld", "-flavor", "gnu", "-shared"]
    link_cmd.append("-o")
    link_cmd.append(image_output_name)
    link_cmd.append(llc_output_name)

    print("Running " + " ".join(cmd))

    try:
        # Run llc
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.returncode > 0:
            return result.returncode
        # Run the linker
        print("...and " + " ".join(link_cmd))
        link_result = subprocess.run(link_cmd, capture_output=True, text=True)
        print(link_result.stdout)
        if link_result.stderr:
            print(link_result.stderr, file=sys.stderr)
        return link_result.returncode
    except FileNotFoundError as e:
        print(
            "Error: 'llc' or 'ld.lld' command not found. Make sure LLVM is installed and in PATH.",
            file=sys.stderr,
        )
        print(e)
        return 1
    except Exception as e:
        print(f"Error running pipeline: {e}", file=sys.stderr)
        return 1


def run_kernel_replay(
    json_path: str,
    pipeline_config: PipelineConfType,
    rp_output_dir: str,
    run_bitcode: bool,
    dry_run: bool,
) -> int:
    """
    Run the LLVM OpenMP kernel replay tool through rocprofv3, using
    the JSON file at json_path and specifying the profiler output
    directory rp_output_dir. If run_bitcode is true, pass the
    '--load-bitcode' flag to the tool to make it JIT-compile the IR
    bitcode file before replaying (necessary when testing opt). If
    dry_run is True, output a stub message and exit.
    """
    if dry_run:
        print(f"Simulating dry run for kernel replay for JSON file at {json_path}.")
        return 0

    cmd = [
        "rocprofv3",
        "--stats",
        "--kernel-trace",
        "--output-directory",
        rp_output_dir,
        "--output-file",
        "output",
        "--",
        "llvm-omp-kernel-replay",
    ]

    if run_bitcode:
        cmd.append("--load-bitcode")
    cmd = cmd + pipeline_config["replay_args"]
    cmd.append(json_path)

    print("Running " + " ".join(cmd))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return result.returncode
    except FileNotFoundError:
        print(
            "Error: 'llvm-omp-kernel-replay' or 'rocprofv3' command not found. Make sure LLVM is installed and in PATH.",
            file=sys.stderr,
        )
        return 1
    except Exception as e:
        print(f"Error running pipeline: {e}", file=sys.stderr)
        return 1


def get_rocprof_output_dir(pipeline_name: str) -> str:
    """
    Helper function to format a folder name for a rocprofv3 output
    directory.
    """
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    return f"{now}-{pipeline_name}"


def profiler_csv_reader(
    rp_output_dir: str,
) -> Generator[dict[str, str | float], str, None] | None:
    """
    Yield a generator that reads line by line from the kernel trace CSV file
    located at rp_output_dir. The kernel trace CSV file contains
    information about one kernel launch per line.
    """
    csv_path = f"./{rp_output_dir}/output_kernel_trace.csv"
    try:
        with open(csv_path, mode="r", newline="", encoding="utf-8") as kernel_trace_csv:
            kt_reader = csv.DictReader(kernel_trace_csv, quoting=csv.QUOTE_NONNUMERIC)
            yield from kt_reader
    except OSError as e:
        print(f"Error reading CSV results from {rp_output_dir}: {e}", file=sys.stderr)


def replay_and_measure_kernel(
    json_path: str,
    pipeline_config: PipelineConfType,
    output_dir: str,
    run_bitcode: bool,
    dry_run: bool,
) -> list[float] | None:
    """
    Replay the specified kernel via calling run_kernel_replay and
    return a list that contains kernel runtimes in nanoseconds.
    """
    if not dry_run:
        code = run_kernel_replay(
            json_path, pipeline_config, output_dir, run_bitcode, dry_run
        )
        if code != 0:
            sys.exit(code)
        csv_reader = profiler_csv_reader(output_dir)
        runtimes: list[float] = []
        if csv_reader:
            for row in csv_reader:
                end_ts = row["End_Timestamp"]
                start_ts = row["Start_Timestamp"]
                if isinstance(end_ts, float) and isinstance(start_ts, float):
                    runtime = end_ts - start_ts
                    runtimes.append(runtime)
                else:
                    print(
                        f"'End_Timestamp' or 'Start_Timestamp' fields in {output_dir} are not floats, exiting"
                    )
                    sys.exit(1)
        return runtimes


def write_results(results: list[dict[str, str | float]]) -> None:
    """
    Output results as a CSV file.
    """
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"results-{now}.csv"
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ["pipeline", "runtime_ns"]
        writer = csv.DictWriter(
            csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Successfully wrote results to {csv_path}")


def write_boxplots(results: list[dict[str, str | float]]) -> None:
    """
    Output results as boxplot image.
    """
    # Convert results to be associative
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    png_path = f"results-{now}.png"

    results_assoc = defaultdict(list)
    for row in results:
        results_assoc[row["pipeline"]].append(row["runtime_ns"])

    data = list(results_assoc.values())
    plt.figure(figsize=(14, 6))
    plt.boxplot(data, tick_labels=list(results_assoc.keys()))
    plt.xticks(rotation=45, ha="right")
    plt.title("Kernel runtime")
    plt.savefig(png_path, dpi=400, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLVM opt or llc with various pass pipelines on a bitcode file."
    )
    parser.add_argument("bitcode_file", help="Path to the bitcode file (.bc)")
    parser.add_argument(
        "--pipelines-file",
        default="opt-passes.json",
        help="Path to a JSON file containing pipeline configurations (default: ./opt-passes.json)",
    )
    parser.add_argument(
        "--pipeline",
        help="Specific pipeline to run (if not specified, runs all pipelines)",
    )
    parser.add_argument(
        "--dry-run",
        help="Does not actually invoke opt, replay or other tools, useful for testing",
        action="store_true",
    )
    parser.add_argument(
        "--arch",
        help="GPU Architecture (default: gfx90a), only used for --llc",
        default="gfx90a",
    )
    parser.add_argument(
        "--verbose",
        help="Use the -debug-pass-manager flag when running opt",
        action="store_true",
    )
    parser.add_argument(
        "--llc",
        help="Use the llc-based backend optimization instead of opt",
        action="store_true",
    )
    parser.add_argument(
        "--llc-also-use-opt",
        help="When doing backend optimization, also use opt first before running llc",
        action="store_true",
    )
    parser.add_argument(
        "--no-plot",
        default=False,
        help="Don't plot the results as a PNG image",
        action="store_true",
    )

    args = parser.parse_args()
    json_path = check_record_json_file(args.bitcode_file)
    results = []
    pipelines = load_pipelines(args.pipelines_file)
    also_opt = args.llc_also_use_opt
    arch = args.arch

    original_bitcode_file = get_original_bitcode(args.bitcode_file)
    backup_image(image_output_file(args.bitcode_file))

    if args.pipeline:
        pipelines_to_run = [(args.pipeline, pipelines[args.pipeline])]
    else:
        pipelines_to_run = list(pipelines.items())

    run_bitcode = not args.llc

    for pipeline_name, pipeline_config in pipelines_to_run:
        if args.llc:
            returncode = run_llc(
                original_bitcode_file,
                pipeline_name,
                pipeline_config,
                args.dry_run,
                also_opt,
                arch,
                args.verbose,
            )
        else:
            returncode = run_opt(
                original_bitcode_file,
                pipeline_name,
                pipeline_config,
                args.dry_run,
                args.verbose,
            )
        if returncode != 0:
            print(
                f"Warning: Pipeline '{pipeline_name}' exited with code {returncode}",
                file=sys.stderr,
            )
            continue
        rp_output_dir = get_rocprof_output_dir(pipeline_name)
        runtimes = replay_and_measure_kernel(
            json_path, pipeline_config, rp_output_dir, run_bitcode, args.dry_run
        )
        if runtimes:
            for runtime in runtimes:
                runtime_dict = {"pipeline": pipeline_name, "runtime_ns": runtime}
                results.append(runtime_dict)

    if not args.dry_run and returncode == 0:
        write_results(results)
        if not args.no_plot:
            write_boxplots(results)


if __name__ == "__main__":
    main()
