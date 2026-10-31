#!/usr/bin/env python3

"""Report SelectionDAG differences caused by topological DAG combining.

This tool reads LLVM lit tests, extracts llc RUN lines, and runs each supported
line twice: once as written and once with -combiner-topological-sorting=1.  It
captures the SelectionDAG debug dump and reports the first DAG stage where each
function/basic-block pair diverges.
"""

import argparse
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from UpdateTestChecks import asm as asm_utils
from UpdateTestChecks import common


STAGE_HEADER_RE = re.compile(
    r"^(?P<stage>.+?selection DAG): (?P<bb>%bb\.[0-9]+) '(?P<func>.*)'$"
)
FUNCTION_HEADER_RE = re.compile(r"^=== (?P<func>.*)$")
TOPO_FLAG_RE = re.compile(r"(^|\s)-combiner-topological-sorting(?:[=\s][^\s]+)?")
NODE_RE = re.compile(r"^\s*(t[0-9]+):\s*(?P<body>\S.*)$")
DEBUG_ONLY_RE = re.compile(r"(^|\s)-debug-only(?:[= ][^\s]+)?")
STDERR_TO_STDOUT_RE = re.compile(r"\s+2>&1(?=\s|$)")


@dataclass(frozen=True)
class RunCommand:
    index: int
    original: str
    llc_cmd: str
    preprocess_cmd: Optional[str]


@dataclass
class DagStage:
    function: str
    bb: str
    stage: str
    text: str


@dataclass
class InvocationResult:
    command: str
    returncode: int
    output: str


@dataclass(frozen=True)
class NodeChange:
    kind: str
    node_id: str
    old: Optional[str]
    new: Optional[str]


def split_pipeline(run_line: str) -> List[str]:
    # LLVM's update scripts use a simple split here too.  That is good enough
    # for the lit RUN lines this diagnostic tool is intended to consume.
    return [cmd.strip() for cmd in run_line.split("|")]


def first_shell_word(command: str) -> Optional[str]:
    try:
        words = shlex.split(command, posix=True)
    except ValueError:
        return None
    return words[0] if words else None


def extract_llc_command(run_line: str, index: int) -> Optional[RunCommand]:
    if TOPO_FLAG_RE.search(run_line):
        return None

    commands = split_pipeline(run_line)
    llc_pos = None
    for i, command in enumerate(commands):
        word = first_shell_word(command)
        if word in ("llc", "%llc_dwarf"):
            llc_pos = i
            break
        if word == "not":
            return None

    if llc_pos is None:
        return None

    preprocess_cmd = " | ".join(commands[:llc_pos]) or None
    return RunCommand(index, run_line, commands[llc_pos], preprocess_cmd)


def replace_llc_tool(command: str, llc_binary: str) -> str:
    return re.sub(r"^(\s*)(llc|%llc_dwarf)(?=\s|$)", r"\1" + llc_binary, command, 1)


def make_invocation(
    run: RunCommand, test_path: str, llc_binary: str, enable_topo: bool
) -> str:
    substitutions = common.getSubstitutions(test_path)
    llc_cmd = common.applySubstitutions(run.llc_cmd, substitutions)
    llc_cmd = replace_llc_tool(llc_cmd, llc_binary)

    extra_args = ["-debug-only=isel-dump", "-o", os.devnull]
    if enable_topo:
        extra_args.insert(0, "-combiner-topological-sorting=1")
    llc_cmd = llc_cmd + " " + " ".join(shlex.quote(arg) for arg in extra_args)

    if run.preprocess_cmd:
        preprocess = common.applySubstitutions(run.preprocess_cmd, substitutions)
        return preprocess + " | " + llc_cmd
    return llc_cmd


def make_asm_invocation(
    run: RunCommand, test_path: str, llc_binary: str, enable_topo: bool
) -> str:
    substitutions = common.getSubstitutions(test_path)
    llc_cmd = common.applySubstitutions(run.llc_cmd, substitutions)
    llc_cmd = replace_llc_tool(llc_cmd, llc_binary)
    llc_cmd = DEBUG_ONLY_RE.sub(" ", llc_cmd)
    llc_cmd = STDERR_TO_STDOUT_RE.sub("", llc_cmd)

    extra_args = ["-filetype=asm", "-o", "-"]
    if enable_topo:
        extra_args.insert(0, "-combiner-topological-sorting=1")
    llc_cmd = llc_cmd + " " + " ".join(shlex.quote(arg) for arg in extra_args)

    if run.preprocess_cmd:
        preprocess = common.applySubstitutions(run.preprocess_cmd, substitutions)
        return preprocess + " | " + llc_cmd
    return llc_cmd


def run_invocation(command: str) -> InvocationResult:
    proc = subprocess.run(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    return InvocationResult(command, proc.returncode, proc.stdout.replace("\r\n", "\n"))


def run_stdout_invocation(command: str) -> InvocationResult:
    proc = subprocess.run(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    return InvocationResult(command, proc.returncode, proc.stdout.replace("\r\n", "\n"))


def get_ir_triple(test_path: str) -> Optional[str]:
    with open(test_path, encoding="utf-8") as handle:
        for line in handle:
            match = common.TRIPLE_IR_RE.match(line)
            if match:
                return match.group(1)
    return None


def get_run_triple(test_path: str, run: RunCommand) -> str:
    triple_match = common.TRIPLE_ARG_RE.search(run.llc_cmd)
    if triple_match:
        return triple_match.group(1)
    ir_triple = get_ir_triple(test_path)
    if ir_triple:
        return ir_triple
    march_match = common.MARCH_ARG_RE.search(run.llc_cmd)
    if march_match:
        return common.get_triple_from_march(march_match.group(1))
    return "x86"


def normalize_asm_body(body: str) -> str:
    lines = [line.rstrip() for line in body.splitlines()]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def parse_asm_functions(raw_asm: str, triple: str) -> Optional[Dict[str, str]]:
    try:
        scrubber, function_re = asm_utils.get_run_handler(triple)
    except KeyError:
        return None

    scrub_args = type(
        "ScrubArgs",
        (object,),
        {
            "no_x86_scrub_mem_shuffle": True,
            "x86_scrub_sp": True,
            "x86_scrub_rip": False,
            "extra_scrub": False,
        },
    )()
    functions = {}
    for match in function_re.finditer(raw_asm):
        body = scrubber(match.group("body"), scrub_args)
        functions[match.group("func")] = normalize_asm_body(body)
    return functions if functions else None


def get_changed_asm_functions(
    test_path: str, run: RunCommand, llc_binary: str
) -> Optional[set]:
    baseline_command = make_asm_invocation(run, test_path, llc_binary, False)
    topo_command = make_asm_invocation(run, test_path, llc_binary, True)
    baseline_result = run_stdout_invocation(baseline_command)
    topo_result = run_stdout_invocation(topo_command)
    if baseline_result.returncode != 0 or topo_result.returncode != 0:
        return None

    triple = get_run_triple(test_path, run)
    baseline_functions = parse_asm_functions(baseline_result.output, triple)
    topo_functions = parse_asm_functions(topo_result.output, triple)
    if baseline_functions is None or topo_functions is None:
        return None

    changed = set()
    for function in set(baseline_functions) | set(topo_functions):
        if baseline_functions.get(function) != topo_functions.get(function):
            changed.add(function)
    return changed


def collect_stage(
    lines: Sequence[str], start: int, function_hint: Optional[str]
) -> Tuple[Optional[DagStage], int]:
    match = STAGE_HEADER_RE.match(lines[start])
    if not match:
        return None, start + 1

    end = start + 1
    while end < len(lines):
        if STAGE_HEADER_RE.match(lines[end]) or FUNCTION_HEADER_RE.match(lines[end]):
            break
        if lines[end].startswith("*** MachineFunction at end of ISel ***"):
            break
        end += 1

    function = match.group("func")
    if function.endswith(":"):
        function = function[:-1]
    if not function and function_hint:
        function = function_hint

    text = "\n".join(lines[start:end]).rstrip() + "\n"
    return (
        DagStage(
            function=function,
            bb=match.group("bb"),
            stage=match.group("stage"),
            text=text,
        ),
        end,
    )


def parse_dag_stages(output: str) -> Dict[Tuple[str, str], List[DagStage]]:
    stages: Dict[Tuple[str, str], List[DagStage]] = {}
    lines = output.splitlines()
    current_function = None
    i = 0
    while i < len(lines):
        function_match = FUNCTION_HEADER_RE.match(lines[i])
        if function_match:
            current_function = function_match.group("func")
            i += 1
            continue

        stage, next_i = collect_stage(lines, i, current_function)
        if stage is None:
            i = next_i
            continue

        stages.setdefault((stage.function, stage.bb), []).append(stage)
        i = next_i
    return stages


def node_sort_key(node: str) -> int:
    return int(node[1:])


def get_stage_nodes(stage: Optional[DagStage]) -> Dict[str, str]:
    if stage is None:
        return {}
    nodes = {}
    for line in stage.text.splitlines():
        match = NODE_RE.match(line)
        if match:
            nodes[match.group(1)] = line.strip()
    return nodes


def stages_match(left: DagStage, right: DagStage) -> bool:
    return left.stage == right.stage and get_stage_nodes(left) == get_stage_nodes(right)


def find_first_divergence(
    baseline: Sequence[DagStage], topo: Sequence[DagStage]
) -> Optional[Tuple[int, Optional[DagStage], Optional[DagStage], Optional[DagStage]]]:
    count = min(len(baseline), len(topo))
    previous = None
    for i in range(count):
        left = baseline[i]
        right = topo[i]
        if not stages_match(left, right):
            return (i, previous, left, right)
        previous = left

    if len(baseline) != len(topo):
        left = baseline[count] if count < len(baseline) else None
        right = topo[count] if count < len(topo) else None
        return (count, previous, left, right)
    return None


def indent(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line else prefix for line in text.splitlines())


def stage_title(stage: Optional[DagStage]) -> str:
    return stage.stage if stage else "<missing stage>"


def get_transition_changes(
    previous_stage: Optional[DagStage], next_stage: Optional[DagStage]
) -> List[NodeChange]:
    previous_nodes = get_stage_nodes(previous_stage)
    next_nodes = get_stage_nodes(next_stage)
    previous_ids = set(previous_nodes)
    next_ids = set(next_nodes)
    changed_ids = {
        node_id
        for node_id in previous_ids & next_ids
        if previous_nodes[node_id] != next_nodes[node_id]
    }
    interesting_ids = sorted(
        (previous_ids ^ next_ids) | changed_ids,
        key=node_sort_key,
    )

    changes = []
    for node_id in interesting_ids:
        if node_id not in next_nodes:
            changes.append(NodeChange("removed", node_id, previous_nodes[node_id], None))
        elif node_id not in previous_nodes:
            changes.append(NodeChange("added", node_id, None, next_nodes[node_id]))
        else:
            changes.append(
                NodeChange(
                    "changed", node_id, previous_nodes[node_id], next_nodes[node_id]
                )
            )
    return changes


def change_sort_key(change: NodeChange) -> int:
    return node_sort_key(change.node_id)


def report_transition_changes(
    out: List[str], title: str, changes: Sequence[NodeChange]
) -> None:
    out.append(title)
    if not changes:
        out.append("  <none>")
        return

    for change in sorted(changes, key=change_sort_key):
        if change.kind == "removed":
            out.append(f"  removed {change.node_id}: {change.old}")
        elif change.kind == "added":
            out.append(f"  added {change.node_id}: {change.new}")
        else:
            out.append(f"  changed {change.node_id}:")
            out.append(f"    from: {change.old}")
            out.append(f"      to: {change.new}")


def report_stage_changes(
    out: List[str],
    previous_stage: Optional[DagStage],
    baseline_stage: Optional[DagStage],
    topo_stage: Optional[DagStage],
) -> None:
    baseline_changes = set(get_transition_changes(previous_stage, baseline_stage))
    topo_changes = set(get_transition_changes(previous_stage, topo_stage))

    out.append("=== Changes From Previous Matching DAG ===")
    report_transition_changes(
        out,
        "Baseline-only stage changes:",
        sorted(baseline_changes - topo_changes, key=change_sort_key),
    )
    out.append("")
    report_transition_changes(
        out,
        "Topological-sort-only stage changes:",
        sorted(topo_changes - baseline_changes, key=change_sort_key),
    )


def print_stage(out: List[str], title: str, stage: Optional[DagStage]) -> None:
    out.append(title)
    if stage is None:
        out.append("  <missing>")
        return
    out.append(indent(stage.text.rstrip(), 2))


def append_run_context(out: List[str], test_path: str, run: RunCommand) -> None:
    out.append(f"File: {test_path}")
    out.append(f"RUN {run.index}:")
    out.append(f"  {run.original}")


def report_divergence(
    out: List[str],
    test_path: str,
    run: RunCommand,
    key: Tuple[str, str],
    divergence: Tuple[int, Optional[DagStage], Optional[DagStage], Optional[DagStage]],
) -> None:
    function, bb = key
    _, previous, baseline_stage, topo_stage = divergence
    if out:
        out.append("")
    append_run_context(out, test_path, run)
    out.append(f"Function: {function}")
    out.append(f"Basic block: {bb}")
    out.append(f"First divergent stage: {stage_title(baseline_stage or topo_stage)}")
    if previous:
        out.append(f"Previous matching stage: {previous.stage}")
    else:
        out.append("Previous matching stage: <none>")
    out.append("")

    print_stage(out, "=== Previous Matching DAG ===", previous)
    out.append("")
    print_stage(out, "=== Baseline DAG ===", baseline_stage)
    out.append("")
    print_stage(out, "=== Topological-Sort DAG ===", topo_stage)
    out.append("")
    report_stage_changes(out, previous, baseline_stage, topo_stage)
    out.append("")
    out.append("-" * 80)


def compare_run(
    test_path: str, run: RunCommand, llc_binary: str, verbose: bool
) -> Tuple[int, str]:
    baseline_command = make_invocation(run, test_path, llc_binary, False)
    topo_command = make_invocation(run, test_path, llc_binary, True)

    baseline_result = run_invocation(baseline_command)
    topo_result = run_invocation(topo_command)

    out: List[str] = []

    if verbose:
        append_run_context(out, test_path, run)
        out.append("  Baseline command:")
        out.append(f"    {baseline_command}")
        out.append("  Topological-sort command:")
        out.append(f"    {topo_command}")

    if baseline_result.returncode != 0 or topo_result.returncode != 0:
        if not out:
            append_run_context(out, test_path, run)
        out.append("  ERROR: llc invocation failed")
        out.append(f"  Baseline exit code: {baseline_result.returncode}")
        out.append(f"  Topological-sort exit code: {topo_result.returncode}")
        if baseline_result.output:
            out.append("  Baseline output:")
            out.append(indent(baseline_result.output.rstrip(), 4))
        if topo_result.output:
            out.append("  Topological-sort output:")
            out.append(indent(topo_result.output.rstrip(), 4))
        return 2, "\n".join(out)

    baseline_stages = parse_dag_stages(baseline_result.output)
    topo_stages = parse_dag_stages(topo_result.output)
    changed_asm_functions = get_changed_asm_functions(test_path, run, llc_binary)

    if not baseline_stages and not topo_stages:
        if not out:
            append_run_context(out, test_path, run)
        out.append("  WARNING: no SelectionDAG stages found")
        return 0, "\n".join(out)

    differences = 0
    all_keys = sorted(set(baseline_stages) | set(topo_stages))
    for key in all_keys:
        function, _ = key
        if changed_asm_functions is not None and function not in changed_asm_functions:
            continue
        divergence = find_first_divergence(
            baseline_stages.get(key, []), topo_stages.get(key, [])
        )
        if divergence is not None:
            differences += 1
            report_divergence(out, test_path, run, key, divergence)

    if differences == 0:
        if not out:
            append_run_context(out, test_path, run)
        out.append("  No DAG differences found with changed assembly.")
    return (1 if differences else 0), "\n".join(out)


def iter_run_commands(test_path: str) -> Iterable[RunCommand]:
    with open(test_path, encoding="utf-8") as handle:
        lines = handle.readlines()
    for i, run_line in enumerate(common.find_run_lines(test_path, lines), start=1):
        run = extract_llc_command(run_line, i)
        if run is not None:
            yield run


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Report SelectionDAG dump differences caused by "
            "-combiner-topological-sorting=1."
        )
    )
    parser.add_argument(
        "--llc-binary",
        default="llc",
        help="llc binary to run (default: llc)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print synthesized commands"
    )
    parser.add_argument("tests", nargs="+", help="LLVM lit tests to inspect")
    args = parser.parse_args(argv)

    exit_code = 0
    reports: List[str] = []
    for test_path in args.tests:
        runs = list(iter_run_commands(test_path))
        if not runs:
            reports.append(f"File: {test_path}")
            reports.append("  No supported llc RUN lines found.")
            continue

        for run in runs:
            status, report = compare_run(
                test_path, run, args.llc_binary, args.verbose
            )
            reports.append(report)
            if status == 2:
                exit_code = 2
            elif status == 1 and exit_code == 0:
                exit_code = 1
        reports.append("")

    print("\n".join(reports).rstrip())
    return exit_code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
