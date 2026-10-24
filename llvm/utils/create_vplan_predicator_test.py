#!/usr/bin/env python3
"""Generate an alloca-based loop input for VPlan predicator experiments.

The generated loop contains N numbered forward body blocks. Every body block
has one mandatory successor which keeps the whole chain reachable and, when
possible, a random second forward successor. Branches selected as uniform use
function arguments as their conditions. Varying branches use a small integer
hash of the loop induction variable, producing a less regular lane pattern. The
``--uniform-only`` option restricts conditional branches to uniform branches.

Each body block computes ``%addN = add i64 %iv, %aN``, adds it to the
current value in one entry-block alloca, and stores the accumulated value back.
The loop latch loads that alloca and stores the value to an output array. This
intentionally leaves PHI/SSA construction to mem2reg; accumulating is
important so that stores in earlier blocks are not dead.

The generated module contains one scalar-memory function, @test, plus a
small lli-compatible harness. The RUN lines exercise the unoptimized input,
mem2reg, the mem2reg vectorizer pipeline, and the test-only fake-privatize
input-preparation vectorizer pipeline. The vectorized RUN lines check that
@test contains a vector loop.
"""

from __future__ import annotations

import argparse
import random
import sys
import uuid
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

TRIP_COUNT = 128


@dataclass(frozen=True)
class Block:
    index: int
    successors: List[int]
    branch_kind: Optional[str]
    condition_name: Optional[str]
    varying_salt: Optional[int]
    varying_threshold: Optional[int]


@dataclass(frozen=True)
class Layout:
    first_body: int
    latch: int
    exit: int


def make_layout(num_blocks: int) -> Layout:
    # Body blocks are [1, N]. The latch and exit are outside that range and
    # can therefore be used as forward destinations.
    return Layout(first_body=1, latch=num_blocks + 1, exit=num_blocks + 2)


def choose_successors(
    num_blocks: int,
    rng: random.Random,
    layout: Layout,
    uniform_only: bool = False,
) -> List[Block]:
    blocks: List[Block] = []

    for index in range(1, num_blocks + 1):
        mandatory = index + 1 if index < num_blocks else layout.latch

        # Every numbered block is reachable through the mandatory chain. Any
        # optional edge is strictly forward and distinct from that edge.
        successors = [mandatory]
        if index < num_blocks and rng.choice((False, True)):
            successors.append(rng.randint(index + 2, layout.latch))

        if len(successors) == 1:
            blocks.append(Block(index, successors, None, None, None, None))
            continue

        condition_name = (
            f"u{index}" if uniform_only or rng.choice((False, True)) else f"vc{index}"
        )
        if condition_name.startswith("u"):
            blocks.append(
                Block(index, successors, "uniform", condition_name, None, None)
            )
            continue

        # The hash is evaluated over 32 bits after a 64-bit multiply. The
        # threshold controls the approximate true probability. Restricting it
        # to this range avoids intentionally constant-looking conditions while
        # still covering sparse and dense cases.
        salt = rng.randrange(1, 1 << 32)
        threshold = rng.randint(1, (1 << 32) - 1)
        blocks.append(
            Block(index, successors, "varying", condition_name, salt, threshold)
        )

    return blocks


def expected_results(num_blocks: int, blocks: Sequence[Block]) -> List[int]:
    """Evaluate the generated scalar CFG for the harness's fixed arguments."""
    layout = make_layout(num_blocks)
    uniform_values = {
        block.condition_name: bool(index % 2)
        for index, block in enumerate(
            (block for block in blocks if block.branch_kind == "uniform"), 1
        )
    }
    block_by_index = {block.index: block for block in blocks}
    arguments = {index: 17 * index - 53 for index in range(1, num_blocks + 1)}
    mask = (1 << 64) - 1

    def as_i64(value: int) -> int:
        value &= mask
        return value - (1 << 64) if value & (1 << 63) else value

    results = []
    for iv in range(TRIP_COUNT):
        current = 0
        index = 1
        while index != layout.latch:
            block = block_by_index[index]
            current = as_i64(current + iv + arguments[index])
            if len(block.successors) == 1:
                index = block.successors[0]
                continue

            if block.branch_kind == "uniform":
                condition = uniform_values[block.condition_name]
            else:
                x = (iv ^ block.varying_salt) & mask
                y = (x * 6364136223846793005) & mask
                z = y >> 32
                condition = z < block.varying_threshold
            index = block.successors[0] if condition else block.successors[1]
        results.append(current)
    return results


def make_block_labels(blocks: Sequence[Block]) -> Dict[int, str]:
    labels = {}
    for block in blocks:
        suffix = {"uniform": "_u", "varying": "_v"}.get(block.branch_kind, "")
        labels[block.index] = f"bb{block.index}{suffix}"
    return labels


def label(index: int, layout: Layout, block_labels: Mapping[int, str]) -> str:
    if index <= layout.latch - 1:
        return block_labels[index]
    if index == layout.latch:
        return "loop.latch"
    if index == layout.exit:
        return "exit"
    raise ValueError(f"unknown block index {index}")


def function_arguments(num_blocks: int, blocks: Sequence[Block]) -> List[str]:
    uniform_args = [
        block.condition_name for block in blocks if block.branch_kind == "uniform"
    ]
    return (
        ["ptr noalias %p"]
        + [f"i64 %a{index}" for index in range(1, num_blocks + 1)]
        + [f"i1 %{name}" for name in uniform_args]
    )


def function_signature(
    name: str, num_blocks: int, blocks: Sequence[Block], attributes: str = ""
) -> str:
    suffix = f" {attributes}" if attributes else ""
    args = ", ".join(function_arguments(num_blocks, blocks))
    return f"define void @{name}({args}){suffix} {{"


def emit_branch(
    successors: Sequence[int],
    condition: Optional[str],
    layout: Layout,
    block_labels: Mapping[int, str],
) -> str:
    if len(successors) == 1:
        return f"  br label %{label(successors[0], layout, block_labels)}"
    assert condition is not None
    return (
        f"  br i1 %{condition}, "
        f"label %{label(successors[0], layout, block_labels)}, "
        f"label %{label(successors[1], layout, block_labels)}"
    )


def emit_body(
    num_blocks: int,
    blocks: Sequence[Block],
    name: str,
    attributes: str = "",
) -> List[str]:
    layout = make_layout(num_blocks)
    block_labels = make_block_labels(blocks)
    lines = [
        function_signature(name, num_blocks, blocks, attributes),
        "entry:",
        "  %slot = alloca i64, align 8",
        "  br label %loop.header",
        "",
        "loop.header:",
        "  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]",
        "  store i64 0, ptr %slot, align 8",
        f"  br label %{label(layout.first_body, layout, block_labels)}",
        "",
    ]

    for block in blocks:
        block_label = label(block.index, layout, block_labels)
        lines.extend(
            [
                f"{block_label}:",
                f"  %add{block.index} = add i64 %iv, %a{block.index}",
                f"  %old{block.index} = load i64, ptr %slot, align 8",
                f"  %sum{block.index} = add i64 %old{block.index}, %add{block.index}",
                f"  store i64 %sum{block.index}, ptr %slot, align 8",
            ]
        )

        if block.branch_kind == "varying":
            assert block.varying_salt is not None
            assert block.varying_threshold is not None
            lines.extend(
                [
                    f"  %{block.condition_name}.x = xor i64 %iv, {block.varying_salt}",
                    f"  %{block.condition_name}.y = mul i64 %{block.condition_name}.x, 6364136223846793005",
                    f"  %{block.condition_name}.z = lshr i64 %{block.condition_name}.y, 32",
                    f"  %{block.condition_name} = icmp ult i64 %{block.condition_name}.z, {block.varying_threshold}",
                ]
            )

        lines.append(
            emit_branch(block.successors, block.condition_name, layout, block_labels)
        )
        lines.append("")

    lines.extend(
        [
            "loop.latch:",
            "  %value = load i64, ptr %slot, align 8",
            "  %gep = getelementptr i64, ptr %p, i64 %iv",
            "  store i64 %value, ptr %gep, align 8",
            "  %iv.next = add nuw nsw i64 %iv, 1",
            f"  %done = icmp eq i64 %iv.next, {TRIP_COUNT}",
            "  br i1 %done, label %exit, label %loop.header",
            "",
            "exit:",
            "  ret void",
            "}",
            "",
        ]
    )
    return lines


def emit_header(
    num_blocks: int,
    seed: str,
    blocks: Sequence[Block],
    uniform_only: bool = False,
) -> List[str]:
    command = f"{num_blocks} --seed={seed}"
    if uniform_only:
        command += " --uniform-only"
    lines = [
        "; NOTE: Generated by llvm/utils/generate_vplan_predicator_input.py "
        f"{command}",
    ]
    lines.extend(
        [
            "; RUN: lli %s",
            "; RUN: opt -p=mem2reg %s | lli -",
            "; RUN: opt -S -p=mem2reg,loop-vectorize "
            "-force-vector-width=4 -force-vector-interleave=1 %s | FileCheck %s",
            "; RUN: opt -p=mem2reg,loop-vectorize "
            "-force-vector-width=4 -force-vector-interleave=1 %s | lli -",
            "; RUN: opt -S -p=fake-privatize,loop-vectorize "
            "-force-vector-width=4 -force-vector-interleave=1 %s | FileCheck %s",
            "; RUN: opt -p=fake-privatize,loop-vectorize "
            "-force-vector-width=4 -force-vector-interleave=1 %s | lli -",
            "; CHECK-LABEL: define void @test(",
            "; CHECK: vector.body:",
        ]
    )
    lines.append("; successor choices:")
    layout = make_layout(num_blocks)
    block_labels = make_block_labels(blocks)
    for block in blocks:
        kind = block.branch_kind or "unconditional"
        details = ""
        if block.condition_name:
            details = f", condition={block.condition_name}"
        if block.branch_kind == "varying":
            details += (
                f", salt={block.varying_salt}, threshold={block.varying_threshold}"
            )
        successors = ", ".join(
            label(successor, layout, block_labels) for successor in block.successors
        )
        lines.append(
            f";   {label(block.index, layout, block_labels)} -> "
            f"{successors} ({kind}{details})"
        )
    lines.append("")
    return lines


def emit_input_mem() -> List[str]:
    # Keep this externally visible and unused. It provides a convenient named
    # memory-input hook for experiments without affecting the generated loop.
    return [
        "define void @input_mem(ptr %p) noinline {",
        "entry:",
        "  store i64 0, ptr %p, align 8",
        "  ret void",
        "}",
        "",
    ]


def emit_test_harness(
    num_blocks: int, blocks: Sequence[Block], test_name: str
) -> List[str]:
    # Turn the function arguments into concrete constants for the two calls
    # from main, using different output buffers.
    value_args = [str(17 * index - 53) for index in range(1, num_blocks + 1)]
    uniform_args = [
        "true" if index % 2 else "false"
        for index, block in enumerate(
            (block for block in blocks if block.branch_kind == "uniform"), 1
        )
    ]

    def call_arguments(pointer: str) -> str:
        arguments = [f"ptr {pointer}"]
        arguments += [f"i64 {value}" for value in value_args]
        arguments += [f"i1 {value}" for value in uniform_args]
        return ", ".join(arguments)

    selected_call = call_arguments("%selected.ptr")
    expected = expected_results(num_blocks, blocks)
    expected_values = ", ".join(f"i64 {value}" for value in expected)

    lines = [
        f"@expected = private constant [128 x i64] [{expected_values}]",
        "",
        "define void @input_mem(ptr %p) noinline {",
        "entry:",
        "  store i64 0, ptr %p, align 8",
        "  ret void",
        "}",
        "",
        f"define internal i1 @check_results(ptr %lhs, ptr %rhs) {{",
        "entry:",
        "  br label %check.loop",
        "",
        "check.loop:",
        "  %i = phi i64 [ 0, %entry ], [ %next, %check.body ]",
        "  %done = icmp eq i64 %i, 128",
        "  br i1 %done, label %check.ok, label %check.body",
        "",
        "check.body:",
        "  %lhs.ptr = getelementptr i64, ptr %lhs, i64 %i",
        "  %rhs.ptr = getelementptr i64, ptr %rhs, i64 %i",
        "  %lhs.value = load i64, ptr %lhs.ptr, align 8",
        "  %rhs.value = load i64, ptr %rhs.ptr, align 8",
        "  %equal = icmp eq i64 %lhs.value, %rhs.value",
        "  %next = add nuw i64 %i, 1",
        "  br i1 %equal, label %check.loop, label %check.fail",
        "",
        "check.fail:",
        "  ret i1 false",
        "",
        "check.ok:",
        "  ret i1 true",
        "}",
        "",
        "define i32 @main() {",
        "entry:",
        "  %selected = alloca [128 x i64], align 8",
        "  %selected.ptr = getelementptr [128 x i64], ptr %selected, i64 0, i64 0",
        f"  call void @{test_name}({selected_call})",
        "  br label %check",
        "",
        "check:",
        "  %expected.ptr = getelementptr [128 x i64], ptr @expected, i64 0, i64 0",
        "  %same = call i1 @check_results(ptr %selected.ptr, ptr %expected.ptr)",
        "  %result = zext i1 %same to i32",
        "  %status = xor i32 %result, 1",
        "  ret i32 %status",
        "}",
        "",
    ]
    return lines


def emit_module(
    num_blocks: int,
    seed: str,
    blocks: Sequence[Block],
    uniform_only: bool = False,
) -> str:
    test_name = "test"
    lines = emit_header(num_blocks, seed, blocks, uniform_only)
    lines.extend(emit_body(num_blocks, blocks, test_name, "noinline"))
    lines.extend(emit_test_harness(num_blocks, blocks, test_name))
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "num_blocks",
        type=int,
        metavar="N",
        help="number of generated numbered blocks inside the loop",
    )
    parser.add_argument(
        "--seed",
        help="RNG seed (default: random UUID); kept as text for reproducibility",
    )
    parser.add_argument(
        "--uniform-only",
        action="store_true",
        help="only generate conditional branches with uniform conditions",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_blocks < 1:
        print("N must be at least 1", file=sys.stderr)
        return 2

    seed = args.seed if args.seed is not None else str(uuid.uuid4())
    rng = random.Random(seed)
    layout = make_layout(args.num_blocks)
    blocks = choose_successors(
        args.num_blocks, rng, layout, uniform_only=args.uniform_only
    )
    module = emit_module(args.num_blocks, seed, blocks, uniform_only=args.uniform_only)
    sys.stdout.write(module)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
