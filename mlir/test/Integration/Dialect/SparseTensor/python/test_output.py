# RUN: env SUPPORT_LIB=%mlir_c_runner_utils \
# RUN:   %PYTHON %s | FileCheck %s

import ctypes
import os
import sys
import tempfile

from mlir import ir
from mlir import runtime as rt
from mlir.dialects import builtin
from mlir.dialects import sparse_tensor as st

_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SCRIPT_PATH)
from tools import sparsifier


def boilerplate(attr: st.EncodingAttr):
    """Returns boilerplate main method."""
    return f"""
func.func @main(%p : !llvm.ptr) -> () attributes {{ llvm.emit_c_interface }} {{
  %d = arith.constant sparse<[[0, 0], [1, 1], [0, 9], [9, 0], [4, 4]],
                             [1.0, 2.0, 3.0, 4.0, 5.0]> : tensor<10x10xf64>
  %a = sparse_tensor.convert %d : tensor<10x10xf64> to tensor<10x10xf64, {attr}>
  sparse_tensor.out %a, %p : tensor<10x10xf64, {attr}>, !llvm.ptr
  return
}}
"""


def expected(id_map):
    """Returns expected contents of output.

    +-----+-----+-----+-----+-----+
    | 1 0 | . . | . . | . . | 0 3 |
    | 0 2 | . . | . . | . . | 0 0 |
    +-----+-----+-----+-----+-----+
    | . . | . . | . . | . . | . . |
    | . . | . . | . . | . . | . . |
    +-----+-----+-----+-----+-----+
    | . . | . . | 5 0 | . . | . . |
    | . . | . . | 0 0 | . . | . . |
    +-----+-----+-----+-----+-----+
    | . . | . . | . . | . . | . . |
    | . . | . . | . . | . . | . . |
    +-----+-----+-----+-----+-----+
    | 0 0 | . . | . . | . . | . . |
    | 4 0 | . . | . . | . . | . . |
    +-----+-----+-----+-----+-----+

    Output appears as dimension coordinates but lexicographically
    sorted by level coordinates. For BSR, the blocks are filled.
    """
    if id_map is 0:
        return f"""# extended FROSTT format
2 5
10 10
1 1 1
1 10 3
2 2 2
5 5 5
10 1 4
"""
    if id_map is 1:
        return f"""# extended FROSTT format
2 5
10 10
1 1 1
10 1 4
2 2 2
5 5 5
1 10 3
"""
    if id_map is 2:
        return f"""# extended FROSTT format
2 16
10 10
1 1 1
1 2 0
2 1 0
2 2 2
1 9 0
1 10 3
2 9 0
2 10 0
5 5 5
5 6 0
6 5 0
6 6 0
9 1 0
9 2 0
10 1 4
10 2 0
"""
    raise AssertionError("unexpected id_map")


def build_compile_and_run_output(attr: st.EncodingAttr, compiler, expected):
    # Build and Compile.
    module = ir.Module.parse(boilerplate(attr))
    engine = compiler.compile_and_jit(module)
    # Invoke the kernel and compare output.
    with tempfile.TemporaryDirectory() as test_dir:
        out = os.path.join(test_dir, "out.tns")
        buf = out.encode("utf-8")
        mem_a = ctypes.pointer(ctypes.pointer(ctypes.create_string_buffer(buf)))
        engine.invoke("main", mem_a)
        actual = open(out).read()
        if actual != expected:
            quit("FAILURE")


def main():
    support_lib = os.getenv("SUPPORT_LIB")
    assert support_lib is not None, "SUPPORT_LIB is undefined"
    if not os.path.exists(support_lib):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), support_lib)

    # CHECK-LABEL: TEST: test_output
    print("\nTEST: test_output")
    count = 0
    with ir.Context() as ctx, ir.Location.unknown():
        # Loop over various sparse types (COO, CSR, DCSR, CSC, DCSC) with
        # regular and loose compression and various metadata bitwidths.
        # For these simple orderings, dim2lvl and lvl2dim are the same.
        builder = st.EncodingAttr.build_level_type
        fmt = st.LevelFormat
        prop = st.LevelProperty
        levels = [
            [builder(fmt.compressed, [prop.non_unique]), builder(fmt.singleton)],
            [builder(fmt.dense), builder(fmt.compressed)],
            [builder(fmt.dense), builder(fmt.loose_compressed)],
            [builder(fmt.compressed), builder(fmt.compressed)],
        ]
        orderings = [
            (ir.AffineMap.get_permutation([0, 1]), 0),
            (ir.AffineMap.get_permutation([1, 0]), 1),
        ]
        bitwidths = [8, 64]
        compiler = sparsifier.Sparsifier(
            options="", opt_level=2, shared_libs=[support_lib]
        )
        for level in levels:
            for ordering, id_map in orderings:
                for bwidth in bitwidths:
                    attr = st.EncodingAttr.get(
                        level, ordering, ordering, bwidth, bwidth
                    )
                    build_compile_and_run_output(attr, compiler, expected(id_map))
                    count = count + 1

        # Now do the same for BSR.
        level = [
            builder(fmt.dense),
            builder(fmt.compressed),
            builder(fmt.dense),
            builder(fmt.dense),
        ]
        d0 = ir.AffineDimExpr.get(0)
        d1 = ir.AffineDimExpr.get(1)
        c2 = ir.AffineConstantExpr.get(2)
        dim2lvl = ir.AffineMap.get(
            2,
            0,
            [
                ir.AffineExpr.get_floor_div(d0, c2),
                ir.AffineExpr.get_floor_div(d1, c2),
                ir.AffineExpr.get_mod(d0, c2),
                ir.AffineExpr.get_mod(d1, c2),
            ],
        )
        l0 = ir.AffineDimExpr.get(0)
        l1 = ir.AffineDimExpr.get(1)
        l2 = ir.AffineDimExpr.get(2)
        l3 = ir.AffineDimExpr.get(3)
        lvl2dim = ir.AffineMap.get(4, 0, [2 * l0 + l2, 2 * l1 + l3])
        attr = st.EncodingAttr.get(level, dim2lvl, lvl2dim, 0, 0)
        build_compile_and_run_output(attr, compiler, expected(2))
        count = count + 1

    # CHECK: Passed 17 tests
    print("Passed", count, "tests")


if __name__ == "__main__":
    main()
