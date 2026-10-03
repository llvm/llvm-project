// REQUIRES: mlir-expensive-checks
// RUN: not --crash mlir-opt %s -pass-pipeline='builtin.module(func.func(canonicalize))' 2>&1 | FileCheck %s

// The fold changes the op in place, replaces one result, and does not mark the
// in-place change. The op survives, so the expensive checks report the missing
// mark.

// CHECK: LLVM ERROR: fold changed the operation without an in-place mark
func.func @partial_fold_unmarked_in_place(%arg0: i32) -> (i32, i32) {
  %0:2 = "test.op_partial_fold_unmarked_in_place"(%arg0) : (i32) -> (i32, i32)
  return %0#0, %0#1 : i32, i32
}
