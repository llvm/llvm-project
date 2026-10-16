// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(canonicalize))' -mlir-print-ir-after=canonicalize -o /dev/null 2>&1 | FileCheck --check-prefix=DEFAULT %s
// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(canonicalize))' -mlir-print-ir-after=canonicalize -mlir-print-always-use-aliases -o /dev/null 2>&1 | FileCheck --check-prefix=ALIASES %s

// Per-function IR dumps (canonicalize here only triggers one; the pass
// itself is irrelevant) print the func op, which is not a top-level op,
// so types expand in full by default. With the flag, the dump uses aliases.

// DEFAULT: IR Dump After
// DEFAULT-NOT: !tuple =
// DEFAULT: func.func @f(%arg0: tuple<i32,
// DEFAULT: return %arg0 : tuple<i32,

// ALIASES: IR Dump After
// ALIASES: !tuple = tuple<i32,
// ALIASES: func.func @f(%arg0: !tuple, %arg1: !tuple) -> !tuple
// ALIASES: return %arg0 : !tuple

func.func @f(%arg0: tuple<i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32>, %arg1: tuple<i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32>) -> tuple<i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32> {
  return %arg0 : tuple<i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32>
}
