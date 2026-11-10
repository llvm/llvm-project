// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(canonicalize{region-simplify=disabled dead-code-elimination=disabled}))' | FileCheck %s --check-prefix=DISABLED
// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(canonicalize{region-simplify=disabled dead-code-elimination=newly-dead}))' | FileCheck %s --check-prefix=NEWLY
// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(canonicalize{region-simplify=disabled}))' | FileCheck %s --check-prefix=ALL

// The multiplication is dead before rewriting. Folding the truncation of the
// extension leaves the extension dead, and folding the addition of zero leaves
// the constant dead.

// DISABLED-LABEL: func @dead_code
// DISABLED-NEXT:    arith.constant 0 : i32
// DISABLED-NEXT:    arith.muli
// DISABLED-NEXT:    arith.extsi
// DISABLED-NEXT:    return

// NEWLY-LABEL: func @dead_code
// NEWLY-NEXT:    arith.muli
// NEWLY-NEXT:    return

// ALL-LABEL: func @dead_code
// ALL-NEXT:    return
func.func @dead_code(%a: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %dead = arith.muli %a, %a : i32
  %ext = arith.extsi %a : i32 to i64
  %trunc = arith.trunci %ext : i64 to i32
  %r = arith.addi %trunc, %c0 : i32
  return %r : i32
}

// Folding the cast into the load modifies the load in place and leaves the
// cast dead.

// DISABLED-LABEL: func @in_place_fold
// DISABLED-NEXT:    memref.cast
// DISABLED-NEXT:    memref.load
// DISABLED-NEXT:    return

// NEWLY-LABEL: func @in_place_fold
// NEWLY-NEXT:    memref.load
// NEWLY-NEXT:    return

// ALL-LABEL: func @in_place_fold
// ALL-NEXT:    memref.load
// ALL-NEXT:    return
func.func @in_place_fold(%m: memref<4xf32>, %i: index) -> f32 {
  %cast = memref.cast %m : memref<4xf32> to memref<?xf32>
  %v = memref.load %cast[%i] : memref<?xf32>
  return %v : f32
}
