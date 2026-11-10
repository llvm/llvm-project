// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(canonicalize{region-simplify=disabled dead-code-elimination=false}))' | FileCheck %s --check-prefix=NODCE
// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(canonicalize{region-simplify=disabled}))' | FileCheck %s --check-prefix=DCE

// Without dead code elimination the unused multiplication is kept, while
// folding still replaces the addition of zero.

// NODCE-LABEL: func @dead_code
// NODCE:         arith.muli %[[A:.*]], %[[A]] : i32
// NODCE-NOT:     arith.addi
// NODCE:         return %[[A]] : i32

// DCE-LABEL: func @dead_code
// DCE-NOT:     arith.muli
// DCE-NOT:     arith.addi
// DCE:         return %{{.*}} : i32
func.func @dead_code(%a: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %dead = arith.muli %a, %a : i32
  %r = arith.addi %a, %c0 : i32
  return %r : i32
}
