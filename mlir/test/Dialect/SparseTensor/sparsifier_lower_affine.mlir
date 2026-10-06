// RUN: mlir-opt %s --sparsifier | FileCheck %s

// Verify that SCF control flow nested in an affine loop can be lowered by the
// sparsifier pipeline without creating an invalid multi-block affine region.
// CHECK-LABEL: llvm.func @nested_scf_in_affine
// CHECK-NOT: affine.for
func.func @nested_scf_in_affine(%cond: i1) -> i64 {
  %zero = arith.constant 0 : i64
  %one = arith.constant 1 : i64
  %result = affine.for %i = 0 to 4 iter_args(%acc = %zero) -> i64 {
    %value = scf.if %cond -> i64 {
      %sum = arith.addi %acc, %one : i64
      scf.yield %sum : i64
    } else {
      scf.yield %acc : i64
    }
    affine.yield %value : i64
  }
  return %result : i64
}
