// RUN: mlir-opt %s --scf-loop-unroll --split-input-file | FileCheck %s
module {
  func.func @main() -> f32 {
    %sum = arith.constant 0.0 : f32
    %val = arith.constant 2.0 : f32
    %N = arith.constant 10 : index

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %result = scf.for %i = %c0 to %N step %c1 iter_args(%iter_sum = %sum) -> (f32) {
      %new_sum = arith.mulf %iter_sum, %val : f32
      scf.yield %new_sum : f32
    }
    return %result : f32
  }
}
//CHECK-LABEL: func.func @main() -> f32 {
//CHECK-NEXT: %cst = arith.constant 0.000000e+00 : f32
//CHECK-NEXT: %cst_0 = arith.constant 2.000000e+00 : f32
//CHECK-NEXT: %c10 = arith.constant 10 : index
//CHECK-NEXT: %c0 = arith.constant 0 : index
//CHECK-NEXT: %c1 = arith.constant 1 : index
//CHECK-NEXT: %c8 = arith.constant 8 : index
//CHECK-NEXT: %c4 = arith.constant 4 : index
//CHECK-NEXT: %0 = scf.for %arg0 = %c0 to %c10 step %c4 iter_args(%arg1 = %cst) -> (f32) {
//CHECK-NEXT:   %2 = arith.mulf %arg1, %cst_0 : f32
//CHECK-NEXT:   %3 = arith.mulf %2, %cst_0 : f32
//CHECK-NEXT:   %4 = arith.mulf %3, %cst_0 : f32
//CHECK-NEXT:   %5 = arith.mulf %4, %cst_0 : f32
//CHECK-NEXT:   scf.yield %5 : f32
//CHECK-NEXT: }
//CHECK-NEXT: %1 = scf.for %arg0 = %c8 to %c10 step %c1 iter_args(%arg1 = %0) -> (f32) {
//CHECK-NEXT:   %2 = arith.mulf %arg1, %cst_0 : f32
//CHECK-NEXT:   scf.yield %2 : f32
//CHECK-NEXT:  }
//CHECK-NEXT: return
//CHECK-NEXT: }
