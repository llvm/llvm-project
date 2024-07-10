// RUN: mlir-opt %s --scf-loop-unroll-jam="unroll-jam-factor=2"  --split-input-file | FileCheck %s

module {
  func.func @main() -> f32 {
    %sum = arith.constant 0.0 : f32
    %val = arith.constant 2.0 : f32
    %N = arith.constant 16 : index
    %num = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %result = scf.for %i = %c0 to %N step %c1 iter_args(%iter_sum = %sum) -> (f32) {
      %new_sum = arith.addf %iter_sum, %val : f32
      %result2 = scf.for %j = %c0 to %num step %c1 iter_args(%iter_sum2 = %val) -> (f32) {
        %new_sum2 = arith.addf %iter_sum2, %val : f32
        scf.yield %new_sum2 : f32
      }
      %new_sum3 = arith.addf %result2, %val : f32
      scf.yield %new_sum : f32
    }
    return %result : f32
  }
}

// CHECK-LABEL: func.func @main() -> f32 {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:   %cst_0 = arith.constant 2.000000e+00 : f32
// CHECK-NEXT:   %c16 = arith.constant 16 : index
// CHECK-NEXT:   %c16_1 = arith.constant 16 : index
// CHECK-NEXT:   %c0 = arith.constant 0 : index
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   %c2 = arith.constant 2 : index
// CHECK-NEXT:   %c2_2 = arith.constant 2 : index
// CHECK-NEXT:   %0:2 = scf.for %arg0 = %c0 to %c16 step %c2_2 iter_args(%arg1 = %cst, %arg2 = %cst) -> (f32, f32) {
// CHECK-NEXT:     %2 = arith.addf %arg1, %cst_0 : f32
// CHECK-NEXT:     %3 = arith.addf %arg2, %cst_0 : f32
// CHECK-NEXT:     %4:2 = scf.for %arg3 = %c0 to %c16_1 step %c1 iter_args(%arg4 = %cst_0, %arg5 = %cst_0) -> (f32, f32) {
// CHECK-NEXT:       %7 = arith.addf %arg4, %cst_0 : f32
// CHECK-NEXT:       %8 = arith.addf %arg5, %cst_0 : f32
// CHECK-NEXT:       scf.yield %7, %8 : f32, f32
// CHECK-NEXT:     }
// CHECK-NEXT:     %5 = arith.addf %4#0, %cst_0 : f32
// CHECK-NEXT:     %6 = arith.addf %4#1, %cst_0 : f32
// CHECK-NEXT:     scf.yield %2, %3 : f32, f32
// CHECK-NEXT:   }
// CHECK-NEXT:   %1 = arith.addf %0#0, %0#1 : f32
// CHECK-NEXT:   return %1 : f32
// CHECK-NEXT: }
