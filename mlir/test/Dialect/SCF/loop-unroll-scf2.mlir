// RUN: mlir-opt %s --scf-loop-unroll --split-input-file | FileCheck %s

module {
  func.func @main() -> f32 {
    %N = arith.constant 10 : index
    %val = arith.constant 2.0 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %array = memref.alloc() : memref<10xf32>

    // Initialize array with %val
    scf.for %i = %c0 to %N step %c1 {
      memref.store %val, %array[%i] : memref<10xf32>
    }

    %sum = arith.constant 0.0 : f32

    %result = scf.for %j = %c0 to %N step %c1 iter_args(%iter_sum = %sum) -> (f32) {
      %current_val = memref.load %array[%j] : memref<10xf32>
      %new_sum = arith.addf %iter_sum, %current_val : f32
      scf.yield %new_sum : f32
    }

    return %result : f32
  }
}

//CHECK-LABEL: func.func @main() -> f32 {
//CHECK-NEXT: %c10 = arith.constant 10 : index
//CHECK-NEXT: %cst = arith.constant 2.000000e+00 : f32
//CHECK-NEXT: %c0 = arith.constant 0 : index
//CHECK-NEXT: %c1 = arith.constant 1 : index
//CHECK-NEXT: %alloc = memref.alloc() : memref<10xf32>
//CHECK-NEXT: %c8 = arith.constant 8 : index
//CHECK-NEXT: %c4 = arith.constant 4 : index
//CHECK-NEXT: scf.for %arg0 = %c0 to %c10 step %c4 {
//CHECK-NEXT:   memref.store %cst, %alloc[%arg0] : memref<10xf32>
//CHECK-NEXT:   %c1_3 = arith.constant 1 : index
//CHECK-NEXT:   %c1_4 = arith.constant 1 : index
//CHECK-NEXT:   %2 = arith.muli %c1_3, %c1_4 : index
//CHECK-NEXT:   %3 = arith.addi %arg0, %2 : index
//CHECK-NEXT:   memref.store %cst, %alloc[%3] : memref<10xf32>
//CHECK-NEXT:   %c2 = arith.constant 2 : index
//CHECK-NEXT:   %c1_5 = arith.constant 1 : index
//CHECK-NEXT:   %4 = arith.muli %c2, %c1_5 : index
//CHECK-NEXT:   %5 = arith.addi %arg0, %4 : index
//CHECK-NEXT:   memref.store %cst, %alloc[%5] : memref<10xf32>
//CHECK-NEXT:   %c3 = arith.constant 3 : index
//CHECK-NEXT:   %c1_6 = arith.constant 1 : index
//CHECK-NEXT:   %6 = arith.muli %c3, %c1_6 : index
//CHECK-NEXT:   %7 = arith.addi %arg0, %6 : index
//CHECK-NEXT:   memref.store %cst, %alloc[%7] : memref<10xf32>
//CHECK-NEXT: }
//CHECK-NEXT: scf.for %arg0 = %c8 to %c10 step %c1 {
//CHECK-NEXT:   memref.store %cst, %alloc[%arg0] : memref<10xf32>
//CHECK-NEXT: }
//CHECK-NEXT: %cst_0 = arith.constant 0.000000e+00 : f32
//CHECK-NEXT: %c8_1 = arith.constant 8 : index
//CHECK-NEXT: %c4_2 = arith.constant 4 : index
//CHECK-NEXT: %0 = scf.for %arg0 = %c0 to %c10 step %c4_2 iter_args(%arg1 = %cst_0) -> (f32) {
//CHECK-NEXT:   %2 = memref.load %alloc[%arg0] : memref<10xf32>
//CHECK-NEXT:   %3 = arith.addf %arg1, %2 : f32
//CHECK-NEXT:   %c1_3 = arith.constant 1 : index
//CHECK-NEXT:   %c1_4 = arith.constant 1 : index
//CHECK-NEXT:   %4 = arith.muli %c1_3, %c1_4 : index
//CHECK-NEXT:   %5 = arith.addi %arg0, %4 : index
//CHECK-NEXT:   %6 = memref.load %alloc[%5] : memref<10xf32>
//CHECK-NEXT:   %7 = arith.addf %3, %6 : f32
//CHECK-NEXT:   %c2 = arith.constant 2 : index
//CHECK-NEXT:   %c1_5 = arith.constant 1 : index
//CHECK-NEXT:   %8 = arith.muli %c2, %c1_5 : index
//CHECK-NEXT:   %9 = arith.addi %arg0, %8 : index
//CHECK-NEXT:   %10 = memref.load %alloc[%9] : memref<10xf32>
//CHECK-NEXT:   %11 = arith.addf %7, %10 : f32
//CHECK-NEXT:   %c3 = arith.constant 3 : index
//CHECK-NEXT:   %c1_6 = arith.constant 1 : index
//CHECK-NEXT:   %12 = arith.muli %c3, %c1_6 : index
//CHECK-NEXT:   %13 = arith.addi %arg0, %12 : index
//CHECK-NEXT:   %14 = memref.load %alloc[%13] : memref<10xf32>
//CHECK-NEXT:   %15 = arith.addf %11, %14 : f32
//CHECK-NEXT:   scf.yield %15 : f32
//CHECK-NEXT: }
//CHECK-NEXT: %1 = scf.for %arg0 = %c8_1 to %c10 step %c1 iter_args(%arg1 = %0) -> (f32) {
//CHECK-NEXT:   %2 = memref.load %alloc[%arg0] : memref<10xf32>
//CHECK-NEXT:   %3 = arith.addf %arg1, %2 : f32
//CHECK-NEXT:   scf.yield %3 : f32
//CHECK-NEXT: }
//CHECK-NEXT: return %1 : f32
//CHECK-NEXT: }
