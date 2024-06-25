// RUN: toyc-ch7 %s -emit=mlir-affine 2>&1 | FileCheck %s

module {
  toy.func @main() {
    %0 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>
    %1 = toy.reshape(%0 : tensor<6xf64>) to tensor<6x1xf64>
    %2 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>
    %3 = toy.reshape(%2 : tensor<6xf64>) to tensor<1x6xf64>
    %4 = toy.matmul %1, %3 : tensor<6x1xf64>, tensor<1x6xf64> -> tensor<*xf64>
    toy.print %4 : tensor<*xf64>
    toy.return
  }
}

//CHECK-LABEL: func @main
//CHECK-NEXT: %cst = arith.constant 0.000000e+00 : f64
//CHECK-NEXT: %cst_0 = arith.constant 6.000000e+00 : f64
//CHECK-NEXT: %cst_1 = arith.constant 5.000000e+00 : f64
//CHECK-NEXT: %cst_2 = arith.constant 4.000000e+00 : f64
//CHECK-NEXT: %cst_3 = arith.constant 3.000000e+00 : f64
//CHECK-NEXT: %cst_4 = arith.constant 2.000000e+00 : f64
//CHECK-NEXT: %cst_5 = arith.constant 1.000000e+00 : f64
//CHECK-NEXT: %alloc = memref.alloc() : memref<1x6xf64>
//CHECK-NEXT: %alloc_6 = memref.alloc() : memref<6x1xf64>
//CHECK-NEXT: affine.store %cst_5, %alloc_6[0, 0] : memref<6x1xf64>
//CHECK-NEXT: affine.store %cst_4, %alloc_6[1, 0] : memref<6x1xf64>
//CHECK-NEXT: affine.store %cst_3, %alloc_6[2, 0] : memref<6x1xf64>
//CHECK-NEXT: affine.store %cst_2, %alloc_6[3, 0] : memref<6x1xf64>
//CHECK-NEXT: affine.store %cst_1, %alloc_6[4, 0] : memref<6x1xf64>
//CHECK-NEXT: affine.store %cst_0, %alloc_6[5, 0] : memref<6x1xf64>
//CHECK-NEXT: affine.store %cst_5, %alloc[0, 0] : memref<1x6xf64>
//CHECK-NEXT: affine.store %cst_4, %alloc[0, 1] : memref<1x6xf64>
//CHECK-NEXT: affine.store %cst_3, %alloc[0, 2] : memref<1x6xf64>
//CHECK-NEXT: affine.store %cst_2, %alloc[0, 3] : memref<1x6xf64>
//CHECK-NEXT: affine.store %cst_1, %alloc[0, 4] : memref<1x6xf64>
//CHECK-NEXT: affine.store %cst_0, %alloc[0, 5] : memref<1x6xf64>
//CHECK-NEXT: %alloc_7 = memref.alloc() : memref<6x6xf64>
//CHECK-NEXT: affine.for %arg0 = 0 to 6 {
//CHECK-NEXT:   affine.for %arg1 = 0 to 6 {
//CHECK-NEXT:     affine.store %cst, %alloc_7[%arg0, %arg1] : memref<6x6xf64>
//CHECK-NEXT:     affine.for %arg2 = 0 to 1 {
//CHECK-NEXT:       %0 = affine.load %alloc_6[%arg0, %arg2] : memref<6x1xf64>
//CHECK-NEXT:       %1 = affine.load %alloc[%arg2, %arg1] : memref<1x6xf64>
//CHECK-NEXT:       %2 = arith.mulf %0, %1 : f64
//CHECK-NEXT:       %3 = arith.addf %2, %cst : f64
//CHECK-NEXT:       %4 = affine.load %alloc_7[%arg0, %arg1] : memref<6x6xf64>
//CHECK-NEXT:       %5 = arith.addf %3, %4 : f64
//CHECK-NEXT:       affine.store %5, %alloc_7[%arg0, %arg1] : memref<6x6xf64>
//CHECK-NEXT:     }
//CHECK-NEXT:   }
//CHECK-NEXT: }
//CHECK-NEXT: toy.print %alloc_7 : memref<6x6xf64>
//CHECK-NEXT: memref.dealloc %alloc_6 : memref<6x1xf64>
//CHECK-NEXT: memref.dealloc %alloc : memref<1x6xf64>
//CHECK-NEXT: return
//CHECK-NEXT: }
