// RUN: mlir-opt -test-memref-to-llvm-with-transforms %s | FileCheck %s

// Checks that the program does not crash. The functionality of the pattern is
// already checked in test/Dialect/MemRef/*.mlir

func.func @subview_folder(%arg0: memref<100x100xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> memref<?x?xf32, strided<[100, 1], offset: ?>> {
  %subview = memref.subview %arg0[%arg1, %arg2] [%arg3, %arg4] [1, 1] : memref<100x100xf32> to memref<?x?xf32, strided<[100, 1], offset: ?>>
  return %subview : memref<?x?xf32, strided<[100, 1], offset: ?>>
}
// CHECK-LABEL: llvm.func @subview_folder
