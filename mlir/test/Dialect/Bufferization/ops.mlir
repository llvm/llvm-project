// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

// CHECK-LABEL: func @test_clone
func @test_clone(%buf : memref<*xf32>) -> memref<*xf32> {
  %clone = bufferization.clone %buf : memref<*xf32> to memref<*xf32>
  return %clone : memref<*xf32>
}

// CHECK-LABEL: test_to_memref
func @test_to_memref(%arg0: tensor<?xi64>, %arg1: tensor<*xi64>)
    -> (memref<?xi64, affine_map<(d0) -> (d0 + 7)>>, memref<*xi64, 1>) {
  %0 = bufferization.to_memref %arg0
    : memref<?xi64, affine_map<(d0) -> (d0 + 7)>>
  %1 = bufferization.to_memref %arg1
    : memref<*xi64, 1>
  return %0, %1 : memref<?xi64, affine_map<(d0) -> (d0 + 7)>>, memref<*xi64, 1>
}

// CHECK-LABEL: func @test_to_tensor
func @test_to_tensor(%buf : memref<2xf32>) -> tensor<2xf32> {
  %tensor = bufferization.to_tensor %buf : memref<2xf32>
  return %tensor : tensor<2xf32>
}
