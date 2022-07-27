// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

// CHECK-LABEL: func @test_clone
func.func @test_clone(%buf : memref<*xf32>) -> memref<*xf32> {
  %clone = bufferization.clone %buf : memref<*xf32> to memref<*xf32>
  return %clone : memref<*xf32>
}

// CHECK-LABEL: test_to_memref
func.func @test_to_memref(%arg0: tensor<?xi64>, %arg1: tensor<*xi64>)
    -> (memref<?xi64, affine_map<(d0) -> (d0 + 7)>>, memref<*xi64, 1>) {
  %0 = bufferization.to_memref %arg0
    : memref<?xi64, affine_map<(d0) -> (d0 + 7)>>
  %1 = bufferization.to_memref %arg1
    : memref<*xi64, 1>
  return %0, %1 : memref<?xi64, affine_map<(d0) -> (d0 + 7)>>, memref<*xi64, 1>
}

// CHECK-LABEL: func @test_to_tensor
func.func @test_to_tensor(%buf : memref<2xf32>) -> tensor<2xf32> {
  %tensor = bufferization.to_tensor %buf : memref<2xf32>
  return %tensor : tensor<2xf32>
}

// CHECK-LABEL: func @test_alloc_tensor_op
func.func @test_alloc_tensor_op(%t: tensor<?x5xf32>, %sz: index)
  -> tensor<?x5xf32>
{
  // CHECK: bufferization.alloc_tensor(%{{.*}}) : tensor<?x5xf32>
  %0 = bufferization.alloc_tensor(%sz) : tensor<?x5xf32>
  // CHECK: bufferization.alloc_tensor() copy(%{{.*}}) : tensor<?x5xf32>
  %1 = bufferization.alloc_tensor() copy(%t) : tensor<?x5xf32>
  // CHECK: bufferization.alloc_tensor() : tensor<5x6xf32>
  %2 = bufferization.alloc_tensor() : tensor<5x6xf32>
  // CHECK: bufferization.alloc_tensor(%{{.*}}, %{{.*}}) : tensor<?x?xf32>
  %3 = bufferization.alloc_tensor(%sz, %sz) : tensor<?x?xf32>
  // CHECK: bufferization.alloc_tensor() copy(%{{.*}}) {escape = true} : tensor<?x5xf32>
  %4 = bufferization.alloc_tensor() copy(%t) {escape = true} : tensor<?x5xf32>
  // CHECK: bufferization.alloc_tensor() copy(%{{.*}}) {escape = false} : tensor<?x5xf32>
  %5 = bufferization.alloc_tensor() copy(%t) {escape = false} : tensor<?x5xf32>
  return %1 : tensor<?x5xf32>
}

// CHECK-LABEL: func @test_dealloc_tensor_op
func.func @test_dealloc_tensor_op(%arg0: tensor<4xi32>) {
  // CHECK: bufferization.dealloc_tensor {{.*}} : tensor<4xi32>
  bufferization.dealloc_tensor %arg0 : tensor<4xi32>
  return
}
