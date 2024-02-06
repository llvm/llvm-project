// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

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
  %tensor = bufferization.to_tensor %buf restrict writable : memref<2xf32>
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
  %c100 = arith.constant 100 : index
  // CHECK: bufferization.alloc_tensor() size_hint=
  %6 = bufferization.alloc_tensor() size_hint=%c100 : tensor<100x100xf64, #CSR>
  // CHECK: bufferization.alloc_tensor(%{{.+}}) {memory_space = "foo"} : tensor<?xf32>
  %7 = bufferization.alloc_tensor(%sz) {memory_space = "foo"} : tensor<?xf32>
  return %1 : tensor<?x5xf32>
}

// CHECK-LABEL: func @test_dealloc_tensor_op
func.func @test_dealloc_tensor_op(%arg0: tensor<4xi32>) {
  // CHECK: bufferization.dealloc_tensor {{.*}} : tensor<4xi32>
  bufferization.dealloc_tensor %arg0 : tensor<4xi32>
  return
}

// CHECK-LABEL: func @test_materialize_in_destination_op
func.func @test_materialize_in_destination_op(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: memref<?xf32, 3>)
    -> tensor<?xf32> {
  // CHECK: bufferization.materialize_in_destination {{.*}} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %1 = bufferization.materialize_in_destination %arg0 in %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  // CHECK: bufferization.materialize_in_destination {{.*}} : (tensor<?xf32>, memref<?xf32, 3>) -> ()
  bufferization.materialize_in_destination %arg0 in restrict writable %arg2 : (tensor<?xf32>, memref<?xf32, 3>) -> ()
  return %1 : tensor<?xf32>
}

// CHECK-LABEL: func @test_dealloc_op
func.func @test_dealloc_op(%arg0: memref<2xf32>, %arg1: memref<4xi32>,
                           %arg2: i1, %arg3: i1, %arg4: memref<?xf32>,
                           %arg5: memref<*xf64>) -> (i1, i1) {
  // CHECK: bufferization.dealloc (%arg0, %arg1 : memref<2xf32>, memref<4xi32>) if (%arg2, %arg3) retain (%arg4, %arg5 : memref<?xf32>, memref<*xf64>)
  %0:2 = bufferization.dealloc (%arg0, %arg1 : memref<2xf32>, memref<4xi32>) if (%arg2, %arg3) retain (%arg4, %arg5 : memref<?xf32>, memref<*xf64>)
  // CHECK: bufferization.dealloc (%arg0 : memref<2xf32>) if (%arg2)
  bufferization.dealloc (%arg0 : memref<2xf32>) if (%arg2)
  // CHECK: bufferization.dealloc
  bufferization.dealloc
  return %0#0, %0#1 : i1, i1
}
