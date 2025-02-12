// DEFINE: %{tosa-to-linalg-pipeline} = -pass-pipeline="builtin.module(func.func(tosa-infer-shapes,tosa-to-linalg-named,tosa-to-linalg,tosa-to-arith))"

// RUN:   mlir-opt %s \
// RUN:     %{tosa-to-linalg-pipeline} \
// RUN: | mlir-opt \
// RUN:     -one-shot-bufferize="bufferize-function-boundaries" \
// RUN:     -buffer-deallocation-pipeline \
// RUN:     -test-lower-to-llvm \
// RUN: | mlir-runner \
// RUN:     -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils \
// RUN: | FileCheck %s

// Validate that the TOSA lowering for tosa.max_pool2d produces the same results when
// for fully static and fully dynamic inputs.

!tensor_type = tensor<1x4x4x1xf32>
!memref_type = memref<1x4x4x1xf32>

// Utility functions
func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }

func.func @max_pool_static(%arg0: !tensor_type) -> (!tensor_type) {
  %0 = tosa.max_pool2d %arg0 {
    pad = array<i64: 1, 1, 1, 1>,
    kernel = array<i64: 3, 3>,
    stride = array<i64: 1, 1>
  } : (tensor<1x4x4x1xf32>) -> tensor<1x4x4x1xf32>
  return %0 : tensor<1x4x4x1xf32>
}

func.func @max_pool_dynamic(%arg0: tensor<?x?x?x?xf32>) -> (tensor<?x?x?x?xf32>) {
  %0 = tosa.max_pool2d %arg0 {
    pad = array<i64: 1, 1, 1, 1>,
    kernel = array<i64: 3, 3>,
    stride = array<i64: 1, 1>
  } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

// Test harness to compare the results of a fully statically shaped max_pool2d with
// a fully dynamically shaped max_pool2d on the same inputs.
func.func @main() {
  %A = arith.constant dense<[[
    [[0.0], [0.1], [0.2], [0.3]], // H = 0
    [[1.0], [1.1], [1.2], [1.3]], // H = 1
    [[2.0], [2.1], [2.2], [2.3]], // H = 2
    [[3.0], [3.1], [3.2], [3.3]]  // H = 3
  ]]> : tensor<1x4x4x1xf32>

  %A_dynamic = tensor.cast %A : !tensor_type to tensor<?x?x?x?xf32>

  // Call both static and dynamically sized variants
  %result_static  = func.call @max_pool_static(%A) : (!tensor_type) -> !tensor_type
  %result_dynamic = func.call @max_pool_dynamic(%A_dynamic) : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>

  %static_buffer = bufferization.to_memref %result_static : !tensor_type to !memref_type
  %unranked_static_buffer = memref.cast %static_buffer : !memref_type to memref<*xf32>

  // CHECK: Unranked Memref base@ = {{.*}} rank = 4 offset = 0 sizes = [1, 4, 4, 1] strides = [16, 4, 1, 1] data =

  // CHECK-NEXT: 1.1
  // CHECK-NEXT: 1.2
  // CHECK-NEXT: 1.3
  // CHECK-NEXT: 1.3

  // CHECK-NEXT: 2.1
  // CHECK-NEXT: 2.2
  // CHECK-NEXT: 2.3
  // CHECK-NEXT: 2.3

  // CHECK-NEXT: 3.1
  // CHECK-NEXT: 3.2
  // CHECK-NEXT: 3.3
  // CHECK-NEXT: 3.3

  // CHECK-NEXT: 3.1
  // CHECK-NEXT: 3.2
  // CHECK-NEXT: 3.3
  // CHECK-NEXT: 3.3

  func.call @printMemrefF32(%unranked_static_buffer) : (memref<*xf32>) -> ()

  %dynamic_buffer = bufferization.to_memref %result_dynamic : tensor<?x?x?x?xf32> to memref<?x?x?x?xf32>
  %unranked_dynamic_buffer = memref.cast %dynamic_buffer : memref<?x?x?x?xf32> to memref<*xf32>

  // CHECK: Unranked Memref base@ = {{.*}} rank = 4 offset = 0 sizes = [1, 4, 4, 1] strides = [16, 4, 1, 1] data =
  // CHECK-NEXT: 1.1
  // CHECK-NEXT: 1.2
  // CHECK-NEXT: 1.3
  // CHECK-NEXT: 1.3

  // CHECK-NEXT: 2.1
  // CHECK-NEXT: 2.2
  // CHECK-NEXT: 2.3
  // CHECK-NEXT: 2.3

  // CHECK-NEXT: 3.1
  // CHECK-NEXT: 3.2
  // CHECK-NEXT: 3.3
  // CHECK-NEXT: 3.3

  // CHECK-NEXT: 3.1
  // CHECK-NEXT: 3.2
  // CHECK-NEXT: 3.3
  // CHECK-NEXT: 3.3

  func.call @printMemrefF32(%unranked_dynamic_buffer) : (memref<*xf32>) -> ()

  return
}

