// DEFINE: %{entry_point} = main

// DEFINE: %{vectorize} = mlir-opt -transform-interpreter
// DEFINE: %{lower} = mlir-opt -one-shot-bufferize=bufferize-function-boundaries \
// DEFINE:   -test-lower-to-llvm -test-transform-dialect-erase-schedule
// DEFINE: %{run} = mlir-runner -e %{entry_point} -entry-point-result=void \
// DEFINE:    -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils

/// End-to-end test for vectorizing a rank-reducing tensor.insert_slice. Both
/// runs use the same lowering and must print the same thing.

/// _Without_ vectorization (reference)
// RUN: %{lower} %s | %{run} | FileCheck %s

/// _With_ vectorization
// RUN: %{vectorize} %s | %{lower} | %{run} | FileCheck %s

func.func @main() {
  %pad = arith.constant 0 : i32
  %src = arith.constant dense<[1, 2, 3, 4, 5]> : tensor<5xi32>
  %empty = tensor.empty() : tensor<5x3xi32>
  %init = linalg.fill ins(%pad : i32) outs(%empty : tensor<5x3xi32>) -> tensor<5x3xi32>

  /// Dim 1 of the slice is dropped, so the source's only dim corresponds to
  /// dim 0 of the result.
  %res = tensor.insert_slice %src into %init[0, 2] [5, 1] [1, 1]
    : tensor<5xi32> into tensor<5x3xi32>

  %res_cast = tensor.cast %res : tensor<5x3xi32> to tensor<*xi32>

  // CHECK: Unranked Memref base@ = 0x{{.*}} rank = 2 offset = 0 sizes = [5, 3] strides = [3, 1] data =
  // CHECK-NEXT: [0,   0,   1]
  // CHECK-NEXT: [0,   0,   2]
  // CHECK-NEXT: [0,   0,   3]
  // CHECK-NEXT: [0,   0,   4]
  // CHECK-NEXT: [0,   0,   5]
  call @printMemrefI32(%res_cast) : (tensor<*xi32>) -> ()

  return
}

module @transforms attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
    %slice = transform.structured.match ops{["tensor.insert_slice"]} in %module
      : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %slice : !transform.any_op
    transform.yield
  }
}

func.func private @printMemrefI32(%ptr : tensor<*xi32>)
