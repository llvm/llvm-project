// DEFINE: %{vectorize} = mlir-opt -transform-interpreter
// DEFINE: %{lower} = mlir-opt -test-transform-dialect-erase-schedule \
// DEFINE:   -one-shot-bufferize=bufferize-function-boundaries \
// DEFINE:   -buffer-deallocation-pipeline -test-lower-to-llvm
// DEFINE: %{run} = mlir-runner -e main -entry-point-result=f32 -O0 \
// DEFINE:   -shared-libs=%mlir_c_runner_utils

// RUN: %{lower} %s | %{run} | FileCheck %s
// RUN: %{vectorize} %s | %{lower} | %{run} | FileCheck %s

// The subview keeps strides [3, 1]: a flat logical index of 2 must read
// storage[1, 0] (20), not the excluded storage[0, 2] (99).
// CHECK: 2.000000e+01

module attributes {transform.with_named_sequence} {
  func.func @main() -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    %f10 = arith.constant 10.0 : f32
    %f11 = arith.constant 11.0 : f32
    %f99 = arith.constant 99.0 : f32
    %f20 = arith.constant 20.0 : f32
    %f21 = arith.constant 21.0 : f32

    // [10, 11, 99]
    // [20, 21, 99]
    %storage = memref.alloc() : memref<2x3xf32>
    memref.store %f10, %storage[%c0, %c0] : memref<2x3xf32>
    memref.store %f11, %storage[%c0, %c1] : memref<2x3xf32>
    memref.store %f99, %storage[%c0, %c2] : memref<2x3xf32>
    memref.store %f20, %storage[%c1, %c0] : memref<2x3xf32>
    memref.store %f21, %storage[%c1, %c1] : memref<2x3xf32>
    memref.store %f99, %storage[%c1, %c2] : memref<2x3xf32>

    %source = memref.subview %storage[0, 0] [2, 2] [1, 1]
      : memref<2x3xf32> to memref<2x2xf32, strided<[3, 1]>>

    %tensor = bufferization.to_tensor %source restrict
      : memref<2x2xf32, strided<[3, 1]>> to tensor<2x2xf32>

    %empty = tensor.empty() : tensor<2xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]
    } outs(%empty : tensor<2xf32>) {
    ^bb0(%unused: f32):
      %i = linalg.index 0 : index
      %value = tensor.extract %tensor[%i, %c0] : tensor<2x2xf32>
      linalg.yield %value : f32
    } -> tensor<2xf32>

    %second = tensor.extract %result[%c1] : tensor<2xf32>
    return %second : f32
  }

  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %target = transform.structured.match ops{["linalg.generic"]} in %root
      : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %target vectorize_nd_extract
      : !transform.any_op
    transform.yield
  }
}
