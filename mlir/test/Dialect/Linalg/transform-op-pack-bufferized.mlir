// RUN: mlir-opt -sparsification-and-bufferization -transform-interpreter -split-input-file -verify-diagnostics -allow-unregistered-dialect %s

module attributes {gpu.data_layout = "chunked", impl.libfunc.name = "test.matmul"} {
  func.func @matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %0 : tensor<?x?xf32>
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      // expected-error @below {{data tiling failed}}
      %1 = transform.structured.pack %0 packed_sizes = [2, 3, 4] : (!transform.any_op) -> !transform.op<"linalg.generic">
      transform.yield
    }
  }
}
