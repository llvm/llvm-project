// RUN: mlir-opt -transform-interpreter -verify-diagnostics %s

func.func @two_matmuls(
    %arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : tensor<?x?xf32>, %arg3 : tensor<?x?xf32>,
    %arg4 : tensor<?x?xf32>, %arg5 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul
    ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.matmul
    ins(%arg3, %arg4 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg5 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    // expected-error @+1 {{generated 2 loops but only 1 loop results were declared}}
    %tiled, %loop = transform.test.tile_using_forall %matmul [10, 20]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
