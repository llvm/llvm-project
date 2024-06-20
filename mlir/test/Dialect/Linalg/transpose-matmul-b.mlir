// RUN: mlir-opt %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul", "linalg.batch_matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.transpose_matmul %matmul <rhs> : (!transform.any_op) -> (!transform.any_op)
    %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_cse to %0 : !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.yield
  }
}
