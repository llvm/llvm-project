module attributes {transform.with_named_sequence} {
  transform.named_sequence @flatten(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.vector.flatten_vector_transfer_ops
    } : !transform.any_op
    transform.yield
  }
}
