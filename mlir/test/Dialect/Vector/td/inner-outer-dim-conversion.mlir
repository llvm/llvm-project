module attributes {transform.with_named_sequence} {
  transform.named_sequence @inner_outer_dim_reduction_conversion(%module_op: !transform.any_op {transform.readonly}) {

    %func_op = transform.structured.match ops{["func.func"]} in %module_op
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_op {
      // Test patterns
      transform.apply_patterns.vector.inner_outer_dim_reduction_conversion
    } : !transform.any_op

    transform.yield
  }
}

