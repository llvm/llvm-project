module attributes {transform.with_named_sequence} {
  transform.named_sequence @unroll_to_elements(%module_op: !transform.any_op {transform.readonly}) {

    %func_op = transform.structured.match ops{["func.func"]} in %module_op
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_op {
      // Test patterns
      transform.apply_patterns.vector.unroll_to_elements
      transform.apply_patterns.vector.unroll_from_elements
    } : !transform.any_op

    transform.yield
  }
}
