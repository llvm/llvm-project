module attributes {transform.with_named_sequence} {
  transform.named_sequence @unroll_to_elements(%module_op: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %module_op
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.unroll_to_elements
    } : !transform.any_op
    transform.yield
  }
}
