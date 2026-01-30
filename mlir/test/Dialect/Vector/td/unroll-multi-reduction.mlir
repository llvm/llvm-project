module attributes {transform.with_named_sequence} {
  transform.named_sequence @unroll_multi_reduction(%module_op: !transform.any_op {transform.readonly}) {

    %func_op = transform.structured.match ops{["func.func"]} in %module_op
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_op {
      // Test patterns
      transform.apply_patterns.vector.unroll_multi_reduction
    } : !transform.any_op

    transform.yield
  }
  transform.named_sequence @unroll_multi_reduction_inner(%module_op: !transform.any_op {transform.readonly}) {

    %func_op = transform.structured.match ops{["func.func"]} in %module_op
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_op {
      // Test patterns
      transform.apply_patterns.vector.unroll_multi_reduction lowering_strategy = "innerreduction"
    } : !transform.any_op

    transform.yield
  }
}
