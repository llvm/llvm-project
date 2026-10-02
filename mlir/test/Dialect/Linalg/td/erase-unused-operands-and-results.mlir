module @transforms attributes { transform.with_named_sequence } {
  transform.named_sequence @erase_unused_operands_and_results(%module: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %func {
      transform.apply_patterns.linalg.erase_unused_operands_and_results
    } : !transform.any_op

    transform.yield
  }
}
