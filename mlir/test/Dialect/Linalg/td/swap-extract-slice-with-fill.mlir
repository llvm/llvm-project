module @transforms attributes { transform.with_named_sequence } {
  transform.named_sequence @swap_extract_slice_with_fill(%module: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %func {
      transform.apply_patterns.linalg.swap_extract_slice_with_fill
    } : !transform.any_op

    transform.yield
  }
}
