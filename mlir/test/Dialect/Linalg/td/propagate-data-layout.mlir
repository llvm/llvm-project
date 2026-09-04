module @transforms attributes { transform.with_named_sequence } {
  transform.named_sequence @propagate_data_layout(%module: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %func {
      transform.apply_patterns.linalg.data_layout_propagation {poison_padding = true}
      transform.apply_patterns.linalg.extract_slice_sinking
    } : !transform.any_op

    transform.yield
  }
}
