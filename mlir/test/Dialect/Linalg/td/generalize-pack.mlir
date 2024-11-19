module @transforms attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
    %pack = transform.structured.match ops{["tensor.pack"]} in %module : (!transform.any_op) -> !transform.any_op

    %1 = transform.get_parent_op %pack {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %1 {
      transform.apply_patterns.linalg.generalize_pack_unpack
    } : !transform.any_op

    transform.yield
  }
}
