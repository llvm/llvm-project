module @transforms attributes { transform.with_named_sequence } {
  transform.named_sequence @vectorize_with_patterns(%module: !transform.any_op {transform.readonly}) {

    %0 = transform.structured.match ops{["linalg.generic"]} in %module : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 { vectorize_nd_extract } : (!transform.any_op) -> !transform.any_op

    transform.yield
   }
}
