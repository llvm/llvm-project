module @transforms attributes { transform.with_named_sequence } {
  transform.named_sequence @fold_reshape_ops_by_expansion(%module: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.linalg.fold_reshape_ops_by_expansion
    } : !transform.any_op
    transform.yield
  }

  transform.named_sequence @fold_reshape_ops_by_expansion_unrestricted(%module: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.linalg.fold_reshape_ops_by_expansion fold_single_use_only = false
    } : !transform.any_op
    transform.yield
  }

  transform.named_sequence @fold_reshape_ops_by_expansion_single_use(%module: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.linalg.fold_reshape_ops_by_expansion fold_single_use_only = true
    } : !transform.any_op
    transform.yield
  }
}
