module @transforms attributes { transform.with_named_sequence } {
  transform.named_sequence @drop_unit_dims(%module: !transform.any_op {transform.readonly}) {

    %func_op = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.drop_inner_most_unit_dims_from_xfer_ops
    } : !transform.op<"func.func">

    transform.yield
   }
}
