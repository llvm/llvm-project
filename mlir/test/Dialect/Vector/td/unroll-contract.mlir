module attributes {transform.with_named_sequence} {
  transform.named_sequence @unroll_contract(%module_op: !transform.any_op {transform.readonly}) {

    %func = transform.structured.match ops{["func.func"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %contract = transform.structured.match ops{["vector.contract"]} in %func : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %func {
      transform.apply_patterns.vector.unroll_contract [1, 8, 4]
    } : !transform.any_op

    transform.yield
  }
}
