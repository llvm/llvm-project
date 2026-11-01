module attributes {transform.with_named_sequence} {
  transform.named_sequence @linearize(%module: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.any_op
    transform.apply_conversion_patterns to %func {
      transform.apply_conversion_patterns.vector.linearize shuffle_like_ops = true
      transform.apply_conversion_patterns.scf.structural_conversions
    } with type_converter {
      transform.apply_conversion_patterns.vector.linearize_type_converter
    } <partial_conversion> : !transform.any_op
    transform.yield
  }

  transform.named_sequence @linearize_bitwidth_128(%module: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.any_op
    transform.apply_conversion_patterns to %func {
      transform.apply_conversion_patterns.vector.linearize shuffle_like_ops = true target_vector_bitwidth = 128
    } with type_converter {
      transform.apply_conversion_patterns.vector.linearize_type_converter
    } <partial_conversion> : !transform.any_op
    transform.yield
  }

  transform.named_sequence @linearize_bitwidth_0(%module: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.any_op
    transform.apply_conversion_patterns to %func {
      transform.apply_conversion_patterns.vector.linearize shuffle_like_ops = true target_vector_bitwidth = 0
    } with type_converter {
      transform.apply_conversion_patterns.vector.linearize_type_converter
    } <partial_conversion> : !transform.any_op
    transform.yield
  }
}
