module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module:
  !transform.any_op{transform.consumed}) {
    // 1. TOSA --> Linalg
    %func_h_1 = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.any_op
    transform.apply_registered_pass "tosa-to-linalg"
      to %func_h_1 : (!transform.any_op) -> !transform.any_op

    // 2. Bufferize
    // As per BufferizationEnums.td, value 1 for `LayoutMapOption` corresponds
    // to `IdentityLayoutMap`.
    %module_bufferized = transform.bufferization.one_shot_bufferize %module
      { bufferize_function_boundaries=true,
        function_boundary_type_conversion = 1 : i32}
      : (!transform.any_op) -> !transform.op<"builtin.module">

    // 3. Apply BufferResultsToOutParams - otherwise the following error is raised:
    //    * "error: 'emitc.func' op cannot return array type"
    // "hoist-static-allocs" is an optional optimization step.
    %func_h_2 = transform.structured.match ops{["func.func"]} in %module_bufferized : (!transform.op<"builtin.module">) -> !transform.any_op
    %module_h_1 = transform.get_parent_op %func_h_2 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %module_results_as_out_param = transform.apply_registered_pass "buffer-results-to-out-params"
      with options = { "hoist-static-allocs" = true }
      to %module_h_1 : (!transform.any_op) -> !transform.any_op

    %module_final_no_linalg = transform.apply_registered_pass "convert-linalg-to-loops"
       to %module_results_as_out_param : (!transform.any_op) -> !transform.any_op

    // 4. Canonicalize - not strictly required
    transform.apply_patterns to %module_final_no_linalg {
      transform.apply_patterns.canonicalization
    } : !transform.any_op

    // FIXME: This causes a crash, hence its commented out. See:
    //  * https://github.com/llvm/llvm-project/issues/179247
    // %func_h_3 = transform.structured.match ops{["func.func"]} in %module_final_no_linalg
    //   : (!transform.any_op) -> !transform.any_op
    // %module_h_2 = transform.get_parent_op %func_h_3 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    // transform.apply_registered_pass "convert-to-emitc" to %module_h_2
    //   : (!transform.any_op) -> !transform.op<"builtin.module">

    transform.yield
  }
}
