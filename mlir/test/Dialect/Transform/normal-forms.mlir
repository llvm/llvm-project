// RUN: mlir-opt %s --split-input-file --verify-diagnostics --transform-interpreter | FileCheck %s

// expected-error @below {{normal form test_single_block_normal_form requires payload operations to have a single region}}
func.func private @empty()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.structured.match attributes {sym_name = "empty"} in %arg0 : (!transform.any_op) -> !transform.normalized_op<#transform.test_single_block_normal_form<nested false>>
    transform.yield
  }
}

// -----

// expected-remark @below {{found}}
func.func private @single() {
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match attributes {sym_name = "single"} in %arg0 : (!transform.any_op) -> !transform.normalized_op<#transform.test_single_block_normal_form<nested false>>
    transform.debug.emit_remark_at %0, "found" : !transform.normalized_op<#transform.test_single_block_normal_form<nested false>>
    transform.yield
  }
}

// -----

// expected-error @below {{normal form test_single_block_normal_form requires payload operations to have a single region}}
func.func private @branchy() {
  cf.br ^bb1
^bb1:
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.structured.match attributes {sym_name = "branchy"} in %arg0 : (!transform.any_op) -> !transform.normalized_op<#transform.test_single_block_normal_form<nested false>>
    transform.yield
  }
}

// -----

// expected-remark @below {{found}}
func.func private @nested() {
  scf.execute_region {
    cf.br ^bb1
  ^bb1:
    scf.yield
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match attributes {sym_name = "nested"} in %arg0 : (!transform.any_op) -> !transform.normalized_op<#transform.test_single_block_normal_form<nested false>>
    transform.debug.emit_remark_at %0, "found" : !transform.normalized_op<#transform.test_single_block_normal_form<nested false>>
    transform.yield
  }
}

// -----

func.func private @nested() {
  // expected-error @below {{normal form test_single_block_normal_form requires payload operations to have a single region}}
  scf.execute_region {
    cf.br ^bb1
  ^bb1:
    scf.yield
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.structured.match attributes {sym_name = "nested"} in %arg0 : (!transform.any_op) -> !transform.normalized_op<#transform.test_single_block_normal_form<nested true>>
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  // expected-remark @below {{matched}}
  transform.payload attributes {
      normal_forms = [#transform.test_single_block_normal_form<nested true>]} {
    transform.test_dummy_payload_op : () -> ()
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["transform.payload"]} in %arg0
        : (!transform.any_op) ->
            !transform.normalized_op<#transform.test_single_block_normal_form<nested true>>
    transform.debug.emit_remark_at %0, "matched"
        : !transform.normalized_op<#transform.test_single_block_normal_form<nested true>>
    transform.yield
  }
}

// -----

// expected-note @below {{previous instance}}
// expected-error @below {{duplicate normal form}}
transform.payload attributes {normal_forms = [
    #transform.test_single_block_normal_form<nested false>,
    #transform.test_single_block_normal_form<nested true>]} {
}

// -----

transform.payload attributes {
    normal_forms = [#transform.test_single_block_normal_form<nested true>]} {
  // expected-error @below {{normal form test_single_block_normal_form requires payload operations to have a single region}}
  "test.foo"() ({
    cf.br ^bb1
  ^bb1:
    "test.bar"() : () -> ()
  }) : () -> ()
}

// -----

transform.payload attributes {
    normal_forms = [#transform.test_single_block_normal_form<nested true>]} {
  // We should see the diagnostic from the inner op verifier, and never hit
  // the normal form check.
  // expected-error @below {{fail_to_verify is set}}
  transform.test_dummy_payload_op {fail_to_verify} : () -> ()
  "test.foo"() ({
    cf.br ^bb1
  ^bb1:
    "test.bar"() : () -> ()
  }) : () -> ()
}

// -----

// We have surprisingly many invocations of the verifier here:
//  1. after the initial parsing (reasonable)
//  2. in transform::detail::mergeSymbolsInto (looks excessive)
//  3. also in transform::detail::mergeSymbolsInto (has a TODO to be removed)
//  4. after the transform interpreter pass (reasonable)
//  5. before printing (generally reasonable, but would be nice to avoid if 
//     the IR is known-verified after by the pass manager).
// Notably this doesn't include an extra run from checkPayload, which is
// what we intend to test here.

// CHECK-LABEL: @verification_count
// CHECK: transform.payload
// CHECK-SAME: test.counting_normal_form_count = 5

module @verification_count attributes {transform.with_named_sequence} {
  transform.payload attributes {
      normal_forms = [#transform.test_counting_normal_form]} {
    transform.test_dummy_payload_op : () -> ()
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["transform.payload"]} in %arg0
        : (!transform.any_op) ->
            !transform.normalized_op<#transform.test_counting_normal_form>
    transform.yield
  }
}
