// RUN: mlir-opt %s --transform-interpreter --split-input-file \
// RUN:     --verify-diagnostics | FileCheck %s

// CHECK-LABEL: @schedule_with_nondet_knobs
module attributes {transform.with_named_sequence} {
  transform.named_sequence @schedule_with_nondet_knobs(%arg0: !transform.any_op {transform.readonly}) {
    // CHECK: %[[HEADS_OR_TAILS:.*]] = transform.tune.knob<"coin"> options = [true, false] -> !transform.any_param
    %heads_or_tails = transform.tune.knob<"coin"> options = [true, false] -> !transform.any_param
    // CHECK: transform.tune.knob<"animal"> options = ["cat", "dog", unit] -> !transform.any_param
    %chosen_category = transform.tune.knob<"animal"> options = ["cat", "dog", unit] -> !transform.any_param
    // CHECK: transform.tune.knob<"tile_size"> options = [2, 4, 8, 16, 24, 32] -> !transform.any_param
    %chosen_tile_size = transform.tune.knob<"tile_size"> options = [2, 4, 8, 16, 24, 32] -> !transform.any_param
    // CHECK: transform.tune.knob<"magic_value"> options = [2.000000e+00 : f32, 2.250000e+00 : f32, 2.500000e+00 : f32, 2.750000e+00 : f32, 3.000000e+00 : f32] -> !transform.any_param
    %chosen_constant = transform.tune.knob<"magic_value"> options = [2.0 : f32, 2.25 : f32, 2.5 : f32, 2.75 : f32, 3.0 : f32] -> !transform.any_param
    // CHECK: transform.debug.emit_param_as_remark %[[HEADS_OR_TAILS]]
    transform.debug.emit_param_as_remark %heads_or_tails : !transform.any_param
    transform.yield
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    // Dummy sequence to appease -transform-interpreter invocation
    transform.yield
  }
}

// -----

// Schedule where non-determinism on knobs has been resolved by selecting a valid option.

// CHECK-LABEL: payload_for_schedule_with_selected_knobs
func.func private @payload_for_schedule_with_selected_knobs()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    // CHECK: %[[HEADS_OR_TAILS:.*]] = transform.tune.knob<"coin"> = true from options = [true, false] -> !transform.any_param
    %heads_or_tails = transform.tune.knob<"coin"> = true from options = [true, false] -> !transform.any_param
    // expected-remark@below {{true}}
    transform.debug.emit_param_as_remark %heads_or_tails : !transform.any_param

    // CHECK: transform.tune.knob<"animal"> = "dog" from options = ["cat", "dog", unit] -> !transform.any_param
    %chosen_category = transform.tune.knob<"animal"> = "dog" from options = ["cat", "dog", unit] -> !transform.any_param
    // CHECK: transform.tune.knob<"tile_size"> = 8 : i64 from options = [2, 4, 8, 16, 24, 32] -> !transform.any_param
    %chosen_tile_size = transform.tune.knob<"tile_size"> = 8 from options = [2, 4, 8, 16, 24, 32] -> !transform.any_param
    // CHECK: transform.tune.knob<"magic_value"> = 2.500000e+00 : f32 from options = [2.000000e+00 : f32, 2.250000e+00 : f32, 2.500000e+00 : f32, 2.750000e+00 : f32, 3.000000e+00 : f32] -> !transform.any_param
    %chosen_constant = transform.tune.knob<"magic_value"> = 2.5 : f32  from options = [2.0 : f32, 2.25 : f32, 2.5 : f32, 2.75 : f32, 3.0 : f32] -> !transform.any_param
    transform.yield
  }
}

// -----

// CHECK: #[[AFFINE_SET:.*]] = affine_set<(d0) : (d0 - 2 >= 0)>
// CHECK: payload_for_schedule_where_selected_knob_being_a_member_of_options_is_unverified
func.func private @payload_for_schedule_where_selected_knob_being_a_member_of_options_is_unverified()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    // CHECK: transform.tune.knob<"bounded"> = 4242 : i64 from options = #[[AFFINE_SET]] -> !transform.any_param
    %value_in_half_range = transform.tune.knob<"bounded"> = 4242 from options = affine_set<(d0) : (d0 - 2 >= 0)>  -> !transform.any_param
    transform.yield
  }
}
