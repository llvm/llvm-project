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


// -----

// CHECK-LABEL: schedule_with_two_independent_choices_already_made
func.func @schedule_with_two_independent_choices_already_made(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32> {
//      CHECK-NOT: scf.forall
//      CHECK:     scf.for
//      CHECK-NOT:   scf.for
//      CHECK:       scf.forall
//      CHECK-NOT:   scf.for
//      CHECK:         tensor.extract_slice
//      CHECK:         tensor.extract_slice
//      CHECK:         tensor.extract_slice
//      CHECK:         linalg.matmul
//      CHECK:         scf.forall.in_parallel
//      CHECK:           tensor.parallel_insert_slice
//      CHECK:       tensor.insert_slice
//      CHECK:       scf.yield
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op

    %tiled_matmul = transform.tune.alternatives<"outer_par_or_seq_tiling"> selected_region = 0 -> !transform.any_op
    { // First alternative/region, with index = 0
      %contained_matmul, %loop = transform.structured.tile_using_for %matmul tile_sizes [8] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield %contained_matmul : !transform.any_op
    }, { // Second alternative/region, with index = 1
      %contained_matmul, %loop = transform.structured.tile_using_forall %matmul tile_sizes [8] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield %contained_matmul : !transform.any_op
    }

    transform.tune.alternatives<"inner_par_or_seq_tiling"> selected_region = 1 -> !transform.any_op {
      %contained_matmul, %loop = transform.structured.tile_using_for %tiled_matmul tile_sizes [0, 16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield %contained_matmul : !transform.any_op
    }, {
      %contained_matmul, %loop = transform.structured.tile_using_forall %tiled_matmul tile_sizes [0, 16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield %contained_matmul : !transform.any_op
    }

    transform.yield
  }
}

// -----

// CHECK-LABEL: subschedule_with_choice_resolved_in_main_schedule
func.func @subschedule_with_choice_resolved_in_main_schedule(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32> {
//      CHECK-NOT: scf.for
//      CHECK:     scf.forall
//      CHECK-NOT:   scf.forall
//      CHECK:       scf.for
//      CHECK-NOT:   scf.forall
//      CHECK:         tensor.extract_slice
//      CHECK:         tensor.extract_slice
//      CHECK:         tensor.extract_slice
//      CHECK:         linalg.matmul
//      CHECK:         tensor.insert_slice
//      CHECK:         scf.yield
//      CHECK:       scf.forall.in_parallel
//      CHECK:         tensor.parallel_insert_slice
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @subschedule_with_embedded_choice(%matmul: !transform.any_op {transform.readonly},
                                                             %par_or_seq: !transform.param<i64> {transform.readonly},
                                                             %tile_size: !transform.param<i64> {transform.readonly}) -> !transform.any_op {
    %tiled_matmul = transform.tune.alternatives<"par_or_seq_tiling"> selected_region = %par_or_seq : !transform.param<i64> -> !transform.any_op {
      %contained_matmul, %loop = transform.structured.tile_using_for %matmul tile_sizes [%tile_size] : (!transform.any_op, !transform.param<i64>) -> (!transform.any_op, !transform.any_op)
      transform.yield %contained_matmul : !transform.any_op
    }, {
      %contained_matmul, %loop = transform.structured.tile_using_forall %matmul tile_sizes [%tile_size] : (!transform.any_op, !transform.param<i64>) -> (!transform.any_op, !transform.any_op)
      transform.yield %contained_matmul : !transform.any_op
    }
    transform.yield %tiled_matmul : !transform.any_op
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %outer_par = transform.param.constant 1 -> !transform.param<i64>
    %outer_tile_size = transform.param.constant 32 -> !transform.param<i64>
    %inner_seq = transform.tune.knob<"inner_par_or_seq"> = 0 from options = [0, 1] -> !transform.param<i64>
    %inner_tile_size = transform.param.constant 8 -> !transform.param<i64>
    %tiled_matmul = transform.include @subschedule_with_embedded_choice failures(propagate) (%matmul, %outer_par, %outer_tile_size) : (!transform.any_op, !transform.param<i64>, !transform.param<i64>) -> !transform.any_op
    %tiled_tiled_matmul = transform.include @subschedule_with_embedded_choice failures(propagate) (%tiled_matmul, %inner_seq, %inner_tile_size) : (!transform.any_op, !transform.param<i64>, !transform.param<i64>) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: eeny_meeny_miny_moe
func.func private @eeny_meeny_miny_moe()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op

    %tiled_matmul = transform.tune.alternatives<"4way"> selected_region = 3 -> !transform.any_param
    { // First alternative/region, with index = 0
      %out = transform.param.constant "eeny" -> !transform.any_param
      transform.yield %out : !transform.any_param
    }, { // Second alternative/region, with index = 1
      %out = transform.param.constant "meeny" -> !transform.any_param
      transform.yield %out : !transform.any_param
    }, { // Third alternative/region, with index = 2
      %out = transform.param.constant "miny" -> !transform.any_param
      transform.yield %out : !transform.any_param
    }, { // Fourth alternative/region, with index = 3
      %out = transform.param.constant "moe" -> !transform.any_param
      transform.yield %out : !transform.any_param
    }
    transform.yield
  }
}