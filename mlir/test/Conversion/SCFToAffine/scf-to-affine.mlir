// RUN: mlir-opt --raise-scf-to-affine --split-input-file %s | FileCheck %s

// CHECK-LABEL: @constant_step
// CHECK-SAME:  %[[ARR:.*]]: memref<?xi32>, %[[LB:.*]]: index, %[[UB:.*]]: index
// CHECK:         affine.for %[[IV:.*]] = %[[LB]] to %[[UB]] step 3 {
// CHECK:           memref.store %{{.*}}, %[[ARR]][%[[IV]]]
func.func @constant_step(%arr: memref<?xi32>, %lb: index, %ub: index) {
  %c0_i32 = arith.constant 0 : i32
  %c3 = arith.constant 3 : index
  scf.for %idx = %lb to %ub step %c3 {
    memref.store %c0_i32, %arr[%idx] : memref<?xi32>
  }
  return
}

// -----

// CHECK: #[[$UB_MAP:.+]] = affine_map<()[s0, s1, s2] -> ((s0 - s1 + s2 - 1) floordiv s0)>
// CHECK: #[[$IV_MAP:.+]] = affine_map<(d0, d1)[s0] -> (d0 + d1 * s0)>
// CHECK-LABEL: @dynamic_step
// CHECK-SAME:  %[[ARR:.*]]: memref<?xi32>, %[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index
// CHECK:         affine.for %[[IV:.*]] = 0 to #[[$UB_MAP]]()[%[[STEP]], %[[LB]], %[[UB]]] {
// CHECK:           %[[IDX:.*]] = affine.apply #[[$IV_MAP]](%[[LB]], %[[IV]])[%[[STEP]]]
// CHECK:           memref.store %{{.*}}, %[[ARR]][%[[IDX]]]
func.func @dynamic_step(%arr: memref<?xi32>, %lb: index, %ub: index, %step: index) {
  %c0_i32 = arith.constant 0 : i32
  scf.for %idx = %lb to %ub step %step {
    memref.store %c0_i32, %arr[%idx] : memref<?xi32>
  }
  return
}

// -----

// CHECK-LABEL: @nested_loop
// CHECK-SAME:  %[[ARR:.*]]: memref<?x?xi32>, %[[UB1:.*]]: index, %[[UB2:.*]]: index
// CHECK:         affine.for %[[I:.*]] = 0 to %[[UB1]] {
// CHECK:           affine.for %[[J:.*]] = 0 to %[[UB2]] {
// CHECK:             memref.store %{{.*}}, %[[ARR]][%[[I]], %[[J]]]
func.func @nested_loop(%arg0: memref<?x?xi32>, %ub1: index, %ub2: index) {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %ub1 step %c1 {
    scf.for %j = %c0 to %ub2 step %c1 {
      memref.store %c0_i32, %arg0[%i, %j] : memref<?x?xi32>
    }
  }
  return
}

// -----

// CHECK-LABEL: @index_cast_simple
// CHECK-SAME:  %[[LB:.*]]: i32, %[[UB:.*]]: i32
// CHECK:         %[[LB_IDX:.*]] = arith.index_cast %[[LB]] : i32 to index
// CHECK:         %[[UB_IDX:.*]] = arith.index_cast %[[UB]] : i32 to index
// CHECK:         affine.for %[[IV:.*]] = 0 to #{{.*}}%[[LB_IDX]]{{.*}}%[[UB_IDX]]
// CHECK:           %[[IDX:.*]] = affine.apply #{{.*}}%[[LB_IDX]]{{.*}}%[[IV]]
// CHECK:           %[[IV_I32:.*]] = arith.index_cast %[[IDX]] : index to i32
// CHECK:           func.call @some_func(%[[IV_I32]])

func.func private @some_func(%arg: i32)

func.func @index_cast_simple(%lb: i32, %ub: i32) {
  %step = arith.constant 1 : i32
  scf.for %i = %lb to %ub step %step : i32 {
    func.call @some_func(%i) : (i32) -> ()
  }
  return
}

// -----

// CHECK-LABEL: @index_cast_unsigned
// CHECK-SAME:  %[[LB:.*]]: i32, %[[UB:.*]]: i32
// CHECK:         %[[LB_IDX:.*]] = arith.index_castui %[[LB]] : i32 to index
// CHECK:         %[[UB_IDX:.*]] = arith.index_castui %[[UB]] : i32 to index
// CHECK:         affine.for
// CHECK:           %[[IV_I32:.*]] = arith.index_castui %{{.*}} : index to i32
// CHECK:           func.call @some_func(%[[IV_I32]])

func.func private @some_func(%arg: i32)

func.func @index_cast_unsigned(%lb: i32, %ub: i32) {
  %step = arith.constant 1 : i32
  scf.for unsigned %i = %lb to %ub step %step : i32 {
    func.call @some_func(%i) : (i32) -> ()
  }
  return
}

// -----

// CHECK-LABEL:   func.func @nested_loop_index_cast(
// CHECK-SAME:      %[[UB1:.*]]: i16, %[[UB2:.*]]: i16) {
// NOTE: index casts for *all* upper bounds deliberately hoisted to top-level
// CHECK:           %[[UB1_IDX:.*]] = arith.index_cast %[[UB1]] : i16 to index
// CHECK:           %[[UB2_IDX:.*]] = arith.index_cast %[[UB2]] : i16 to index
// CHECK:           affine.for %[[I_NEW:.*]] = 0 to %[[UB1_IDX]] {
// CHECK:             %[[I_OLD_IDX:.*]] = affine.apply #{{.*}}%[[I_NEW]]
// CHECK:             %[[I_OLD:.*]] = arith.index_cast %[[I_OLD_IDX]] : index to i16
// CHECK:             affine.for %[[J_NEW:.*]] = 0 to %[[UB2_IDX]] {
// CHECK:               %[[J_OLD_IDX:.*]] = affine.apply #{{.*}}%[[J_NEW]]
// CHECK:               %[[J_OLD:.*]] = arith.index_cast %[[J_OLD_IDX]] : index to i16
// CHECK:               func.call @some_func(%[[I_OLD]], %[[J_OLD]]) : (i16, i16) -> ()

func.func private @some_func(%i: i16, %j: i16)

func.func @nested_loop_index_cast(%ub1: i16, %ub2: i16) {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : i16
  %c1 = arith.constant 1 : i16
  scf.for %i = %c0 to %ub1 step %c1 : i16{
    scf.for %j = %c0 to %ub2 step %c1 : i16 {
      func.call @some_func(%i, %j) : (i16, i16) -> ()
    }
  }
  return
}

// -----

// CHECK: #[[$LB_MAP:.+]] = affine_map<(d0) -> (0, -d0 + 3)>
// CHECK: #[[$UB_MAP:.+]] = affine_map<(d0) -> (3, -d0 + 10)>
// CHECK-LABEL:   func.func @constant_step_non_rectangular_nest
// CHECK:           affine.for %[[I:.*]] = 0 to 10 {
// CHECK:             affine.for %[[J:.*]] = max #[[$LB_MAP]](%[[I]]) to min #[[$UB_MAP]](%[[I]]) {
// CHECK:               func.call @some_func(%[[I]], %[[J]]) : (index, index) -> ()

#lbs = affine_map<(i)[K, N] -> (0, K - i)>
#ubs = affine_map<(i)[K, N] -> (K, N - i)>

func.func private @some_func(%i: index, %j: index)

func.func @constant_step_non_rectangular_nest() {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index

  %N = arith.constant 10 : index
  %K = arith.constant 3 : index

  scf.for %i = %zero to %N step %one {
    %lb = affine.max #lbs(%i)[%K, %N] // NOTE: %lb is *not* a dimension.
    %ub = affine.min #ubs(%i)[%K, %N] // NOTE: %ub is *not* a dimension.
    scf.for %j = %lb to %ub step %one {
      func.call @some_func(%i, %j) : (index, index) -> ()
    }
  }

  return
}

// -----

// CHECK: #[[$UB_MAP:.+]] = affine_map<(d0)[s0] -> ((s0 + 5) floordiv s0, (-d0 + s0 + 98) floordiv s0)>
// CHECK: #[[$IV_MAP:.+]] = affine_map<(d0, d1)[s0] -> (d0 + d1 * s0)>
// CHECK-LABEL:   func.func @dynamic_step_non_rectangular_nest(
// CHECK-SAME:      %[[INNER_STEP:.*]]: index) {
// CHECK:           %[[ONE:.*]] = arith.constant 1 : index
// CHECK:           affine.for %[[I:.*]] = 0 to 100 {
// CHECK:             affine.for %[[J:.*]] = 0 to min #[[$UB_MAP]](%[[I]])[%[[INNER_STEP]]] {
// CHECK:               %[[OLD_IV:.*]] = affine.apply #[[$IV_MAP]](%[[ONE]], %[[J]])[%[[INNER_STEP]]]
// CHECK:               func.call @some_func(%[[I]], %[[OLD_IV]]) : (index, index) -> ()

#ub_map = affine_map<(i)[K, N] -> (K, N - i)>

func.func private @some_func(%i: index, %j: index)

func.func @dynamic_step_non_rectangular_nest(%inner_step: index) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index

  %N = arith.constant 100 : index
  %K = arith.constant 7 : index

  scf.for %i = %zero to %N step %one {
    // NOTE: lower bounds cannot be a max in general if step is not constant.
    %ub = affine.min #ub_map(%i)[%K, %N] // NOTE: %ub is *not* a dimension.
    scf.for %j = %one to %ub step %inner_step {
      func.call @some_func(%i, %j) : (index, index) -> ()
    }
  }

  return
}

// -----

// CHECK-LABEL:   func.func @with_iter_args_simple(
// CHECK-SAME:  %[[LB:.*]]: index, %[[UB:.*]]: index, %[[INIT:.*]]: f32
// CHECK:           %[[RESULT:.*]] = affine.for %[[I:.*]] = %[[LB]] to %[[UB]] iter_args(%[[ACC:.*]] = %[[INIT]]) -> (f32) {
// CHECK:             %[[NEXT_ACC:.*]] = arith.addf %[[ACC]], %[[ACC]] : f32
// CHECK:             affine.yield %[[NEXT_ACC]] : f32
// CHECK:           }
// CHECK:           return %[[RESULT]] : f32
func.func @with_iter_args_simple(%lb: index, %ub: index, %init: f32) -> f32 {
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %lb to %ub step %c1 iter_args(%acc = %init) -> (f32) {
    %v = arith.addf %acc, %acc : f32
    scf.yield %v : f32
  }
  return %r : f32
}

// -----

// CHECK: #[[$LB_MAP:.+]] = affine_map<()[s0] -> (0, s0)>
// CHECK: #[[$UB_MAP:.+]] = affine_map<()[s0, s1] -> ((s0 - s1 + 99) floordiv s0)>
// CHECK: #[[$IV_MAP:.+]] = affine_map<(d0, d1)[s0] -> (d0 + d1 * s0)>
// CHECK-LABEL:   func.func @max_lb_symbol_dynamic_step(
// CHECK-SAME:      %[[STEP:.*]]: index, %[[K:.*]]: index) {
// CHECK:           %[[LB:.*]] = affine.max #[[$LB_MAP]]()[%[[K]]]
// CHECK:           affine.for %[[NEW_I:.*]] = 0 to #[[$UB_MAP]]()[%[[STEP]], %[[LB]]] {
// CHECK:             %[[OLD_I:.*]] = affine.apply #[[$IV_MAP]](%[[LB]], %[[NEW_I]])[%[[STEP]]]
// CHECK:             func.call @some_func(%[[OLD_I]]) : (index) -> ()

#lb_map = affine_map<()[K] -> (0, K)>

func.func private @some_func(%i: index)

func.func @max_lb_symbol_dynamic_step(%step: index, %K: index) {
  %ub = arith.constant 100 : index
  %lb = affine.max #lb_map()[%K]
  scf.for %i = %lb to %ub step %step {
    func.call @some_func(%i) : (index) -> ()
  }
  return
}

// -----

// CHECK-LABEL: @max_lb_iv_dynamic_step_not_raised
// CHECK:         affine.for %{{.*}} = 0 to %{{.*}} {
// CHECK:           scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NOT:     affine.for

#lb_map = affine_map<(i)[K] -> (0, K - i)>

func.func private @some_func(%i: index, %j: index)

func.func @max_lb_iv_dynamic_step_not_raised(%n: index, %ub: index, %k: index,
                                             %step: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %lb = affine.max #lb_map(%i)[%k] // this lb prevents the inner scf.for from raising
    scf.for %j = %lb to %ub step %step {
      func.call @some_func(%i, %j) : (index, index) -> ()
    }
  }
  return
}

// -----

// CHECK-LABEL:   func.func @index_cast_with_iter_args(
// CHECK-SAME:      %[[LB:.*]]: i32, %[[UB:.*]]: i32, %[[STEP:.*]]: i32, %[[INIT:.*]]: f32) -> f32 {
// CHECK:           %[[LB_IDX:.*]] = arith.index_cast %[[LB]] : i32 to index
// CHECK:           %[[UB_IDX:.*]] = arith.index_cast %[[UB]] : i32 to index
// CHECK:           %[[STEP_IDX:.*]] = arith.index_cast %[[STEP]] : i32 to index
// CHECK:           %[[RESULT:.*]] = affine.for %[[IV_NEW:.*]] = 0 to {{.*}} iter_args(%[[ACC:.*]] = %[[INIT]]) -> (f32) {
// CHECK:             %[[IV_OLD_IDX:.*]] = affine.apply #{{.*}}%[[LB_IDX]]{{.*}}%[[IV_NEW]]{{.*}}%[[STEP_IDX]]
// CHECK:             %[[IV_OLD:.*]] = arith.index_cast %[[IV_OLD_IDX]] : index to i32
// CHECK:             %[[VAL:.*]] = arith.sitofp %[[IV_OLD]] : i32 to f32
// CHECK:             %[[ACC_NEXT:.*]] = arith.addf %[[ACC]], %[[VAL]] : f32
// CHECK:             affine.yield %[[ACC_NEXT]] : f32
// CHECK:           }
// CHECK:           return %[[RESULT]] : f32

func.func @index_cast_with_iter_args(%lb: i32, %ub: i32, %step: i32, %init: f32) -> f32 {
  %r = scf.for %i = %lb to %ub step %step iter_args(%acc = %init) -> (f32) : i32 {
    %f = arith.sitofp %i : i32 to f32
    %v = arith.addf %acc, %f : f32
    scf.yield %v : f32
  }
  return %r : f32
}

// -----

// CHECK-LABEL:   @wider_than_index_not_raised
// CHECK:           scf.for %{{.*}} : i64
// CHECK-NOT:       affine.for

// Use dlti dialect to pin width(index) == 32.
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>} {
  func.func private @some_func(%arg: i64)

  func.func @wider_than_index_not_raised(%lb: i64, %ub: i64, %step: i64) {
    scf.for %i = %lb to %ub step %step : i64 {
      func.call @some_func(%i) : (i64) -> ()
    }
    return
  }
}

// -----

// CHECK-LABEL:   @unsigned_same_width_not_raised
// CHECK:           scf.for unsigned %{{.*}} : i32
// CHECK-NOT:       affine.for

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>} {
  func.func private @some_func(%arg: i32)

  func.func @unsigned_same_width_not_raised(%lb: i32, %ub: i32, %step: i32) {
    scf.for unsigned %i = %lb to %ub step %step : i32 {
      func.call @some_func(%i) : (i32) -> ()
    }
    return
  }
}

// -----

// CHECK-LABEL:   @signed_same_width_raised
// CHECK:           affine.for
// CHECK-NOT:       scf.for

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>} {
  func.func private @some_func(%arg: i32)

  func.func @signed_same_width_raised(%lb: i32, %ub: i32, %step: i32) {
    scf.for %i = %lb to %ub step %step : i32 {
      func.call @some_func(%i) : (i32) -> ()
    }
    return
  }
}