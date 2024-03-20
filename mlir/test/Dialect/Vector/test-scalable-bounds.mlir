// RUN: mlir-opt %s -test-affine-reify-value-bounds -cse -verify-diagnostics \
// RUN:   -verify-diagnostics -split-input-file | FileCheck %s

#map_dim_i = affine_map<(d0)[s0] -> (-d0 + 32400, s0)>
#map_dim_j = affine_map<(d0)[s0] -> (-d0 + 16, s0)>

// Here the upper bound for min_i is 4 x vscale, as we know 4 x vscale is
// always less than 32400. The bound for min_j is 16, as 16 is always less
// 4 x vscale_max (vscale_max is the UB for vscale).

// CHECK: #[[$SCALABLE_BOUND_MAP_0:.*]] = affine_map<()[s0] -> (s0 * 4)>

// CHECK-LABEL: @fixed_size_loop_nest
//   CHECK-DAG:   %[[VSCALE:.*]] = vector.vscale
//   CHECK-DAG:   %[[UB_i:.*]] = affine.apply #[[$SCALABLE_BOUND_MAP_0]]()[%[[VSCALE]]]
//   CHECK-DAG:   %[[UB_j:.*]] = arith.constant 16 : index
//       CHECK:   "test.some_use"(%[[UB_i]], %[[UB_j]]) : (index, index) -> ()
func.func @fixed_size_loop_nest() {
  %c16 = arith.constant 16 : index
  %c32400 = arith.constant 32400 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %vscale = vector.vscale
  %c4_vscale = arith.muli %vscale, %c4 : index
  scf.for %i = %c0 to %c32400 step %c4_vscale {
    %min_i = affine.min #map_dim_i(%i)[%c4_vscale]
    scf.for %j = %c0 to %c16 step %c4_vscale {
      %min_j = affine.min #map_dim_j(%j)[%c4_vscale]
      %bound_i = "test.reify_scalable_bound"(%min_i) {type = "UB", vscale_min = 1, vscale_max = 16} : (index) -> index
      %bound_j = "test.reify_scalable_bound"(%min_j) {type = "UB", vscale_min = 1, vscale_max = 16} : (index) -> index
      "test.some_use"(%bound_i, %bound_j) : (index, index) -> ()
    }
  }
  return
}

// -----

#map_dynamic_dim = affine_map<(d0)[s0, s1] -> (-d0 + s1, s0)>

// Here upper bounds for both min_i and min_j are both (conservatively)
// 4 x vscale, as we know that is always the largest value they could take. As
// if `dim < 4 x vscale` then 4 x vscale is an overestimate, and if
// `dim > 4 x vscale` then the min will be clamped to 4 x vscale.

// CHECK: #[[$SCALABLE_BOUND_MAP_1:.*]] = affine_map<()[s0] -> (s0 * 4)>

// CHECK-LABEL: @dynamic_size_loop_nest
//       CHECK:   %[[VSCALE:.*]] = vector.vscale
//       CHECK:   %[[UB_ij:.*]] = affine.apply #[[$SCALABLE_BOUND_MAP_1]]()[%[[VSCALE]]]
//       CHECK:   "test.some_use"(%[[UB_ij]], %[[UB_ij]]) : (index, index) -> ()
func.func @dynamic_size_loop_nest(%dim0: index, %dim1: index) {
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %vscale = vector.vscale
  %c4_vscale = arith.muli %vscale, %c4 : index
  scf.for %i = %c0 to %dim0 step %c4_vscale {
    %min_i = affine.min #map_dynamic_dim(%i)[%c4_vscale, %dim0]
    scf.for %j = %c0 to %dim1 step %c4_vscale {
      %min_j = affine.min #map_dynamic_dim(%j)[%c4_vscale, %dim1]
      %bound_i = "test.reify_scalable_bound"(%min_i) {type = "UB", vscale_min = 1, vscale_max = 16} : (index) -> index
      %bound_j = "test.reify_scalable_bound"(%min_j) {type = "UB", vscale_min = 1, vscale_max = 16} : (index) -> index
      "test.some_use"(%bound_i, %bound_j) : (index, index) -> ()
    }
  }
  return
}

// -----

// Here the bound is just a value + a constant.

// CHECK: #[[$SCALABLE_BOUND_MAP_2:.*]] = affine_map<()[s0] -> (s0 + 8)>

// CHECK-LABEL: @add_to_vscale
//       CHECK:   %[[VSCALE:.*]] = vector.vscale
//       CHECK:   %[[SCALABLE_BOUND:.*]] = affine.apply #[[$SCALABLE_BOUND_MAP_2]]()[%[[VSCALE]]]
//       CHECK:   "test.some_use"(%[[SCALABLE_BOUND]]) : (index) -> ()
func.func @add_to_vscale() {
  %vscale = vector.vscale
  %c8 = arith.constant 8 : index
  %vscale_plus_c8 = arith.addi %vscale, %c8 : index
  %bound = "test.reify_scalable_bound"(%vscale_plus_c8) {type = "EQ", vscale_min = 1, vscale_max = 16} : (index) -> index
  "test.some_use"(%bound) : (index) -> ()
  return
}

// -----

// Here we know vscale is always 2 so we get a constant bound.

// CHECK-LABEL: @vscale_fixed_size
//       CHECK:   %[[C2:.*]] = arith.constant 2 : index
//       CHECK:   "test.some_use"(%[[C2]]) : (index) -> ()
func.func @vscale_fixed_size() {
  %vscale = vector.vscale
  %bound = "test.reify_scalable_bound"(%vscale) {type = "EQ", vscale_min = 2, vscale_max = 2} : (index) -> index
  "test.some_use"(%bound) : (index) -> ()
  return
}

// -----

// Here we don't know the upper bound (%a is underspecified)

func.func @unknown_bound(%a: index) {
  %vscale = vector.vscale
  %vscale_plus_a = arith.muli %vscale, %a : index
  // expected-error @below{{could not reify bound}}
  %bound = "test.reify_scalable_bound"(%vscale_plus_a) {type = "UB", vscale_min = 1, vscale_max = 16} : (index) -> index
  "test.some_use"(%bound) : (index) -> ()
  return
}

// -----

// Here we have two vscale values (that have not been CSE'd), but they should
// still be treated as equivalent.

// CHECK: #[[$SCALABLE_BOUND_MAP_3:.*]] = affine_map<()[s0] -> (s0 * 6)>

// CHECK-LABEL: @duplicate_vscale_values
//       CHECK:   %[[VSCALE:.*]] = vector.vscale
//       CHECK:   %[[SCALABLE_BOUND:.*]] = affine.apply #[[$SCALABLE_BOUND_MAP_3]]()[%[[VSCALE]]]
//       CHECK:   "test.some_use"(%[[SCALABLE_BOUND]]) : (index) -> ()
func.func @duplicate_vscale_values() {
  %c4 = arith.constant 4 : index
  %vscale_0 = vector.vscale

  %c2 = arith.constant 2 : index
  %vscale_1 = vector.vscale

  %c4_vscale = arith.muli %vscale_0, %c4 : index
  %c2_vscale = arith.muli %vscale_1, %c2 : index
  %add = arith.addi %c2_vscale, %c4_vscale : index

  %bound = "test.reify_scalable_bound"(%add) {type = "EQ", vscale_min = 1, vscale_max = 16} : (index) -> index
  "test.some_use"(%bound) : (index) -> ()
  return
}

// -----

// Test some non-scalable code to ensure that works too:

#map_dim_i = affine_map<(d0)[s0] -> (-d0 + 1024, s0)>

// CHECK-LABEL: @non_scalable_code
//       CHECK:   %[[C4:.*]] = arith.constant 4 : index
//       CHECK:   "test.some_use"(%[[C4]]) : (index) -> ()
func.func @non_scalable_code() {
  %c1024 = arith.constant 1024 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  scf.for %i = %c0 to %c1024 step %c4 {
    %min_i = affine.min #map_dim_i(%i)[%c4]
    %bound_i = "test.reify_scalable_bound"(%min_i) {type = "UB", vscale_min = 1, vscale_max = 16} : (index) -> index
    "test.some_use"(%bound_i) : (index) -> ()
  }
  return
}
