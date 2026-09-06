// RUN: mlir-opt %s -split-input-file -test-eliminate-vector-masks=fixed-size | FileCheck %s

// The mask bound comes from the induction variable, so it is not constant and
// no fold removes it. Value-bounds analysis proves `%i <= 1020`, hence
// `1024 - %i >= 4`, so the mask is all-true.

// CHECK-LABEL: @eliminate_mask_bounded_by_induction_variable
//       CHECK:   %[[ALL_TRUE:.*]] = vector.constant_mask [4] : vector<4xi1>
//       CHECK:   vector.transfer_read {{.*}}, %[[ALL_TRUE]]
func.func @eliminate_mask_bounded_by_induction_variable(%t: tensor<1024xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1024 = arith.constant 1024 : index
  %f0 = arith.constant 0.0 : f32
  %r = scf.for %i = %c0 to %c1024 step %c4 iter_args(%acc = %f0) -> f32 {
    %rem = arith.subi %c1024, %i : index
    %mask = vector.create_mask %rem : vector<4xi1>
    %v = vector.transfer_read %t[%i], %f0, %mask : tensor<1024xf32>, vector<4xf32>
    %s = vector.reduction <add>, %v : vector<4xf32> into f32
    %n = arith.addf %acc, %s : f32
    scf.yield %n : f32
  }
  return %r : f32
}

// -----

// `max(%n, 4) >= 4` for any `%n`, so the mask is all-true even though the
// operand is dynamic.

// CHECK-LABEL: @eliminate_mask_with_dynamic_operand
//       CHECK:   %[[ALL_TRUE:.*]] = vector.constant_mask [4] : vector<4xi1>
//       CHECK:   vector.transfer_read {{.*}}, %[[ALL_TRUE]]
func.func @eliminate_mask_with_dynamic_operand(%t: tensor<?xf32>, %n: index) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %f0 = arith.constant 0.0 : f32
  %m = arith.maxsi %n, %c4 : index
  %mask = vector.create_mask %m : vector<4xi1>
  %v = vector.transfer_read %t[%c0], %f0, %mask : tensor<?xf32>, vector<4xf32>
  return %v : vector<4xf32>
}

// -----

// Nothing bounds `%n` from below, so the mask cannot be proven all-true.

// CHECK-LABEL: @negative_unbounded_operand
//   CHECK-NOT:   vector.constant_mask
//       CHECK:   vector.create_mask
func.func @negative_unbounded_operand(%t: tensor<?xf32>, %n: index) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32
  %mask = vector.create_mask %n : vector<4xi1>
  %v = vector.transfer_read %t[%c0], %f0, %mask : tensor<?xf32>, vector<4xf32>
  return %v : vector<4xf32>
}

// -----

// The lower bound of `max(%n, 2)` is 2, which is less than the mask dimension,
// so the mask is not all-true. Unlike the unbounded case above, a bound *is*
// computed here; it is the comparison against the mask size that rejects it.

// CHECK-LABEL: @negative_lower_bound_too_small
//   CHECK-NOT:   vector.constant_mask
//       CHECK:   vector.create_mask
func.func @negative_lower_bound_too_small(%t: tensor<?xf32>, %n: index) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %f0 = arith.constant 0.0 : f32
  %m = arith.maxsi %n, %c2 : index
  %mask = vector.create_mask %m : vector<4xi1>
  %v = vector.transfer_read %t[%c0], %f0, %mask : tensor<?xf32>, vector<4xf32>
  return %v : vector<4xf32>
}

// -----

// `max(%n, 4 * vscale) >= 4 * vscale`, which is exactly the runtime size of
// `vector<[4]xi1>`, so with a vscale range this mask is provably all-true.
// Without one a scalable dimension cannot be proven, so it must be left alone.

// CHECK-LABEL: @negative_scalable_dim_without_vscale_range
//   CHECK-NOT:   vector.constant_mask
//       CHECK:   vector.create_mask
func.func @negative_scalable_dim_without_vscale_range(%t: tensor<?xf32>, %n: index) -> vector<[4]xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %f0 = arith.constant 0.0 : f32
  %vscale = vector.vscale
  %c4_vscale = arith.muli %vscale, %c4 : index
  %m = arith.maxsi %n, %c4_vscale : index
  %mask = vector.create_mask %m : vector<[4]xi1>
  %v = vector.transfer_read %t[%c0], %f0, %mask : tensor<?xf32>, vector<[4]xf32>
  return %v : vector<[4]xf32>
}
