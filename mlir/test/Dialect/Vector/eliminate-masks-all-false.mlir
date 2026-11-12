// RUN: mlir-opt %s -split-input-file -eliminate-vector-masks | FileCheck %s

// `4 - 4 * vscale <= 0` as vscale >= 1. One nonpositive dimension makes the
// whole mask all-false, even though `%dim` is unknown.
// CHECK-LABEL: @eliminate_all_false_masks
// CHECK: %[[FALSE:.*]] = vector.constant_mask [0, 0] : vector<[4]x[4]xi1>
// CHECK: return %[[FALSE]] : vector<[4]x[4]xi1>
func.func @eliminate_all_false_masks(%dim: index) -> vector<[4]x[4]xi1> {
  %c4 = arith.constant 4 : index
  %vscale = vector.vscale
  %width = arith.muli %vscale, %c4 : index
  %remaining = arith.subi %c4, %width : index
  %mask = vector.create_mask %dim, %remaining : vector<[4]x[4]xi1>
  return %mask : vector<[4]x[4]xi1>
}

// -----

// Fixed-size equivalent of the above. Value bounds proves `%iv >= 4`, hence
// `4 - %iv <= 0`.
// CHECK-LABEL: @eliminate_all_false_masks_fixed_size
// CHECK: %[[FALSE:.*]] = vector.constant_mask [0] : vector<4xi1>
// CHECK: vector.maskedstore {{.*}}, %[[FALSE]],
func.func @eliminate_all_false_masks_fixed_size(%out: memref<16xf32>, %value: vector<4xf32>) {
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  scf.for %iv = %c4 to %c16 step %c4 {
    %remaining = arith.subi %c4, %iv : index
    %mask = vector.create_mask %remaining : vector<4xi1>
    vector.maskedstore %out[%iv], %mask, %value : memref<16xf32>, vector<4xi1>, vector<4xf32>
  }
  return
}

// -----

// CHECK-LABEL: @trivially_all_false_case
// CHECK: %[[FALSE:.*]] = vector.constant_mask [0, 0] : vector<2x[4]xi1>
// CHECK: return %[[FALSE]] : vector<2x[4]xi1>
func.func @trivially_all_false_case(%dim: index) -> vector<2x[4]xi1> {
  %cneg4 = arith.constant -4 : index
  %vscale = vector.vscale
  %cneg4_vscale = arith.muli %vscale, %cneg4 : index
  // Is found to be all-false _without_ value bounds analysis.
  %mask = vector.create_mask %dim, %cneg4_vscale : vector<2x[4]xi1>
  return %mask : vector<2x[4]xi1>
}

// -----

// CHECK-LABEL: @trivially_all_false_case_fixed_size
// CHECK: %[[FALSE:.*]] = vector.constant_mask [0, 0] : vector<2x4xi1>
// CHECK: return %[[FALSE]] : vector<2x4xi1>
func.func @trivially_all_false_case_fixed_size(%dim: index) -> vector<2x4xi1> {
  %c0 = arith.constant 0 : index
  // Also found to be all-false _without_ value bounds analysis.
  %mask = vector.create_mask %dim, %c0 : vector<2x4xi1>
  return %mask : vector<2x4xi1>
}

// -----

// CHECK-LABEL: @negative_value_bounds_scalable_dim_not_all_false
// CHECK-NOT: vector.constant_mask
// CHECK: vector.create_mask
func.func @negative_value_bounds_scalable_dim_not_all_false() -> vector<[4]xi1> {
  %c4 = arith.constant 4 : index
  %vscale = vector.vscale
  %width = arith.muli %vscale, %c4 : index
  // `4 * vscale - 4` is zero for vscale = 1, but positive for any larger
  // vscale.
  %remaining = arith.subi %width, %c4 : index
  %mask = vector.create_mask %remaining : vector<[4]xi1>
  return %mask : vector<[4]xi1>
}
