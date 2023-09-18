// RUN: mlir-opt %s -convert-vector-to-arm-sme -split-input-file -allow-unregistered-dialect | FileCheck %s

// =============================================================================
// arith.constant dense<0> to arm_sme.zero
// =============================================================================

// -----

// CHECK-LABEL: @arith_constant_dense_2d_zero_i8
// CHECK: %[[ZERO:.*]] = arm_sme.zero : vector<[16]x[16]xi8>
func.func @arith_constant_dense_2d_zero_i8() {
  %zero = arith.constant dense<0> : vector<[16]x[16]xi8>
  "prevent.dce"(%zero) : (vector<[16]x[16]xi8>) -> ()
  return
}

// -----

// CHECK-LABEL: @arith_constant_dense_2d_zero_i16
// CHECK: %[[ZERO:.*]] = arm_sme.zero : vector<[8]x[8]xi16>
func.func @arith_constant_dense_2d_zero_i16() {
  %zero = arith.constant dense<0> : vector<[8]x[8]xi16>
  "prevent.dce"(%zero) : (vector<[8]x[8]xi16>) -> ()
  return
}

// -----

// CHECK-LABEL: @arith_constant_dense_2d_zero_i32
// CHECK: %[[ZERO:.*]] = arm_sme.zero : vector<[4]x[4]xi32>
func.func @arith_constant_dense_2d_zero_i32() {
  %zero = arith.constant dense<0> : vector<[4]x[4]xi32>
  "prevent.dce"(%zero) : (vector<[4]x[4]xi32>) -> ()
  return
}

// -----

// CHECK-LABEL: @arith_constant_dense_2d_zero_i64
// CHECK: %[[ZERO:.*]] = arm_sme.zero : vector<[2]x[2]xi64>
func.func @arith_constant_dense_2d_zero_i64() {
  %zero = arith.constant dense<0> : vector<[2]x[2]xi64>
  "prevent.dce"(%zero) : (vector<[2]x[2]xi64>) -> ()
  return
}

// -----

// CHECK-LABEL: @arith_constant_dense_2d_zero_f16
// CHECK: %[[ZERO:.*]] = arm_sme.zero : vector<[8]x[8]xf16>
func.func @arith_constant_dense_2d_zero_f16() {
  %zero = arith.constant dense<0.0> : vector<[8]x[8]xf16>
  "prevent.dce"(%zero) : (vector<[8]x[8]xf16>) -> ()
  return
}

// -----

// CHECK-LABEL: @arith_constant_dense_2d_zero_bf16
// CHECK: %[[ZERO:.*]] = arm_sme.zero : vector<[8]x[8]xbf16>
func.func @arith_constant_dense_2d_zero_bf16() {
  %zero = arith.constant dense<0.0> : vector<[8]x[8]xbf16>
  "prevent.dce"(%zero) : (vector<[8]x[8]xbf16>) -> ()
  return
}

// -----

// CHECK-LABEL: @arith_constant_dense_2d_zero_f32
// CHECK: %[[ZERO:.*]] = arm_sme.zero : vector<[4]x[4]xf32>
func.func @arith_constant_dense_2d_zero_f32() {
  %zero = arith.constant dense<0.0> : vector<[4]x[4]xf32>
  "prevent.dce"(%zero) : (vector<[4]x[4]xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: @arith_constant_dense_2d_zero_f64
// CHECK: %[[ZERO:.*]] = arm_sme.zero : vector<[2]x[2]xf64>
func.func @arith_constant_dense_2d_zero_f64() {
  %zero = arith.constant dense<0.0> : vector<[2]x[2]xf64>
  "prevent.dce"(%zero) : (vector<[2]x[2]xf64>) -> ()
  return
}

// =============================================================================
// Non-zero arith.constant dense to SME
// =============================================================================

// -----

// CHECK-LABEL: func.func @arith_constant_dense_2d_nonzero_i8() {
// CHECK: %[[C2_SPLAT:.*]] = arith.constant dense<2> : vector<[16]xi8>
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[C16:.*]] = arith.constant 16 : index
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[GET_TILE_ID:.*]] = arm_sme.get_tile_id : i8
// CHECK: %[[TILE:.*]] = arm_sme.cast_tile_to_vector %[[GET_TILE_ID]] : i8 to vector<[16]x[16]xi8>
// CHECK: %[[VSCALE:.*]] = vector.vscale
// CHECK: %[[NUM_TILE_SLICES:.*]] = arith.muli %[[VSCALE]], %[[C16]] : index
// CHECK: scf.for %[[TILE_SLICE_INDEX:.*]] = %[[C0]] to %[[NUM_TILE_SLICES]] step %[[C1]] {
// CHECK:   arm_sme.move_vector_to_tile_slice %[[C2_SPLAT]], %[[TILE]], %[[TILE_SLICE_INDEX]] : vector<[16]xi8> into vector<[16]x[16]xi8>
// CHECK: "prevent.dce"(%[[TILE]]) : (vector<[16]x[16]xi8>) -> ()
func.func @arith_constant_dense_2d_nonzero_i8() {
  %two = arith.constant dense<2> : vector<[16]x[16]xi8>
  "prevent.dce"(%two) : (vector<[16]x[16]xi8>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @arith_constant_dense_2d_nonzero_f64() {
// CHECK: %[[C2_SPLAT:.*]] = arith.constant dense<2.000000e+00> : vector<[2]xf64>
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[GET_TILE_ID:.*]] = arm_sme.get_tile_id : i64
// CHECK: %[[TILE:.*]] = arm_sme.cast_tile_to_vector %[[GET_TILE_ID]] : i64 to vector<[2]x[2]xf64>
// CHECK: %[[VSCALE:.*]] = vector.vscale
// CHECK: %[[NUM_TILE_SLICES:.*]] = arith.muli %[[VSCALE]], %[[C2]] : index
// CHECK: scf.for %[[TILE_SLICE_INDEX:.*]] = %[[C0]] to %[[NUM_TILE_SLICES]] step %[[C1]] {
// CHECK:   arm_sme.move_vector_to_tile_slice %[[C2_SPLAT]], %[[TILE]], %[[TILE_SLICE_INDEX]] : vector<[2]xf64> into vector<[2]x[2]xf64>
// CHECK: "prevent.dce"(%[[TILE]]) : (vector<[2]x[2]xf64>) -> ()
func.func @arith_constant_dense_2d_nonzero_f64() {
  %two = arith.constant dense<2.0> : vector<[2]x[2]xf64>
  "prevent.dce"(%two) : (vector<[2]x[2]xf64>) -> ()
  return
}
