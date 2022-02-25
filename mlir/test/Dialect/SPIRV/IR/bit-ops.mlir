// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.BitCount
//===----------------------------------------------------------------------===//

func @bitcount(%arg: i32) -> i32 {
  // CHECK: spv.BitCount {{%.*}} : i32
  %0 = spv.BitCount %arg : i32
  spv.ReturnValue %0 : i32
}

// -----

//===----------------------------------------------------------------------===//
// spv.BitFieldInsert
//===----------------------------------------------------------------------===//

func @bit_field_insert_vec(%base: vector<3xi32>, %insert: vector<3xi32>, %offset: i32, %count: i16) -> vector<3xi32> {
  // CHECK: {{%.*}} = spv.BitFieldInsert {{%.*}}, {{%.*}}, {{%.*}}, {{%.*}} : vector<3xi32>, i32, i16
  %0 = spv.BitFieldInsert %base, %insert, %offset, %count : vector<3xi32>, i32, i16
  spv.ReturnValue %0 : vector<3xi32>
}

// -----

func @bit_field_insert_invalid_insert_type(%base: vector<3xi32>, %insert: vector<2xi32>, %offset: i32, %count: i16) -> vector<3xi32> {
  // TODO: expand post change in verification order. This is currently only
  // verifying that the type verification is failing but not the specific error
  // message. In final state the error should refer to mismatch in base and
  // insert.
  // expected-error @+1 {{type}}
  %0 = "spv.BitFieldInsert" (%base, %insert, %offset, %count) : (vector<3xi32>, vector<2xi32>, i32, i16) -> vector<3xi32>
  spv.ReturnValue %0 : vector<3xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spv.BitFieldSExtract
//===----------------------------------------------------------------------===//

func @bit_field_s_extract_vec(%base: vector<3xi32>, %offset: i8, %count: i8) -> vector<3xi32> {
  // CHECK: {{%.*}} = spv.BitFieldSExtract {{%.*}}, {{%.*}}, {{%.*}} : vector<3xi32>, i8, i8
  %0 = spv.BitFieldSExtract %base, %offset, %count : vector<3xi32>, i8, i8
  spv.ReturnValue %0 : vector<3xi32>
}

//===----------------------------------------------------------------------===//
// spv.BitFieldUExtract
//===----------------------------------------------------------------------===//

func @bit_field_u_extract_vec(%base: vector<3xi32>, %offset: i8, %count: i8) -> vector<3xi32> {
  // CHECK: {{%.*}} = spv.BitFieldUExtract {{%.*}}, {{%.*}}, {{%.*}} : vector<3xi32>, i8, i8
  %0 = spv.BitFieldUExtract %base, %offset, %count : vector<3xi32>, i8, i8
  spv.ReturnValue %0 : vector<3xi32>
}

// -----

func @bit_field_u_extract_invalid_result_type(%base: vector<3xi32>, %offset: i32, %count: i16) -> vector<4xi32> {
  // expected-error @+1 {{failed to verify that all of {base, result} have same type}}
  %0 = "spv.BitFieldUExtract" (%base, %offset, %count) : (vector<3xi32>, i32, i16) -> vector<4xi32>
  spv.ReturnValue %0 : vector<4xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spv.BitReverse
//===----------------------------------------------------------------------===//

func @bitreverse(%arg: i32) -> i32 {
  // CHECK: spv.BitReverse {{%.*}} : i32
  %0 = spv.BitReverse %arg : i32
  spv.ReturnValue %0 : i32
}

// -----

//===----------------------------------------------------------------------===//
// spv.BitwiseOr
//===----------------------------------------------------------------------===//

func @bitwise_or_scalar(%arg: i32) -> i32 {
  // CHECK: spv.BitwiseOr
  %0 = spv.BitwiseOr %arg, %arg : i32
  return %0 : i32
}

func @bitwise_or_vector(%arg: vector<4xi32>) -> vector<4xi32> {
  // CHECK: spv.BitwiseOr
  %0 = spv.BitwiseOr %arg, %arg : vector<4xi32>
  return %0 : vector<4xi32>
}

// -----

func @bitwise_or_float(%arg0: f16, %arg1: f16) -> f16 {
  // expected-error @+1 {{operand #0 must be 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4}}
  %0 = spv.BitwiseOr %arg0, %arg1 : f16
  return %0 : f16
}

// -----

//===----------------------------------------------------------------------===//
// spv.BitwiseXor
//===----------------------------------------------------------------------===//

func @bitwise_xor_scalar(%arg: i32) -> i32 {
  // CHECK: spv.BitwiseXor
  %0 = spv.BitwiseXor %arg, %arg : i32
  return %0 : i32
}

func @bitwise_xor_vector(%arg: vector<4xi32>) -> vector<4xi32> {
  // CHECK: spv.BitwiseXor
  %0 = spv.BitwiseXor %arg, %arg : vector<4xi32>
  return %0 : vector<4xi32>
}

// -----

func @bitwise_xor_float(%arg0: f16, %arg1: f16) -> f16 {
  // expected-error @+1 {{operand #0 must be 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4}}
  %0 = spv.BitwiseXor %arg0, %arg1 : f16
  return %0 : f16
}

// -----

//===----------------------------------------------------------------------===//
// spv.BitwiseAnd
//===----------------------------------------------------------------------===//

func @bitwise_and_scalar(%arg: i32) -> i32 {
  // CHECK: spv.BitwiseAnd
  %0 = spv.BitwiseAnd %arg, %arg : i32
  return %0 : i32
}

func @bitwise_and_vector(%arg: vector<4xi32>) -> vector<4xi32> {
  // CHECK: spv.BitwiseAnd
  %0 = spv.BitwiseAnd %arg, %arg : vector<4xi32>
  return %0 : vector<4xi32>
}

// -----

func @bitwise_and_float(%arg0: f16, %arg1: f16) -> f16 {
  // expected-error @+1 {{operand #0 must be 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4}}
  %0 = spv.BitwiseAnd %arg0, %arg1 : f16
  return %0 : f16
}

// -----

//===----------------------------------------------------------------------===//
// spv.Not
//===----------------------------------------------------------------------===//

func @not(%arg: i32) -> i32 {
  // CHECK: spv.Not {{%.*}} : i32
  %0 = spv.Not %arg : i32
  spv.ReturnValue %0 : i32
}

// -----

//===----------------------------------------------------------------------===//
// spv.ShiftLeftLogical
//===----------------------------------------------------------------------===//

func @shift_left_logical(%arg0: i32, %arg1 : i16) -> i32 {
  // CHECK: {{%.*}} = spv.ShiftLeftLogical {{%.*}}, {{%.*}} : i32, i16
  %0 = spv.ShiftLeftLogical %arg0, %arg1: i32, i16
  spv.ReturnValue %0 : i32
}

// -----

func @shift_left_logical_invalid_result_type(%arg0: i32, %arg1 : i16) -> i16 {
  // expected-error @+1 {{op failed to verify that all of {operand1, result} have same type}}
  %0 = "spv.ShiftLeftLogical" (%arg0, %arg1) : (i32, i16) -> (i16)
  spv.ReturnValue %0 : i16
}

// -----

//===----------------------------------------------------------------------===//
// spv.ShiftRightArithmetic
//===----------------------------------------------------------------------===//

func @shift_right_arithmetic(%arg0: vector<4xi32>, %arg1 : vector<4xi8>) -> vector<4xi32> {
  // CHECK: {{%.*}} = spv.ShiftRightArithmetic {{%.*}}, {{%.*}} : vector<4xi32>, vector<4xi8>
  %0 = spv.ShiftRightArithmetic %arg0, %arg1: vector<4xi32>, vector<4xi8>
  spv.ReturnValue %0 : vector<4xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spv.ShiftRightLogical
//===----------------------------------------------------------------------===//

func @shift_right_logical(%arg0: vector<2xi32>, %arg1 : vector<2xi8>) -> vector<2xi32> {
  // CHECK: {{%.*}} = spv.ShiftRightLogical {{%.*}}, {{%.*}} : vector<2xi32>, vector<2xi8>
  %0 = spv.ShiftRightLogical %arg0, %arg1: vector<2xi32>, vector<2xi8>
  spv.ReturnValue %0 : vector<2xi32>
}
