// RUN: mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s
// RUN: mlir-opt --split-input-file --verify-diagnostics --canonicalize %s \
// RUN:  | FileCheck %s --check-prefix=CANON

//===----------------------------------------------------------------------===//
// spirv.BitCount
//===----------------------------------------------------------------------===//

func.func @bitcount(%arg: i32) -> i32 {
  // CHECK: spirv.BitCount {{%.*}} : i32
  %0 = spirv.BitCount %arg : i32
  spirv.ReturnValue %0 : i32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.BitFieldInsert
//===----------------------------------------------------------------------===//

func.func @bit_field_insert_vec(%base: vector<3xi32>, %insert: vector<3xi32>, %offset: i32, %count: i16) -> vector<3xi32> {
  // CHECK: {{%.*}} = spirv.BitFieldInsert {{%.*}}, {{%.*}}, {{%.*}}, {{%.*}} : vector<3xi32>, i32, i16
  %0 = spirv.BitFieldInsert %base, %insert, %offset, %count : vector<3xi32>, i32, i16
  spirv.ReturnValue %0 : vector<3xi32>
}

// -----

func.func @bit_field_insert_invalid_insert_type(%base: vector<3xi32>, %insert: vector<2xi32>, %offset: i32, %count: i16) -> vector<3xi32> {
  // TODO: expand post change in verification order. This is currently only
  // verifying that the type verification is failing but not the specific error
  // message. In final state the error should refer to mismatch in base and
  // insert.
  // expected-error @+1 {{type}}
  %0 = "spirv.BitFieldInsert" (%base, %insert, %offset, %count) : (vector<3xi32>, vector<2xi32>, i32, i16) -> vector<3xi32>
  spirv.ReturnValue %0 : vector<3xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.BitFieldSExtract
//===----------------------------------------------------------------------===//

func.func @bit_field_s_extract_vec(%base: vector<3xi32>, %offset: i8, %count: i8) -> vector<3xi32> {
  // CHECK: {{%.*}} = spirv.BitFieldSExtract {{%.*}}, {{%.*}}, {{%.*}} : vector<3xi32>, i8, i8
  %0 = spirv.BitFieldSExtract %base, %offset, %count : vector<3xi32>, i8, i8
  spirv.ReturnValue %0 : vector<3xi32>
}

//===----------------------------------------------------------------------===//
// spirv.BitFieldUExtract
//===----------------------------------------------------------------------===//

func.func @bit_field_u_extract_vec(%base: vector<3xi32>, %offset: i8, %count: i8) -> vector<3xi32> {
  // CHECK: {{%.*}} = spirv.BitFieldUExtract {{%.*}}, {{%.*}}, {{%.*}} : vector<3xi32>, i8, i8
  %0 = spirv.BitFieldUExtract %base, %offset, %count : vector<3xi32>, i8, i8
  spirv.ReturnValue %0 : vector<3xi32>
}

// -----

func.func @bit_field_u_extract_invalid_result_type(%base: vector<3xi32>, %offset: i32, %count: i16) -> vector<4xi32> {
  // expected-error @+1 {{failed to verify that all of {base, result} have same type}}
  %0 = "spirv.BitFieldUExtract" (%base, %offset, %count) : (vector<3xi32>, i32, i16) -> vector<4xi32>
  spirv.ReturnValue %0 : vector<4xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.BitReverse
//===----------------------------------------------------------------------===//

func.func @bitreverse(%arg: i32) -> i32 {
  // CHECK: spirv.BitReverse {{%.*}} : i32
  %0 = spirv.BitReverse %arg : i32
  spirv.ReturnValue %0 : i32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.BitwiseOr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @bitwise_or_scalar
func.func @bitwise_or_scalar(%arg: i32) -> i32 {
  // CHECK: spirv.BitwiseOr
  %0 = spirv.BitwiseOr %arg, %arg : i32
  return %0 : i32
}

// CHECK-LABEL: func @bitwise_or_vector
func.func @bitwise_or_vector(%arg: vector<4xi32>) -> vector<4xi32> {
  // CHECK: spirv.BitwiseOr
  %0 = spirv.BitwiseOr %arg, %arg : vector<4xi32>
  return %0 : vector<4xi32>
}

// CANON-LABEL: func @bitwise_or_zero
// CANON-SAME:    (%[[ARG:.+]]: i32)
func.func @bitwise_or_zero(%arg: i32) -> i32 {
  // CANON: return %[[ARG]]
  %zero = spirv.Constant 0 : i32
  %0 = spirv.BitwiseOr %arg, %zero : i32
  return %0 : i32
}

// CANON-LABEL: func @bitwise_or_zero_vector
// CANON-SAME:    (%[[ARG:.+]]: vector<4xi32>)
func.func @bitwise_or_zero_vector(%arg: vector<4xi32>) -> vector<4xi32> {
  // CANON: return %[[ARG]]
  %zero = spirv.Constant dense<0> : vector<4xi32>
  %0 = spirv.BitwiseOr %arg, %zero : vector<4xi32>
  return %0 : vector<4xi32>
}

// CANON-LABEL: func @bitwise_or_all_ones
func.func @bitwise_or_all_ones(%arg: i8) -> i8 {
  // CANON: %[[CST:.+]] = spirv.Constant -1
  // CANON: return %[[CST]]
  %ones = spirv.Constant 255 : i8
  %0 = spirv.BitwiseOr %arg, %ones : i8
  return %0 : i8
}

// CANON-LABEL: func @bitwise_or_all_ones_vector
func.func @bitwise_or_all_ones_vector(%arg: vector<3xi8>) -> vector<3xi8> {
  // CANON: %[[CST:.+]] = spirv.Constant dense<-1>
  // CANON: return %[[CST]]
  %ones = spirv.Constant dense<255> : vector<3xi8>
  %0 = spirv.BitwiseOr %arg, %ones : vector<3xi8>
  return %0 : vector<3xi8>
}

// -----

func.func @bitwise_or_float(%arg0: f16, %arg1: f16) -> f16 {
  // expected-error @+1 {{operand #0 must be 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2-4294967295}}
  %0 = spirv.BitwiseOr %arg0, %arg1 : f16
  return %0 : f16
}

// -----

//===----------------------------------------------------------------------===//
// spirv.BitwiseXor
//===----------------------------------------------------------------------===//

func.func @bitwise_xor_scalar(%arg: i32) -> i32 {
  // CHECK: spirv.BitwiseXor
  %0 = spirv.BitwiseXor %arg, %arg : i32
  return %0 : i32
}

func.func @bitwise_xor_vector(%arg: vector<4xi32>) -> vector<4xi32> {
  // CHECK: spirv.BitwiseXor
  %0 = spirv.BitwiseXor %arg, %arg : vector<4xi32>
  return %0 : vector<4xi32>
}

// -----

func.func @bitwise_xor_float(%arg0: f16, %arg1: f16) -> f16 {
  // expected-error @+1 {{operand #0 must be 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2-4294967295}}
  %0 = spirv.BitwiseXor %arg0, %arg1 : f16
  return %0 : f16
}

// -----

//===----------------------------------------------------------------------===//
// spirv.BitwiseAnd
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @bitwise_and_scalar
func.func @bitwise_and_scalar(%arg: i32) -> i32 {
  // CHECK: spirv.BitwiseAnd
  %0 = spirv.BitwiseAnd %arg, %arg : i32
  return %0 : i32
}

// CHECK-LABEL: func @bitwise_and_vector
func.func @bitwise_and_vector(%arg: vector<4xi32>) -> vector<4xi32> {
  // CHECK: spirv.BitwiseAnd
  %0 = spirv.BitwiseAnd %arg, %arg : vector<4xi32>
  return %0 : vector<4xi32>
}

// CANON-LABEL: func @bitwise_and_zero
func.func @bitwise_and_zero(%arg: i32) -> i32 {
  // CANON: %[[CST:.+]] = spirv.Constant 0
  // CANON: return %[[CST]]
  %zero = spirv.Constant 0 : i32
  %0 = spirv.BitwiseAnd %arg, %zero : i32
  return %0 : i32
}

// CANON-LABEL: func @bitwise_and_zero_vector
func.func @bitwise_and_zero_vector(%arg: vector<4xi32>) -> vector<4xi32> {
  // CANON: %[[CST:.+]] = spirv.Constant dense<0>
  // CANON: return %[[CST]]
  %zero = spirv.Constant dense<0> : vector<4xi32>
  %0 = spirv.BitwiseAnd %arg, %zero : vector<4xi32>
  return %0 : vector<4xi32>
}

// CANON-LABEL: func @bitwise_and_all_ones
// CANON-SAME:    (%[[ARG:.+]]: i8)
func.func @bitwise_and_all_ones(%arg: i8) -> i8 {
  // CANON: return %[[ARG]]
  %ones = spirv.Constant 255 : i8
  %0 = spirv.BitwiseAnd %arg, %ones : i8
  return %0 : i8
}

// CANON-LABEL: func @bitwise_and_all_ones_vector
// CANON-SAME:    (%[[ARG:.+]]: vector<3xi8>)
func.func @bitwise_and_all_ones_vector(%arg: vector<3xi8>) -> vector<3xi8> {
  // CANON: return %[[ARG]]
  %ones = spirv.Constant dense<255> : vector<3xi8>
  %0 = spirv.BitwiseAnd %arg, %ones : vector<3xi8>
  return %0 : vector<3xi8>
}

// CANON-LABEL: func @bitwise_and_zext_1
// CANON-SAME:    (%[[ARG:.+]]: i8)
func.func @bitwise_and_zext_1(%arg: i8) -> i32 {
  // CANON: %[[ZEXT:.+]] = spirv.UConvert %[[ARG]]
  // CANON: return %[[ZEXT]]
  %zext = spirv.UConvert %arg : i8 to i32
  %ones = spirv.Constant 255 : i32
  %0 = spirv.BitwiseAnd %zext, %ones : i32
  return %0 : i32
}

// CANON-LABEL: func @bitwise_and_zext_2
// CANON-SAME:    (%[[ARG:.+]]: i8)
func.func @bitwise_and_zext_2(%arg: i8) -> i32 {
  // CANON: %[[ZEXT:.+]] = spirv.UConvert %[[ARG]]
  // CANON: return %[[ZEXT]]
  %zext = spirv.UConvert %arg : i8 to i32
  %ones = spirv.Constant 0x12345ff : i32
  %0 = spirv.BitwiseAnd %zext, %ones : i32
  return %0 : i32
}

// CANON-LABEL: func @bitwise_and_zext_3
// CANON-SAME:    (%[[ARG:.+]]: i8)
func.func @bitwise_and_zext_3(%arg: i8) -> i32 {
  // CANON: %[[ZEXT:.+]] = spirv.UConvert %[[ARG]]
  // CANON: %[[AND:.+]]  = spirv.BitwiseAnd %[[ZEXT]]
  // CANON: return %[[AND]]
  %zext = spirv.UConvert %arg : i8 to i32
  %ones = spirv.Constant 254 : i32
  %0 = spirv.BitwiseAnd %zext, %ones : i32
  return %0 : i32
}

// CANON-LABEL: func @bitwise_and_zext_vector
// CANON-SAME:    (%[[ARG:.+]]: vector<2xi8>)
func.func @bitwise_and_zext_vector(%arg: vector<2xi8>) -> vector<2xi32> {
  // CANON: %[[ZEXT:.+]] = spirv.UConvert %[[ARG]]
  // CANON: return %[[ZEXT]]
  %zext = spirv.UConvert %arg : vector<2xi8> to vector<2xi32>
  %ones = spirv.Constant dense<255> : vector<2xi32>
  %0 = spirv.BitwiseAnd %zext, %ones : vector<2xi32>
  return %0 : vector<2xi32>
}

// -----

func.func @bitwise_and_float(%arg0: f16, %arg1: f16) -> f16 {
  // expected-error @+1 {{operand #0 must be 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2-4294967295}}
  %0 = spirv.BitwiseAnd %arg0, %arg1 : f16
  return %0 : f16
}

// -----

//===----------------------------------------------------------------------===//
// spirv.Not
//===----------------------------------------------------------------------===//

func.func @not(%arg: i32) -> i32 {
  // CHECK: spirv.Not {{%.*}} : i32
  %0 = spirv.Not %arg : i32
  spirv.ReturnValue %0 : i32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.ShiftLeftLogical
//===----------------------------------------------------------------------===//

func.func @shift_left_logical(%arg0: i32, %arg1 : i16) -> i32 {
  // CHECK: {{%.*}} = spirv.ShiftLeftLogical {{%.*}}, {{%.*}} : i32, i16
  %0 = spirv.ShiftLeftLogical %arg0, %arg1: i32, i16
  spirv.ReturnValue %0 : i32
}

// -----

func.func @shift_left_logical_invalid_result_type(%arg0: i32, %arg1 : i16) -> i16 {
  // expected-error @+1 {{op failed to verify that all of {operand1, result} have same type}}
  %0 = "spirv.ShiftLeftLogical" (%arg0, %arg1) : (i32, i16) -> (i16)
  spirv.ReturnValue %0 : i16
}

// -----

//===----------------------------------------------------------------------===//
// spirv.ShiftRightArithmetic
//===----------------------------------------------------------------------===//

func.func @shift_right_arithmetic(%arg0: vector<4xi32>, %arg1 : vector<4xi8>) -> vector<4xi32> {
  // CHECK: {{%.*}} = spirv.ShiftRightArithmetic {{%.*}}, {{%.*}} : vector<4xi32>, vector<4xi8>
  %0 = spirv.ShiftRightArithmetic %arg0, %arg1: vector<4xi32>, vector<4xi8>
  spirv.ReturnValue %0 : vector<4xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.ShiftRightLogical
//===----------------------------------------------------------------------===//

func.func @shift_right_logical(%arg0: vector<2xi32>, %arg1 : vector<2xi8>) -> vector<2xi32> {
  // CHECK: {{%.*}} = spirv.ShiftRightLogical {{%.*}}, {{%.*}} : vector<2xi32>, vector<2xi8>
  %0 = spirv.ShiftRightLogical %arg0, %arg1: vector<2xi32>, vector<2xi8>
  spirv.ReturnValue %0 : vector<2xi32>
}
