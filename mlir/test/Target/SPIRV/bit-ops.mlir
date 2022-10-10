// RUN: mlir-translate -test-spirv-roundtrip -split-input-file %s | FileCheck %s

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.func @bitcount(%arg: i32) -> i32 "None" {
    // CHECK: spirv.BitCount {{%.*}} : i32
    %0 = spirv.BitCount %arg : i32
    spirv.ReturnValue %0 : i32
  }
  spirv.func @bit_field_insert(%base: vector<3xi32>, %insert: vector<3xi32>, %offset: i32, %count: i16) -> vector<3xi32> "None" {
    // CHECK: {{%.*}} = spirv.BitFieldInsert {{%.*}}, {{%.*}}, {{%.*}}, {{%.*}} : vector<3xi32>, i32, i16
    %0 = spirv.BitFieldInsert %base, %insert, %offset, %count : vector<3xi32>, i32, i16
    spirv.ReturnValue %0 : vector<3xi32>
  }
  spirv.func @bit_field_s_extract(%base: vector<3xi32>, %offset: i8, %count: i8) -> vector<3xi32> "None" {
    // CHECK: {{%.*}} = spirv.BitFieldSExtract {{%.*}}, {{%.*}}, {{%.*}} : vector<3xi32>, i8, i8
    %0 = spirv.BitFieldSExtract %base, %offset, %count : vector<3xi32>, i8, i8
    spirv.ReturnValue %0 : vector<3xi32>
  }
  spirv.func @bit_field_u_extract(%base: vector<3xi32>, %offset: i8, %count: i8) -> vector<3xi32> "None" {
    // CHECK: {{%.*}} = spirv.BitFieldUExtract {{%.*}}, {{%.*}}, {{%.*}} : vector<3xi32>, i8, i8
    %0 = spirv.BitFieldUExtract %base, %offset, %count : vector<3xi32>, i8, i8
    spirv.ReturnValue %0 : vector<3xi32>
  }
  spirv.func @bitreverse(%arg: i32) -> i32 "None" {
    // CHECK: spirv.BitReverse {{%.*}} : i32
    %0 = spirv.BitReverse %arg : i32
    spirv.ReturnValue %0 : i32
  }
  spirv.func @not(%arg: i32) -> i32 "None" {
    // CHECK: spirv.Not {{%.*}} : i32
    %0 = spirv.Not %arg : i32
    spirv.ReturnValue %0 : i32
  }
  spirv.func @bitwise_scalar(%arg0 : i32, %arg1 : i32) "None" {
    // CHECK: spirv.BitwiseAnd
    %0 = spirv.BitwiseAnd %arg0, %arg1 : i32
    // CHECK: spirv.BitwiseOr
    %1 = spirv.BitwiseOr %arg0, %arg1 : i32
    // CHECK: spirv.BitwiseXor
    %2 = spirv.BitwiseXor %arg0, %arg1 : i32
    spirv.Return
  }
  spirv.func @shift_left_logical(%arg0: i32, %arg1 : i16) -> i32 "None" {
    // CHECK: {{%.*}} = spirv.ShiftLeftLogical {{%.*}}, {{%.*}} : i32, i16
    %0 = spirv.ShiftLeftLogical %arg0, %arg1: i32, i16
    spirv.ReturnValue %0 : i32
  }
  spirv.func @shift_right_arithmetic(%arg0: vector<4xi32>, %arg1 : vector<4xi8>) -> vector<4xi32> "None" {
    // CHECK: {{%.*}} = spirv.ShiftRightArithmetic {{%.*}}, {{%.*}} : vector<4xi32>, vector<4xi8>
    %0 = spirv.ShiftRightArithmetic %arg0, %arg1: vector<4xi32>, vector<4xi8>
    spirv.ReturnValue %0 : vector<4xi32>
  }
  spirv.func @shift_right_logical(%arg0: vector<2xi32>, %arg1 : vector<2xi8>) -> vector<2xi32> "None" {
    // CHECK: {{%.*}} = spirv.ShiftRightLogical {{%.*}}, {{%.*}} : vector<2xi32>, vector<2xi8>
    %0 = spirv.ShiftRightLogical %arg0, %arg1: vector<2xi32>, vector<2xi8>
    spirv.ReturnValue %0 : vector<2xi32>
  }
}
