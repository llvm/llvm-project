; RUN: llc -global-isel -march=amdgcn -mcpu=gfx700 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX7 %s
; RUN: llc -global-isel -march=amdgcn -mcpu=gfx801 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX8 %s
; RUN: llc -global-isel -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX9 %s

define amdgpu_ps i16 @s_mul_i16(i16 inreg %num, i16 inreg %den) {
  %result = mul i16 %num, %den
  ret i16 %result
}

define i16 @v_mul_i16(i16 %num, i16 %den) {
  %result = mul i16 %num, %den
  ret i16 %result
}

define amdgpu_ps zeroext i16 @s_mul_i16_zeroext(i16 inreg zeroext %num, i16 inreg zeroext %den) {
  %result = mul i16 %num, %den
  ret i16 %result
}

define zeroext i16 @v_mul_i16_zeroext(i16 zeroext %num, i16 zeroext %den) {
  %result = mul i16 %num, %den
  ret i16 %result
}

define amdgpu_ps signext i16 @s_mul_i16_signext(i16 inreg signext %num, i16 inreg signext %den) {
  %result = mul i16 %num, %den
  ret i16 %result
}

define signext i16 @v_mul_i16_signext(i16 signext %num, i16 signext %den) {
  %result = mul i16 %num, %den
  ret i16 %result
}

define amdgpu_ps i32 @s_mul_i32(i32 inreg %num, i32 inreg %den) {
  %result = mul i32 %num, %den
  ret i32 %result
}

define i32 @v_mul_i32(i32 %num, i32 %den) {
  %result = mul i32 %num, %den
  ret i32 %result
}

define amdgpu_ps <2 x i32> @s_mul_v2i32(<2 x i32> inreg %num, <2 x i32> inreg %den) {
  %result = mul <2 x i32> %num, %den
  ret <2 x i32> %result
}

define <2 x i32> @v_mul_v2i32(<2 x i32> %num, <2 x i32> %den) {
  %result = mul <2 x i32> %num, %den
  ret <2 x i32> %result
}

define amdgpu_ps i64 @s_mul_i64(i64 inreg %num, i64 inreg %den) {
  %result = mul i64 %num, %den
  ret i64 %result
}

define i64 @v_mul_i64(i64 %num, i64 %den) {
  %result = mul i64 %num, %den
  ret i64 %result
}

define amdgpu_ps <3 x i32> @s_mul_i96(i96 inreg %num, i96 inreg %den) {
  %result = mul i96 %num, %den
  %cast = bitcast i96 %result to <3 x i32>
  ret <3 x i32> %cast
}

define i96 @v_mul_i96(i96 %num, i96 %den) {
  %result = mul i96 %num, %den
  ret i96 %result
}

define amdgpu_ps <4 x i32> @s_mul_i128(i128 inreg %num, i128 inreg %den) {
  %result = mul i128 %num, %den
  %cast = bitcast i128 %result to <4 x i32>
  ret <4 x i32> %cast
}

define i128 @v_mul_i128(i128 %num, i128 %den) {
  %result = mul i128 %num, %den
  ret i128 %result
}

define amdgpu_ps <8 x i32> @s_mul_i256(i256 inreg %num, i256 inreg %den) {
  %result = mul i256 %num, %den
  %cast = bitcast i256 %result to <8 x i32>
  ret <8 x i32> %cast
}

define i256 @v_mul_i256(i256 %num, i256 %den) {
  %result = mul i256 %num, %den
  ret i256 %result
}
