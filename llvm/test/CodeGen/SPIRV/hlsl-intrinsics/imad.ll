; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#int_16:]] = OpTypeInt 16 0
; CHECK-DAG: %[[#vec2_16:]] = OpTypeVector %[[#int_16]] 2
; CHECK-DAG: %[[#vec3_16:]] = OpTypeVector %[[#int_16]] 3
; CHECK-DAG: %[[#vec4_16:]] = OpTypeVector %[[#int_16]] 4
; CHECK-DAG: %[[#int_32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#vec2_32:]] = OpTypeVector %[[#int_32]] 2
; CHECK-DAG: %[[#vec3_32:]] = OpTypeVector %[[#int_32]] 3
; CHECK-DAG: %[[#vec4_32:]] = OpTypeVector %[[#int_32]] 4
; CHECK-DAG: %[[#int_64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#vec2_64:]] = OpTypeVector %[[#int_64]] 2
; CHECK-DAG: %[[#vec3_64:]] = OpTypeVector %[[#int_64]] 3
; CHECK-DAG: %[[#vec4_64:]] = OpTypeVector %[[#int_64]] 4

define spir_func noundef i16 @test_mad_uint16_t(i16 noundef %p0, i16 noundef %p1, i16 noundef %p2) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#mul:]] = OpIMul %[[#int_16]] %[[#arg0]] %[[#arg1]]
  ; CHECK: OpIAdd %[[#int_16]] %[[#mul]] %[[#arg2]]
  %3 = mul nuw i16 %p0, %p1
  %4 = add nuw i16 %3, %p2
  ret i16 %4
}

define spir_func noundef <2 x i16> @test_mad_uint16_t2(<2 x i16> noundef %p0, <2 x i16> noundef %p1, <2 x i16> noundef %p2) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#mul:]] = OpIMul %[[#vec2_16]] %[[#arg0]] %[[#arg1]]
  ; CHECK: OpIAdd %[[#vec2_16]] %[[#mul]] %[[#arg2]]
  %3 = mul nuw <2 x i16> %p0, %p1
  %4 = add nuw <2 x i16> %3, %p2
  ret <2 x i16> %4
}

define spir_func noundef <3 x i16> @test_mad_uint16_t3(<3 x i16> noundef %p0, <3 x i16> noundef %p1, <3 x i16> noundef %p2) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#mul:]] = OpIMul %[[#vec3_16]] %[[#arg0]] %[[#arg1]]
  ; CHECK: OpIAdd %[[#vec3_16]] %[[#mul]] %[[#arg2]]
  %3 = mul nuw <3 x i16> %p0, %p1
  %4 = add nuw <3 x i16> %3, %p2
  ret <3 x i16> %4
}

define spir_func noundef <4 x i16> @test_mad_uint16_t4(<4 x i16> noundef %p0, <4 x i16> noundef %p1, <4 x i16> noundef %p2) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#mul:]] = OpIMul %[[#vec4_16]] %[[#arg0]] %[[#arg1]]
  ; CHECK: OpIAdd %[[#vec4_16]] %[[#mul]] %[[#arg2]]
  %3 = mul nuw <4 x i16> %p0, %p1
  %4 = add nuw <4 x i16> %3, %p2
  ret <4 x i16> %4
}

define spir_func noundef i16 @test_mad_int16_t(i16 noundef %p0, i16 noundef %p1, i16 noundef %p2) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#mul:]] = OpIMul %[[#int_16]] %[[#arg0]] %[[#arg1]]
  ; CHECK: OpIAdd %[[#int_16]] %[[#mul]] %[[#arg2]]
  %3 = mul nsw i16 %p0, %p1
  %4 = add nsw i16 %3, %p2
  ret i16 %4
}

define spir_func noundef <2 x i16> @test_mad_int16_t2(<2 x i16> noundef %p0, <2 x i16> noundef %p1, <2 x i16> noundef %p2) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#mul:]] = OpIMul %[[#vec2_16]] %[[#arg0]] %[[#arg1]]
  ; CHECK: OpIAdd %[[#vec2_16]] %[[#mul]] %[[#arg2]]
  %3 = mul nsw <2 x i16> %p0, %p1
  %4 = add nsw <2 x i16> %3, %p2
  ret <2 x i16> %4
}

define spir_func noundef <3 x i16> @test_mad_int16_t3(<3 x i16> noundef %p0, <3 x i16> noundef %p1, <3 x i16> noundef %p2) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#mul:]] = OpIMul %[[#vec3_16]] %[[#arg0]] %[[#arg1]]
  ; CHECK: OpIAdd %[[#vec3_16]] %[[#mul]] %[[#arg2]]
  %3 = mul nsw <3 x i16> %p0, %p1
  %4 = add nsw <3 x i16> %3, %p2
  ret <3 x i16> %4
}

define spir_func noundef <4 x i16> @test_mad_int16_t4(<4 x i16> noundef %p0, <4 x i16> noundef %p1, <4 x i16> noundef %p2) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#mul:]] = OpIMul %[[#vec4_16]] %[[#arg0]] %[[#arg1]]
  ; CHECK: OpIAdd %[[#vec4_16]] %[[#mul]] %[[#arg2]]
  %3 = mul nsw <4 x i16> %p0, %p1
  %4 = add nsw <4 x i16> %3, %p2
  ret <4 x i16> %4
}
define spir_func noundef i32 @test_mad_int(i32 noundef %p0, i32 noundef %p1, i32 noundef %p2) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#mul:]] = OpIMul %[[#int_32]] %[[#arg0]] %[[#arg1]]
  ; CHECK: OpIAdd %[[#int_32]] %[[#mul]] %[[#arg2]]
  %3 = mul nsw i32 %p0, %p1
  %4 = add nsw i32 %3, %p2
  ret i32 %4
}

define spir_func noundef <2 x i32> @test_mad_int2(<2 x i32> noundef %p0, <2 x i32> noundef %p1, <2 x i32> noundef %p2) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#mul:]] = OpIMul %[[#vec2_32]] %[[#arg0]] %[[#arg1]]
  ; CHECK: OpIAdd %[[#vec2_32]] %[[#mul]] %[[#arg2]]
  %3 = mul nsw <2 x i32> %p0, %p1
  %4 = add nsw <2 x i32> %3, %p2
  ret <2 x i32> %4
}

define spir_func noundef <3 x i32> @test_mad_int3(<3 x i32> noundef %p0, <3 x i32> noundef %p1, <3 x i32> noundef %p2) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#mul:]] = OpIMul %[[#vec3_32]] %[[#arg0]] %[[#arg1]]
  ; CHECK: OpIAdd %[[#vec3_32]] %[[#mul]] %[[#arg2]]
  %3 = mul nsw <3 x i32> %p0, %p1
  %4 = add nsw <3 x i32> %3, %p2
  ret <3 x i32> %4
}

define spir_func noundef <4 x i32> @test_mad_int4(<4 x i32> noundef %p0, <4 x i32> noundef %p1, <4 x i32> noundef %p2) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#mul:]] = OpIMul %[[#vec4_32]] %[[#arg0]] %[[#arg1]]
  ; CHECK: OpIAdd %[[#vec4_32]] %[[#mul]] %[[#arg2]]
  %3 = mul nsw <4 x i32> %p0, %p1
  %4 = add nsw <4 x i32> %3, %p2
  ret <4 x i32> %4
}

define spir_func noundef i64 @test_mad_int64_t(i64 noundef %p0, i64 noundef %p1, i64 noundef %p2) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#mul:]] = OpIMul %[[#int_64]] %[[#arg0]] %[[#arg1]]
  ; CHECK: OpIAdd %[[#int_64]] %[[#mul]] %[[#arg2]]
  %3 = mul nsw i64 %p0, %p1
  %4 = add nsw i64 %3, %p2
  ret i64 %4
}

define spir_func noundef <2 x i64> @test_mad_int64_t2(<2 x i64> noundef %p0, <2 x i64> noundef %p1, <2 x i64> noundef %p2) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#mul:]] = OpIMul %[[#vec2_64]] %[[#arg0]] %[[#arg1]]
  ; CHECK: OpIAdd %[[#vec2_64]] %[[#mul]] %[[#arg2]]
  %3 = mul nsw <2 x i64> %p0, %p1
  %4 = add nsw <2 x i64> %3, %p2
  ret <2 x i64> %4
}

define spir_func noundef <3 x i64> @test_mad_int64_t3(<3 x i64> noundef %p0, <3 x i64> noundef %p1, <3 x i64> noundef %p2) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#mul:]] = OpIMul %[[#vec3_64]] %[[#arg0]] %[[#arg1]]
  ; CHECK: OpIAdd %[[#vec3_64]] %[[#mul]] %[[#arg2]]
  %3 = mul nsw <3 x i64> %p0, %p1
  %4 = add nsw <3 x i64> %3, %p2
  ret <3 x i64> %4
}

define spir_func noundef <4 x i64> @test_mad_int64_t4(<4 x i64> noundef %p0, <4 x i64> noundef %p1, <4 x i64> noundef %p2) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#mul:]] = OpIMul %[[#vec4_64]] %[[#arg0]] %[[#arg1]]
  ; CHECK: OpIAdd %[[#vec4_64]] %[[#mul]] %[[#arg2]]
  %3 = mul nsw <4 x i64> %p0, %p1
  %4 = add nsw <4 x i64> %3, %p2
  ret <4 x i64> %4
}

define spir_func noundef i32 @test_mad_uint(i32 noundef %p0, i32 noundef %p1, i32 noundef %p2) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#mul:]] = OpIMul %[[#int_32]] %[[#arg0]] %[[#arg1]]
  ; CHECK: OpIAdd %[[#int_32]] %[[#mul]] %[[#arg2]]
  %3 = mul nuw i32 %p0, %p1
  %4 = add nuw i32 %3, %p2
  ret i32 %4
}

define spir_func noundef <2 x i32> @test_mad_uint2(<2 x i32> noundef %p0, <2 x i32> noundef %p1, <2 x i32> noundef %p2) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#mul:]] = OpIMul %[[#vec2_32]] %[[#arg0]] %[[#arg1]]
  ; CHECK: OpIAdd %[[#vec2_32]] %[[#mul]] %[[#arg2]]
  %3 = mul nuw <2 x i32> %p0, %p1
  %4 = add nuw <2 x i32> %3, %p2
  ret <2 x i32> %4
}

define spir_func noundef <3 x i32> @test_mad_uint3(<3 x i32> noundef %p0, <3 x i32> noundef %p1, <3 x i32> noundef %p2) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#mul:]] = OpIMul %[[#vec3_32]] %[[#arg0]] %[[#arg1]]
  ; CHECK: OpIAdd %[[#vec3_32]] %[[#mul]] %[[#arg2]]
  %3 = mul nuw <3 x i32> %p0, %p1
  %4 = add nuw <3 x i32> %3, %p2
  ret <3 x i32> %4
}

define spir_func noundef <4 x i32> @test_mad_uint4(<4 x i32> noundef %p0, <4 x i32> noundef %p1, <4 x i32> noundef %p2) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#mul:]] = OpIMul %[[#vec4_32]] %[[#arg0]] %[[#arg1]]
  ; CHECK: OpIAdd %[[#vec4_32]] %[[#mul]] %[[#arg2]]
  %3 = mul nuw <4 x i32> %p0, %p1
  %4 = add nuw <4 x i32> %3, %p2
  ret <4 x i32> %4
}

define spir_func noundef i64 @test_mad_uint64_t(i64 noundef %p0, i64 noundef %p1, i64 noundef %p2) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#mul:]] = OpIMul %[[#int_64]] %[[#arg0]] %[[#arg1]]
  ; CHECK: OpIAdd %[[#int_64]] %[[#mul]] %[[#arg2]]
  %3 = mul nuw i64 %p0, %p1
  %4 = add nuw i64 %3, %p2
  ret i64 %4
}

define spir_func noundef <2 x i64> @test_mad_uint64_t2(<2 x i64> noundef %p0, <2 x i64> noundef %p1, <2 x i64> noundef %p2) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#mul:]] = OpIMul %[[#vec2_64]] %[[#arg0]] %[[#arg1]]
  ; CHECK: OpIAdd %[[#vec2_64]] %[[#mul]] %[[#arg2]]
  %3 = mul nuw <2 x i64> %p0, %p1
  %4 = add nuw <2 x i64> %3, %p2
  ret <2 x i64> %4
}

define spir_func noundef <3 x i64> @test_mad_uint64_t3(<3 x i64> noundef %p0, <3 x i64> noundef %p1, <3 x i64> noundef %p2) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#mul:]] = OpIMul %[[#vec3_64]] %[[#arg0]] %[[#arg1]]
  ; CHECK: OpIAdd %[[#vec3_64]] %[[#mul]] %[[#arg2]]
  %3 = mul nuw <3 x i64> %p0, %p1
  %4 = add nuw <3 x i64> %3, %p2
  ret <3 x i64> %4
}

define spir_func noundef <4 x i64> @test_mad_uint64_t4(<4 x i64> noundef %p0, <4 x i64> noundef %p1, <4 x i64> noundef %p2) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#mul:]] = OpIMul %[[#vec4_64]] %[[#arg0]] %[[#arg1]]
  ; CHECK: OpIAdd %[[#vec4_64]] %[[#mul]] %[[#arg2]]
  %3 = mul nuw <4 x i64> %p0, %p1
  %4 = add nuw <4 x i64> %3, %p2
  ret <4 x i64> %4
}
