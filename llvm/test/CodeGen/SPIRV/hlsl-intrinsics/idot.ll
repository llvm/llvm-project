; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Make sure dxil operation function calls for dot are generated for int/uint vectors.

; CHECK-DAG: %[[#int_16:]] = OpTypeInt 16
; CHECK-DAG: %[[#vec2_int_16:]] = OpTypeVector %[[#int_16]] 2
; CHECK-DAG: %[[#vec3_int_16:]] = OpTypeVector %[[#int_16]] 3
; CHECK-DAG: %[[#int_32:]] = OpTypeInt 32
; CHECK-DAG: %[[#vec4_int_32:]] = OpTypeVector %[[#int_32]] 4
; CHECK-DAG: %[[#int_64:]] = OpTypeInt 64
; CHECK-DAG: %[[#vec2_int_64:]] = OpTypeVector %[[#int_64]] 2

define noundef i16 @dot_int16_t2(<2 x i16> noundef %a, <2 x i16> noundef %b) {
entry:
; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec2_int_16]]
; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#vec2_int_16]]
; CHECK: %[[#mul_vec:]] = OpIMul %[[#vec2_int_16]] %[[#arg0]] %[[#arg1]]
; CHECK: %[[#elt0:]] = OpCompositeExtract %[[#int_16]] %[[#mul_vec]] 0
; CHECK: %[[#elt1:]] = OpCompositeExtract %[[#int_16]] %[[#mul_vec]] 1
; CHECK: %[[#sum:]] = OpIAdd %[[#int_16]] %[[#elt0]] %[[#elt1]]
  %dot = call i16 @llvm.spv.sdot.v3i16(<2 x i16> %a, <2 x i16> %b)
  ret i16 %dot
}

define noundef i32 @dot_int4(<4 x i32> noundef %a, <4 x i32> noundef %b) {
entry:
; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_int_32]]
; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#vec4_int_32]]
; CHECK: %[[#mul_vec:]] = OpIMul %[[#vec4_int_32]] %[[#arg0]] %[[#arg1]]
; CHECK: %[[#elt0:]] = OpCompositeExtract %[[#int_32]] %[[#mul_vec]] 0
; CHECK: %[[#elt1:]] = OpCompositeExtract %[[#int_32]] %[[#mul_vec]] 1
; CHECK: %[[#sum0:]] = OpIAdd %[[#int_32]] %[[#elt0]] %[[#elt1]]
; CHECK: %[[#elt2:]] = OpCompositeExtract %[[#int_32]] %[[#mul_vec]] 2
; CHECK: %[[#sum1:]] = OpIAdd %[[#int_32]] %[[#sum0]] %[[#elt2]]
; CHECK: %[[#elt3:]] = OpCompositeExtract %[[#int_32]] %[[#mul_vec]] 3
; CHECK: %[[#sum2:]] = OpIAdd %[[#int_32]] %[[#sum1]] %[[#elt3]]
  %dot = call i32 @llvm.spv.sdot.v4i32(<4 x i32> %a, <4 x i32> %b)
  ret i32 %dot
}

define noundef i16 @dot_uint16_t3(<3 x i16> noundef %a, <3 x i16> noundef %b) {
entry:
; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec3_int_16]]
; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#vec3_int_16]]
; CHECK: %[[#mul_vec:]] = OpIMul %[[#vec3_int_16]] %[[#arg0]] %[[#arg1]]
; CHECK: %[[#elt0:]] = OpCompositeExtract %[[#int_16]] %[[#mul_vec]] 0
; CHECK: %[[#elt1:]] = OpCompositeExtract %[[#int_16]] %[[#mul_vec]] 1
; CHECK: %[[#sum0:]] = OpIAdd %[[#int_16]] %[[#elt0]] %[[#elt1]]
; CHECK: %[[#elt2:]] = OpCompositeExtract %[[#int_16]] %[[#mul_vec]] 2
; CHECK: %[[#sum1:]] = OpIAdd %[[#int_16]] %[[#sum0]] %[[#elt2]]
  %dot = call i16 @llvm.spv.udot.v3i16(<3 x i16> %a, <3 x i16> %b)
  ret i16 %dot
}

define noundef i32 @dot_uint4(<4 x i32> noundef %a, <4 x i32> noundef %b) {
entry:
; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_int_32]]
; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#vec4_int_32]]
; CHECK: %[[#mul_vec:]] = OpIMul %[[#vec4_int_32]] %[[#arg0]] %[[#arg1]]
; CHECK: %[[#elt0:]] = OpCompositeExtract %[[#int_32]] %[[#mul_vec]] 0
; CHECK: %[[#elt1:]] = OpCompositeExtract %[[#int_32]] %[[#mul_vec]] 1
; CHECK: %[[#sum0:]] = OpIAdd %[[#int_32]] %[[#elt0]] %[[#elt1]]
; CHECK: %[[#elt2:]] = OpCompositeExtract %[[#int_32]] %[[#mul_vec]] 2
; CHECK: %[[#sum1:]] = OpIAdd %[[#int_32]] %[[#sum0]] %[[#elt2]]
; CHECK: %[[#elt3:]] = OpCompositeExtract %[[#int_32]] %[[#mul_vec]] 3
; CHECK: %[[#sum2:]] = OpIAdd %[[#int_32]] %[[#sum1]] %[[#elt3]]
  %dot = call i32 @llvm.spv.udot.v4i32(<4 x i32> %a, <4 x i32> %b)
  ret i32 %dot
}

define noundef i64 @dot_uint64_t4(<2 x i64> noundef %a, <2 x i64> noundef %b) {
entry:
; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec2_int_64]]
; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#vec2_int_64]]
; CHECK: %[[#mul_vec:]] = OpIMul %[[#vec2_int_64]] %[[#arg0]] %[[#arg1]]
; CHECK: %[[#elt0:]] = OpCompositeExtract %[[#int_64]] %[[#mul_vec]] 0
; CHECK: %[[#elt1:]] = OpCompositeExtract %[[#int_64]] %[[#mul_vec]] 1
; CHECK: %[[#sum0:]] = OpIAdd %[[#int_64]] %[[#elt0]] %[[#elt1]]
  %dot = call i64 @llvm.spv.udot.v2i64(<2 x i64> %a, <2 x i64> %b)
  ret i64 %dot
}

declare i16 @llvm.spv.sdot.v2i16(<2 x i16>, <2 x i16>)
declare i32 @llvm.spv.sdot.v4i32(<4 x i32>, <4 x i32>)
declare i16 @llvm.spv.udot.v3i32(<3 x i16>, <3 x i16>)
declare i32 @llvm.spv.udot.v4i32(<4 x i32>, <4 x i32>)
declare i64 @llvm.spv.udot.v2i64(<2 x i64>, <2 x i64>)
