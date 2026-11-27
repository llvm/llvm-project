; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val %}

; Test if llvm.exp10 is lowered with the result correctly reused by the
; original llvm.exp10 user.


; CHECK-DAG: %[[#ExtInstId:]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: %[[#F16Ty:]] = OpTypeFloat 16
; CHECK-DAG: %[[#Vec4_16Ty:]] = OpTypeVector %[[#F16Ty]] 4
; CHECK-DAG: %[[#F32Ty:]] = OpTypeFloat 32
; CHECK-DAG: %[[#Vec4_32Ty:]] = OpTypeVector %[[#F32Ty]] 4
; CHECK-DAG: %[[#Constant_s16:]] = OpConstant %[[#F16Ty]] 17061
; CHECK-DAG: %[[#Constant_s32:]] = OpConstant %[[#F32Ty:]] 3.3219280242919922

; CHECK-LABEL: Begin function test_exp10_f32_scalar
; CHECK: %[[#s32_arg:]] = OpFunctionParameter %[[#F32Ty]]
; CHECK: %[[#s32_mul:]] = OpFMul %[[#F32Ty]] %[[#s32_arg]] %[[#Constant_s32]]
; CHECK: %[[#s32_ret:]] = OpExtInst %[[#F32Ty]] %[[#ExtInstId]] Exp2 %[[#s32_mul]]
; CHECK: OpReturnValue %[[#s32_ret]]
; CHECK-LABEL: OpFunctionEnd
define float @test_exp10_f32_scalar(float %x) {
    %res = call float @llvm.exp10.f32(float %x)
    ret float %res
}

; CHECK-LABEL: Begin function test_exp10_f32_vec4
; CHECK: %[[#v4_32_arg:]] = OpFunctionParameter %[[#Vec4_32Ty]]
; CHECK: %[[#v4_32_mul:]] = OpVectorTimesScalar %[[#Vec4_32Ty]] %[[#v4_32_arg]] %[[#Constant_s32]]
; CHECK: %[[#v4_32_ret:]] = OpExtInst %[[#Vec4_32Ty]] %[[#ExtInstId]] Exp2 %[[#v4_32_mul]]
; CHECK: OpReturnValue %[[#v4_32_ret]]
; CHECK-LABEL: OpFunctionEnd
define <4 x float> @test_exp10_f32_vec4(<4 x float>  %x) {
    %res = call <4 x float>  @llvm.exp10.f32(<4 x float>  %x)
    ret <4 x float>  %res
}

; CHECK-LABEL: Begin function test_exp10_f16_scalar
; CHECK: %[[#s16_arg:]] = OpFunctionParameter %[[#F16Ty]]
; CHECK: %[[#s16_mul:]] = OpFMul %[[#F16Ty]] %[[#s16_arg]] %[[#Constant_s16]]
; CHECK: %[[#s16_ret:]] = OpExtInst %[[#F16Ty]] %[[#ExtInstId]] Exp2 %[[#s16_mul]]
; CHECK: OpReturnValue %[[#s16_ret]]
; CHECK-LABEL: OpFunctionEnd
define half @test_exp10_f16_scalar(half %x) {
    %res = call half @llvm.exp10.f16(half %x)
    ret half %res
}

; CHECK-LABEL: Begin function test_exp10_f16_vec4
; CHECK: %[[#v4_16_arg:]] = OpFunctionParameter %[[#Vec4_16Ty]]
; CHECK: %[[#v4_16_mul:]] = OpVectorTimesScalar %[[#Vec4_16Ty]] %[[#v4_16_arg]] %[[#Constant_s16]]
; CHECK: %[[#v4_16_ret:]] = OpExtInst %[[#Vec4_16Ty]] %[[#ExtInstId]] Exp2 %[[#v4_16_mul]]
; CHECK: OpReturnValue %[[#v4_16_ret]]
; CHECK-LABEL: OpFunctionEnd
define <4 x half> @test_exp10_f16_vec4(<4 x half>  %x) {
    %res = call <4 x half>  @llvm.exp10.f16(<4 x half>  %x)
    ret <4 x half>  %res
}
