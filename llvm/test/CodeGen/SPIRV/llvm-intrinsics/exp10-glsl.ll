; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val %}

; Test if llvm.exp10 is lowered with the result correctly reused by the
; original llvm.exp10 user.


; CHECK-DAG: %[[#ExtInstId:]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: %[[#F32Ty:]] = OpTypeFloat 32
; CHECK-DAG: %[[#Vec4_32Ty:]] = OpTypeVector %[[#F32Ty]] 4
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
