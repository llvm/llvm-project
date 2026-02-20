; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val %}

; Test if llvm.exp10 is lowered with the result correctly reused by the
; original llvm.exp10 user.


; CHECK-DAG: %[[#ExtInstId:]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: %[[#F16Ty:]] = OpTypeFloat 16
; CHECK-DAG: %[[#F32Ty:]] = OpTypeFloat 32
; CHECK-DAG: %[[#F64Ty:]] = OpTypeFloat 64
; CHECK-DAG: %[[#Vec2_16Ty:]] = OpTypeVector %[[#F16Ty]] 2
; CHECK-DAG: %[[#Vec2_32Ty:]] = OpTypeVector %[[#F32Ty]] 2
; CHECK-DAG: %[[#Vec2_64Ty:]] = OpTypeVector %[[#F64Ty]] 2
; CHECK-DAG: %[[#Vec4_16Ty:]] = OpTypeVector %[[#F16Ty]] 4
; CHECK-DAG: %[[#Vec4_32Ty:]] = OpTypeVector %[[#F32Ty]] 4
; CHECK-DAG: %[[#Vec4_64Ty:]] = OpTypeVector %[[#F64Ty]] 4
; CHECK-DAG: %[[#Constant_s16:]] = OpConstant %[[#F16Ty:]] 3.3219280242919922
; CHECK-DAG: %[[#Constant_s32:]] = OpConstant %[[#F32Ty:]] 3.3219280242919922
; CHECK-DAG: %[[#Constant_s64:]] = OpConstant %[[#F64Ty:]] 3.3219280242919922

; CHECK-LABEL: Begin function test_exp10_f16_scalar
; CHECK: %[[#s16_arg:]] = OpFunctionParameter %[[#F16Ty]]
; CHECK: %[[#s16_mul:]] = OpFMul %[[#F16Ty]] %[[#Constant_s16]] %[[#s16_arg]]
; CHECK: %[[#s16_ret:]] = OpExtInst %[[#F16Ty]] %[[#ExtInstId]] Exp2 %[[#s16_mul]]
; CHECK: OpReturnValue %[[#s16_ret]]
; CHECK-LABEL: OpFunctionEnd
define half @test_exp10_f16_scalar(half %x) {
    %res = call half @llvm.exp10.f16(half %x)
    ret half %res
}

; CHECK-LABEL: Begin function test_exp10_f32_scalar
; CHECK: %[[#s32_arg:]] = OpFunctionParameter %[[#F32Ty]]
; CHECK: %[[#s32_mul:]] = OpFMul %[[#F32Ty]] %[[#Constant_s32]] %[[#s32_arg]]
; CHECK: %[[#s32_ret:]] = OpExtInst %[[#F32Ty]] %[[#ExtInstId]] Exp2 %[[#s32_mul]]
; CHECK: OpReturnValue %[[#s32_ret]]
; CHECK-LABEL: OpFunctionEnd
define float @test_exp10_f32_scalar(float %x) {
    %res = call float @llvm.exp10.f32(float %x)
    ret float %res
}

; CHECK-LABEL: Begin function test_exp10_f64_scalar
; CHECK: %[[#s64_arg:]] = OpFunctionParameter %[[#F64Ty]]
; CHECK: %[[#s64_mul:]] = OpFMul %[[#F64Ty]] %[[#Constant_s64]] %[[#s64_arg]]
; CHECK: %[[#s64_ret:]] = OpExtInst %[[#F64Ty]] %[[#ExtInstId]] Exp2 %[[#s64_mul]]
; CHECK: OpReturnValue %[[#s64_ret]]
; CHECK-LABEL: OpFunctionEnd
define double @test_exp10_f64_scalar(double %x) {
    %res = call double @llvm.exp10.f64(double %x)
    ret double %res
}

; CHECK-LABEL: Begin function test_exp10_f16_vec2
; CHECK: %[[#v2_16_arg:]] = OpFunctionParameter %[[#Vec2_16Ty]]
; CHECK: %[[#v2_16_mul:]] = OpVectorTimesScalar %[[#Vec2_16Ty]] %[[#Constant_s16]] %[[#v2_16_arg]]
; CHECK: %[[#v2_16_ret:]] = OpExtInst %[[#Vec2_16Ty]] %[[#ExtInstId]] Exp2 %[[#v2_16_mul]]
; CHECK: OpReturnValue %[[#v2_16_ret]]
; CHECK-LABEL: OpFunctionEnd
define <2 x half> @test_exp10_f16_vec2(<2 x half>  %x) {
    %res = call <2 x half>  @llvm.exp10.f16(<2 x half>  %x)
    ret <2 x half>  %res
}

; CHECK-LABEL: Begin function test_exp10_f32_vec2
; CHECK: %[[#v2_32_arg:]] = OpFunctionParameter %[[#Vec2_32Ty]]
; CHECK: %[[#v2_32_mul:]] = OpVectorTimesScalar %[[#Vec2_32Ty]] %[[#Constant_s32]] %[[#v2_32_arg]]
; CHECK: %[[#v2_32_ret:]] = OpExtInst %[[#Vec2_32Ty]] %[[#ExtInstId]] Exp2 %[[#v2_32_mul]]
; CHECK: OpReturnValue %[[#v2_32_ret]]
; CHECK-LABEL: OpFunctionEnd
define <2 x float> @test_exp10_f32_vec2(<2 x float>  %x) {
    %res = call <2 x float>  @llvm.exp10.f32(<2 x float>  %x)
    ret <2 x float>  %res
}

; CHECK-LABEL: Begin function test_exp10_f64_vec2
; CHECK: %[[#v2_64_arg:]] = OpFunctionParameter %[[#Vec2_64Ty]]
; CHECK: %[[#v2_64_mul:]] = OpVectorTimesScalar %[[#Vec2_64Ty]] %[[#Constant_s64]] %[[#v2_64_arg]]
; CHECK: %[[#v2_64_ret:]] = OpExtInst %[[#Vec2_64Ty]] %[[#ExtInstId]] Exp2 %[[#v2_64_mul]]
; CHECK: OpReturnValue %[[#v2_64_ret]]
; CHECK-LABEL: OpFunctionEnd
define <2 x double> @test_exp10_f64_vec2(<2 x double>  %x) {
    %res = call <2 x double>  @llvm.exp10.f64(<2 x double>  %x)
    ret <2 x double>  %res
}


; CHECK-LABEL: Begin function test_exp10_f16_vec4
; CHECK: %[[#v4_16_arg:]] = OpFunctionParameter %[[#Vec4_16Ty]]
; CHECK: %[[#v4_16_mul:]] = OpVectorTimesScalar %[[#Vec4_16Ty]] %[[#Constant_s16]] %[[#v4_16_arg]]
; CHECK: %[[#v4_16_ret:]] = OpExtInst %[[#Vec4_16Ty]] %[[#ExtInstId]] Exp2 %[[#v4_16_mul]]
; CHECK: OpReturnValue %[[#v4_16_ret]]
; CHECK-LABEL: OpFunctionEnd
define <4 x half> @test_exp10_f16_vec4(<4 x half>  %x) {
    %res = call <4 x half>  @llvm.exp10.f16(<4 x half>  %x)
    ret <4 x half>  %res
}

; CHECK-LABEL: Begin function test_exp10_f32_vec4
; CHECK: %[[#v4_32_arg:]] = OpFunctionParameter %[[#Vec4_32Ty]]
; CHECK: %[[#v4_32_mul:]] = OpVectorTimesScalar %[[#Vec4_32Ty]] %[[#Constant_s32]] %[[#v4_32_arg]]
; CHECK: %[[#v4_32_ret:]] = OpExtInst %[[#Vec4_32Ty]] %[[#ExtInstId]] Exp2 %[[#v4_32_mul]]
; CHECK: OpReturnValue %[[#v4_32_ret]]
; CHECK-LABEL: OpFunctionEnd
define <4 x float> @test_exp10_f32_vec4(<4 x float>  %x) {
    %res = call <4 x float>  @llvm.exp10.f32(<4 x float>  %x)
    ret <4 x float>  %res
}

; CHECK-LABEL: Begin function test_exp10_f64_vec4
; CHECK: %[[#v4_64_arg:]] = OpFunctionParameter %[[#Vec4_64Ty]]
; CHECK: %[[#v4_64_mul:]] = OpVectorTimesScalar %[[#Vec4_64Ty]] %[[#Constant_s64]] %[[#v4_64_arg]]
; CHECK: %[[#v4_64_ret:]] = OpExtInst %[[#Vec4_64Ty]] %[[#ExtInstId]] Exp2 %[[#v4_64_mul]]
; CHECK: OpReturnValue %[[#v4_64_ret]]
; CHECK-LABEL: OpFunctionEnd
define <4 x double> @test_exp10_f64_vec4(<4 x double>  %x) {
    %res = call <4 x double>  @llvm.exp10.f64(<4 x double>  %x)
    ret <4 x double>  %res
}
