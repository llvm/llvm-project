; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Test if llvm.exp10 is lowered to opencl::exp10 with the result correctly
;reused by the original llvm.exp10 user.


; CHECK-DAG: %[[#ExtInstId:]] = OpExtInstImport "OpenCL.std"
; CHECK-DAG: %[[#F16Ty:]] = OpTypeFloat 16
; CHECK-DAG: %[[#F32Ty:]] = OpTypeFloat 32
; CHECK-DAG: %[[#F64Ty:]] = OpTypeFloat 64
; CHECK-DAG: %[[#Vec2_16Ty:]] = OpTypeVector %[[#F16Ty]] 2
; CHECK-DAG: %[[#Vec2_32Ty:]] = OpTypeVector %[[#F32Ty]] 2
; CHECK-DAG: %[[#Vec2_64Ty:]] = OpTypeVector %[[#F64Ty]] 2
; CHECK-DAG: %[[#Vec16_16Ty:]] = OpTypeVector %[[#F16Ty]] 16
; CHECK-DAG: %[[#Vec16_32y:]] = OpTypeVector %[[#F32Ty]] 16
; CHECK-DAG: %[[#Vec16_64Ty:]] = OpTypeVector %[[#F64Ty]] 16

; CHECK-LABEL: Begin function test_exp10_f16_scalar
; CHECK: %[[#s16_arg:]] = OpFunctionParameter %[[#F16Ty]]
; CHECK: %[[#s16_ret:]] = OpExtInst %[[#F16Ty]] %[[#ExtInstId]] exp10 %[[#s16_arg]]
; CHECK: OpReturnValue %[[#s16_ret]]
; CHECK-LABEL: OpFunctionEnd
define half @test_exp10_f16_scalar(half %x) {
    %res = call half @llvm.exp10.f16(half %x)
    ret half %res
}

; CHECK-LABEL: Begin function test_exp10_f32_scalar
; CHECK: %[[#s32_arg:]] = OpFunctionParameter %[[#F32Ty]]
; CHECK: %[[#s32_ret:]] = OpExtInst %[[#F32Ty]] %[[#ExtInstId]] exp10 %[[#s32_arg]]
; CHECK: OpReturnValue %[[#s32_ret]]
; CHECK-LABEL: OpFunctionEnd
define float @test_exp10_f32_scalar(float %x) {
    %res = call float @llvm.exp10.f32(float %x)
    ret float %res
}

; CHECK-LABEL: Begin function test_exp10_f64_scalar
; CHECK: %[[#s64_arg:]] = OpFunctionParameter %[[#F64Ty]]
; CHECK: %[[#s64_ret:]] = OpExtInst %[[#F64Ty]] %[[#ExtInstId]] exp10 %[[#s64_arg]]
; CHECK: OpReturnValue %[[#s64_ret]]
; CHECK-LABEL: OpFunctionEnd
define double @test_exp10_f64_scalar(double %x) {
    %res = call double @llvm.exp10.f64(double %x)
    ret double %res
}

; CHECK-LABEL: Begin function test_exp10_f16_vec2
; CHECK: %[[#v2_16_arg:]] = OpFunctionParameter %[[#Vec2_16Ty]]
; CHECK: %[[#v2_16_ret:]] = OpExtInst %[[#Vec2_16Ty]] %[[#ExtInstId]] exp10 %[[#v2_16_arg]]
; CHECK: OpReturnValue %[[#v2_16_ret]]
; CHECK-LABEL: OpFunctionEnd
define <2 x half> @test_exp10_f16_vec2(<2 x half>  %x) {
    %res = call <2 x half>  @llvm.exp10.v2f16(<2 x half>  %x)
    ret <2 x half>  %res
}

; CHECK-LABEL: Begin function test_exp10_f32_vec2
; CHECK: %[[#v2_32_arg:]] = OpFunctionParameter %[[#Vec2_32Ty]]
; CHECK: %[[#v2_32_ret:]] = OpExtInst %[[#Vec2_32Ty]] %[[#ExtInstId]] exp10 %[[#v2_32_arg]]
; CHECK: OpReturnValue %[[#v2_32_ret]]
; CHECK-LABEL: OpFunctionEnd
define <2 x float> @test_exp10_f32_vec2(<2 x float>  %x) {
    %res = call <2 x float>  @llvm.exp10.v2f32(<2 x float>  %x)
    ret <2 x float>  %res
}

; CHECK-LABEL: Begin function test_exp10_f64_vec2
; CHECK: %[[#v2_64_arg:]] = OpFunctionParameter %[[#Vec2_64Ty]]
; CHECK: %[[#v2_64_ret:]] = OpExtInst %[[#Vec2_64Ty]] %[[#ExtInstId]] exp10 %[[#v2_64_arg]]
; CHECK: OpReturnValue %[[#v2_64_ret]]
; CHECK-LABEL: OpFunctionEnd
define <2 x double> @test_exp10_f64_vec2(<2 x double>  %x) {
    %res = call <2 x double>  @llvm.exp10.v2f64(<2 x double>  %x)
    ret <2 x double>  %res
}


; CHECK-LABEL: Begin function test_exp10_f16_vec16
; CHECK: %[[#v16_16_arg:]] = OpFunctionParameter %[[#Vec16_16Ty]]
; CHECK: %[[#v16_16_ret:]] = OpExtInst %[[#Vec16_16Ty]] %[[#ExtInstId]] exp10 %[[#v16_16_arg]]
; CHECK: OpReturnValue %[[#v16_16_ret]]
; CHECK-LABEL: OpFunctionEnd
define <16 x half> @test_exp10_f16_vec16(<16 x half>  %x) {
    %res = call <16 x half>  @llvm.exp10.v16f16(<16 x half>  %x)
    ret <16 x half>  %res
}

; CHECK-LABEL: Begin function test_exp10_f32_vec16
; CHECK: %[[#v16_32_arg:]] = OpFunctionParameter %[[#Vec16_32y]]
; CHECK: %[[#v16_32_ret:]] = OpExtInst %[[#Vec16_32y]] %[[#ExtInstId]] exp10 %[[#v16_32_arg]]
; CHECK: OpReturnValue %[[#v16_32_ret]]
; CHECK-LABEL: OpFunctionEnd
define <16 x float> @test_exp10_f32_vec16(<16 x float>  %x) {
    %res = call <16 x float>  @llvm.exp10.v16f32(<16 x float>  %x)
    ret <16 x float>  %res
}

; CHECK-LABEL: Begin function test_exp10_f64_vec16
; CHECK: %[[#v16_64_arg:]] = OpFunctionParameter %[[#Vec16_64Ty]]
; CHECK: %[[#v16_64_ret:]] = OpExtInst %[[#Vec16_64Ty]] %[[#ExtInstId]] exp10 %[[#v16_64_arg]]
; CHECK: OpReturnValue %[[#v16_64_ret]]
; CHECK-LABEL: OpFunctionEnd
define <16 x double> @test_exp10_f64_vec16(<16 x double>  %x) {
    %res = call <16 x double>  @llvm.exp10.v16f64(<16 x double>  %x)
    ret <16 x double>  %res
}
