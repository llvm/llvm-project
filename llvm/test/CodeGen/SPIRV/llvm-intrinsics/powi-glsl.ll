; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; Test that llvm.powi on Vulkan targets is lowered by converting the
; integer exponent to float with OpConvertSToF, then calling GLSL.std.450 Pow.

; CHECK-DAG: %[[#ExtInstId:]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: %[[#F32Ty:]] = OpTypeFloat 32
; CHECK-DAG: %[[#I32Ty:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#I64Ty:]] = OpTypeInt 64 0

; CHECK-LABEL: Begin function test_powi_f32_i32
; CHECK: %[[#base32:]] = OpFunctionParameter %[[#F32Ty]]
; CHECK: %[[#exp32:]] = OpFunctionParameter %[[#I32Ty]]
; CHECK: %[[#fexp32:]] = OpConvertSToF %[[#F32Ty]] %[[#exp32]]
; CHECK: %[[#ret32:]] = OpExtInst %[[#F32Ty]] %[[#ExtInstId]] Pow %[[#base32]] %[[#fexp32]]
; CHECK: OpReturnValue %[[#ret32]]
; CHECK-LABEL: OpFunctionEnd
define internal float @test_powi_f32_i32(float %x, i32 %n) {
  %res = call float @llvm.powi.f32.i32(float %x, i32 %n)
  ret float %res
}

; CHECK-LABEL: Begin function test_powi_f32_i64
; CHECK: %[[#base64:]] = OpFunctionParameter %[[#F32Ty]]
; CHECK: %[[#exp64:]] = OpFunctionParameter %[[#I64Ty]]
; CHECK: %[[#fexp64:]] = OpConvertSToF %[[#F32Ty]] %[[#exp64]]
; CHECK: %[[#ret64:]] = OpExtInst %[[#F32Ty]] %[[#ExtInstId]] Pow %[[#base64]] %[[#fexp64]]
; CHECK: OpReturnValue %[[#ret64]]
; CHECK-LABEL: OpFunctionEnd
define internal float @test_powi_f32_i64(float %x, i64 %n) {
  %res = call float @llvm.powi.f32.i64(float %x, i64 %n)
  ret float %res
}

define void @main() #0 {
  ret void
}

declare float @llvm.powi.f32.i32(float, i32)
declare float @llvm.powi.f32.i64(float, i64)

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
