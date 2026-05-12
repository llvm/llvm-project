; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val %}

; Test if llvm.sincos is lowered to glsl::sin and glsl::cos with results
; correctly reused by the original llvm.sincos user.

; CHECK-DAG: %[[#ExtInstId:]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: %[[#FloatTy:]] = OpTypeFloat 32
; CHECK-DAG: %[[#Vec2FloatTy:]] = OpTypeVector %[[#FloatTy]] 2

; CHECK: %[[#XParam:]] = OpFunctionParameter %[[#FloatTy]]
; CHECK: %[[#SinRes:]] = OpExtInst %[[#FloatTy]] %[[#ExtInstId]] Sin %[[#XParam]]
; CHECK: %[[#CosRes:]] = OpExtInst %[[#FloatTy]] %[[#ExtInstId]] Cos %[[#XParam]]
; CHECK: %[[#Sum:]] = OpFAdd %[[#FloatTy]] %[[#SinRes]] %[[#CosRes]]
; CHECK: OpReturnValue %[[#Sum]]
define float @test_sincos_scalar(float %x) {
  %result = call { float, float } @llvm.sincos.f32(float %x)
  %sin = extractvalue { float, float } %result, 0
  %cos = extractvalue { float, float } %result, 1
  %sum = fadd float %sin, %cos
  ret float %sum
}

; CHECK: %[[#XvParam:]] = OpFunctionParameter %[[#Vec2FloatTy]]
; CHECK: %[[#SinResv:]] = OpExtInst %[[#Vec2FloatTy]] %[[#ExtInstId]] Sin %[[#XvParam]]
; CHECK: %[[#CosResv:]] = OpExtInst %[[#Vec2FloatTy]] %[[#ExtInstId]] Cos %[[#XvParam]]
; CHECK: %[[#Sumv:]] = OpFAdd %[[#Vec2FloatTy]] %[[#SinResv]] %[[#CosResv]]
; CHECK: OpReturnValue %[[#Sumv]]
define <2 x float> @test_sincos_vec2(<2 x float> %x) {
  %result = call { <2 x float>, <2 x float> } @llvm.sincos.v2f32(<2 x float> %x)
  %sin = extractvalue { <2 x float>, <2 x float> } %result, 0
  %cos = extractvalue { <2 x float>, <2 x float> } %result, 1
  %sum = fadd <2 x float> %sin, %cos
  ret <2 x float> %sum
}

declare { float, float } @llvm.sincos.f32(float)
declare { <2 x float>, <2 x float> } @llvm.sincos.v2f32(<2 x float>)
