; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Test if llvm.sincos is lowered to opencl::sincos with the result
; correctly reused by the original llvm.sincos user.

; CHECK-DAG: %[[#ExtInstId:]] = OpExtInstImport "OpenCL.std"
; CHECK-DAG: %[[#FloatTy:]] = OpTypeFloat 32
; CHECK-DAG: %[[#FnPtrTy:]] = OpTypePointer Function %[[#FloatTy]]
; CHECK-DAG: %[[#Vec2FloatTy:]] = OpTypeVector %[[#FloatTy]] 2
; CHECK-DAG: %[[#FnPtrVec2Ty:]] = OpTypePointer Function %[[#Vec2FloatTy]]

; CHECK: %[[#XParam:]] = OpFunctionParameter %[[#FloatTy]]
; CHECK: %[[#Var:]] = OpVariable %[[#FnPtrTy]] Function
; CHECK: %[[#SinRes:]] = OpExtInst %[[#FloatTy]] %[[#ExtInstId]] sincos %[[#XParam]] %[[#Var]]
; CHECK: %[[#CosRes:]] = OpLoad %[[#FloatTy]] %[[#Var]]
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
; CHECK: %[[#Varv:]] = OpVariable %[[#FnPtrVec2Ty]] Function
; CHECK: %[[#SinResv:]] = OpExtInst %[[#Vec2FloatTy]] %[[#ExtInstId]] sincos %[[#XvParam]] %[[#Varv]]
; CHECK: %[[#CosResv:]] = OpLoad %[[#Vec2FloatTy]] %[[#Varv]]
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
