; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

;; Test that llvm.sincospi is lowered to the OpenCL.std sinpi and cospi pair,
;; with both results reused by the original llvm.sincospi users.

; CHECK-DAG: %[[#ExtInstId:]] = OpExtInstImport "OpenCL.std"
; CHECK-DAG: %[[#FloatTy:]] = OpTypeFloat 32
; CHECK-DAG: %[[#Vec2FloatTy:]] = OpTypeVector %[[#FloatTy]] 2

; CHECK: %[[#XParam:]] = OpFunctionParameter %[[#FloatTy]]
; CHECK: %[[#SinRes:]] = OpExtInst %[[#FloatTy]] %[[#ExtInstId]] sinpi %[[#XParam]]
; CHECK: %[[#CosRes:]] = OpExtInst %[[#FloatTy]] %[[#ExtInstId]] cospi %[[#XParam]]
; CHECK: %[[#Sum:]] = OpFAdd %[[#FloatTy]] %[[#SinRes]] %[[#CosRes]]
; CHECK: OpReturnValue %[[#Sum]]
define float @test_sincospi_scalar(float %x) {
  %result = call { float, float } @llvm.sincospi.f32(float %x)
  %sin = extractvalue { float, float } %result, 0
  %cos = extractvalue { float, float } %result, 1
  %sum = fadd float %sin, %cos
  ret float %sum
}

; CHECK: %[[#XvParam:]] = OpFunctionParameter %[[#Vec2FloatTy]]
; CHECK: %[[#SinResv:]] = OpExtInst %[[#Vec2FloatTy]] %[[#ExtInstId]] sinpi %[[#XvParam]]
; CHECK: %[[#CosResv:]] = OpExtInst %[[#Vec2FloatTy]] %[[#ExtInstId]] cospi %[[#XvParam]]
; CHECK: %[[#Sumv:]] = OpFAdd %[[#Vec2FloatTy]] %[[#SinResv]] %[[#CosResv]]
; CHECK: OpReturnValue %[[#Sumv]]
define <2 x float> @test_sincospi_vec2(<2 x float> %x) {
  %result = call { <2 x float>, <2 x float> } @llvm.sincospi.v2f32(<2 x float> %x)
  %sin = extractvalue { <2 x float>, <2 x float> } %result, 0
  %cos = extractvalue { <2 x float>, <2 x float> } %result, 1
  %sum = fadd <2 x float> %sin, %cos
  ret <2 x float> %sum
}

;; An unused cosine result must not emit a cospi instruction.
; CHECK: %[[#XoParam:]] = OpFunctionParameter %[[#FloatTy]]
; CHECK-NOT: cospi
; CHECK: %[[#SinOnlyRes:]] = OpExtInst %[[#FloatTy]] %[[#ExtInstId]] sinpi %[[#XoParam]]
; CHECK-NOT: cospi
; CHECK: OpReturnValue %[[#SinOnlyRes]]
define float @test_sincospi_sin_only(float %x) {
  %result = call { float, float } @llvm.sincospi.f32(float %x)
  %sin = extractvalue { float, float } %result, 0
  ret float %sin
}

declare { float, float } @llvm.sincospi.f32(float)
declare { <2 x float>, <2 x float> } @llvm.sincospi.v2f32(<2 x float>)
