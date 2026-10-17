; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#ExtInstId:]] = OpExtInstImport "OpenCL.std"
; CHECK-DAG: %[[#Float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#Half:]] = OpTypeFloat 16
; CHECK-DAG: %[[#Float2:]] = OpTypeVector %[[#Float]] 2
; CHECK-DAG: %[[#Half3:]] = OpTypeVector %[[#Half]] 3
; CHECK-DAG: OpDecorate %[[#SinF:]] FPFastMathMode NotNaN|NotInf
; CHECK-DAG: OpDecorate %[[#CosF:]] FPFastMathMode NotNaN|NotInf

; CHECK: %[[#X:]] = OpFunctionParameter %[[#Float]]
; CHECK: %[[#Sin:]] = OpExtInst %[[#Float]] %[[#ExtInstId]] sinpi %[[#X]]
; CHECK: %[[#Cos:]] = OpExtInst %[[#Float]] %[[#ExtInstId]] cospi %[[#X]]
; CHECK: OpFAdd %[[#Float]] %[[#Sin]] %[[#Cos]]
define spir_func float @test_sincospi(float %x) {
  %r = call { float, float } @llvm.sincospi.f32(float %x)
  %sin = extractvalue { float, float } %r, 0
  %cos = extractvalue { float, float } %r, 1
  %sum = fadd float %sin, %cos
  ret float %sum
}

; CHECK: %[[#Xv:]] = OpFunctionParameter %[[#Float2]]
; CHECK: %[[#Sinv:]] = OpExtInst %[[#Float2]] %[[#ExtInstId]] sinpi %[[#Xv]]
; CHECK: %[[#Cosv:]] = OpExtInst %[[#Float2]] %[[#ExtInstId]] cospi %[[#Xv]]
; CHECK: OpFAdd %[[#Float2]] %[[#Sinv]] %[[#Cosv]]
define spir_func <2 x float> @test_sincospi_v2(<2 x float> %x) {
  %r = call { <2 x float>, <2 x float> } @llvm.sincospi.v2f32(<2 x float> %x)
  %sin = extractvalue { <2 x float>, <2 x float> } %r, 0
  %cos = extractvalue { <2 x float>, <2 x float> } %r, 1
  %sum = fadd <2 x float> %sin, %cos
  ret <2 x float> %sum
}

; CHECK: %[[#Xs:]] = OpFunctionParameter %[[#Float]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: %[[#SinOnly:]] = OpExtInst %[[#Float]] %[[#ExtInstId]] sinpi %[[#Xs]]
; CHECK-NEXT: OpReturnValue %[[#SinOnly]]
define spir_func float @test_sincospi_sin_only(float %x) {
  %r = call { float, float } @llvm.sincospi.f32(float %x)
  %sin = extractvalue { float, float } %r, 0
  ret float %sin
}

; CHECK: %[[#Xc:]] = OpFunctionParameter %[[#Half3]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: %[[#CosOnly:]] = OpExtInst %[[#Half3]] %[[#ExtInstId]] cospi %[[#Xc]]
; CHECK-NEXT: OpReturnValue %[[#CosOnly]]
define spir_func <3 x half> @test_sincospi_cos_only(<3 x half> %x) {
  %r = call { <3 x half>, <3 x half> } @llvm.sincospi.v3f16(<3 x half> %x)
  %cos = extractvalue { <3 x half>, <3 x half> } %r, 1
  ret <3 x half> %cos
}

; CHECK: %[[#Xf:]] = OpFunctionParameter %[[#Float]]
; CHECK: %[[#SinF]] = OpExtInst %[[#Float]] %[[#ExtInstId]] sinpi %[[#Xf]]
; CHECK: %[[#CosF]] = OpExtInst %[[#Float]] %[[#ExtInstId]] cospi %[[#Xf]]
define spir_func float @test_sincospi_fmf(float %x) {
  %r = call nnan ninf { float, float } @llvm.sincospi.f32(float %x)
  %sin = extractvalue { float, float } %r, 0
  %cos = extractvalue { float, float } %r, 1
  %sum = fadd float %sin, %cos
  ret float %sum
}
