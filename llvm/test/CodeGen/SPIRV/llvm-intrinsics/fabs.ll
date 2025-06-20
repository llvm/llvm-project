; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

; CHECK: %[[#extinst_id:]] = OpExtInstImport "OpenCL.std"

; CHECK: %[[#var0:]] = OpTypeFloat 16
; CHECK: %[[#var1:]] = OpTypeFloat 32
; CHECK: %[[#var2:]] = OpTypeFloat 64
; CHECK: %[[#var3:]] = OpTypeVector %[[#var1]] 4

; CHECK: OpFunction
; CHECK: %[[#]] = OpExtInst %[[#var0]] %[[#extinst_id]] fabs
; CHECK: OpFunctionEnd

define spir_func half @TestFabs16(half %x) local_unnamed_addr {
entry:
  %0 = tail call half @llvm.fabs.f16(half %x)
  ret half %0
}

; CHECK: OpFunction
; CHECK: %[[#]] = OpExtInst %[[#var1]] %[[#extinst_id]] fabs
; CHECK: OpFunctionEnd

define spir_func float @TestFabs32(float %x) local_unnamed_addr {
entry:
  %0 = tail call float @llvm.fabs.f32(float %x)
  ret float %0
}

; CHECK: OpFunction
; CHECK: %[[#]] = OpExtInst %[[#var2]] %[[#extinst_id]] fabs
; CHECK: OpFunctionEnd

define spir_func double @TestFabs64(double %x) local_unnamed_addr {
entry:
  %0 = tail call double @llvm.fabs.f64(double %x)
  ret double %0
}

; CHECK: OpFunction
; CHECK: %[[#]] = OpExtInst %[[#var3]] %[[#extinst_id]] fabs
; CHECK: OpFunctionEnd

define spir_func <4 x float> @TestFabsVec(<4 x float> %x) local_unnamed_addr {
entry:
  %0 = tail call <4 x float> @llvm.fabs.v4f32(<4 x float> %x)
  ret <4 x float> %0
}

declare half @llvm.fabs.f16(half)

declare float @llvm.fabs.f32(float)

declare double @llvm.fabs.f64(double)

declare <4 x float> @llvm.fabs.v4f32(<4 x float>)
