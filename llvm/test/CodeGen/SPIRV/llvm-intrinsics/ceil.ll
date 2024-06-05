; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

; CHECK: %[[#extinst_id:]] = OpExtInstImport "OpenCL.std"

; CHECK: %[[#var1:]] = OpTypeFloat 32
; CHECK: %[[#var2:]] = OpTypeFloat 64
; CHECK: %[[#var3:]] = OpTypeVector %[[#var1]] 4

; CHECK: OpFunction
; CHECK: %[[#]] = OpExtInst %[[#var1]] %[[#extinst_id]] ceil
; CHECK: OpFunctionEnd

define spir_func float @TestCeil32(float %x) local_unnamed_addr {
entry:
  %0 = tail call float @llvm.ceil.f32(float %x)
  ret float %0
}

; CHECK: OpFunction
; CHECK: %[[#]] = OpExtInst %[[#var2]] %[[#extinst_id]] ceil
; CHECK: OpFunctionEnd

define spir_func double @TestCeil64(double %x) local_unnamed_addr {
entry:
  %0 = tail call double @llvm.ceil.f64(double %x)
  ret double %0
}

; CHECK: OpFunction
; CHECK: %[[#]] = OpExtInst %[[#var3]] %[[#extinst_id]] ceil
; CHECK: OpFunctionEnd

define spir_func <4 x float> @TestCeilVec(<4 x float> %x) local_unnamed_addr {
entry:
  %0 = tail call <4 x float> @llvm.ceil.v4f32(<4 x float> %x)
  ret <4 x float> %0
}

declare float @llvm.ceil.f32(float)

declare double @llvm.ceil.f64(double)

declare <4 x float> @llvm.ceil.v4f32(<4 x float>)
