; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#extinst:]] = OpExtInstImport "GLSL.std.450"

; CHECK: %[[#float:]] = OpTypeFloat 32
; CHECK: %[[#v4float:]] = OpTypeVector %[[#float]] 4
; CHECK: %[[#float_0_30103001:]] = OpConstant %[[#float]] 0.30103000998497009

define void @main(float %f, <4 x float> %f4) {
entry:
; CHECK-DAG: %[[#f:]] = OpFunctionParameter %[[#float]]
; CHECK-DAG: %[[#f4:]] = OpFunctionParameter %[[#v4float]]
  %logf = alloca float, align 4
  %logf4 = alloca <4 x float>, align 16


; CHECK: %[[#log2:]] = OpExtInst %[[#float]] %[[#extinst]] Log2 %[[#f]]
; CHECK: %[[#res:]] = OpFMul %[[#float]] %[[#log2]] %[[#float_0_30103001]]
  %elt.log10 = call float @llvm.log10.f32(float %f)

; CHECK: %[[#log2:]] = OpExtInst %[[#v4float]] %[[#extinst]] Log2 %[[#f4]]
; CHECK: %[[#res:]] = OpVectorTimesScalar %[[#v4float]] %[[#log2]] %[[#float_0_30103001]]
  %elt.log101 = call <4 x float> @llvm.log10.v4f32(<4 x float> %f4)

  ret void
}

declare float @llvm.log10.f32(float)
declare <4 x float> @llvm.log10.v4f32(<4 x float>)
