; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#extinst:]] = OpExtInstImport "GLSL.std.450"

; CHECK-DAG: %[[#half:]] = OpTypeFloat 16
; CHECK-DAG: %[[#v4half:]] = OpTypeVector %[[#half]] 4
; CHECK-DAG: %[[#float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#v4float:]] = OpTypeVector %[[#float]] 4
; CHECK-DAG: %[[#float_0_30103001:]] = OpConstant %[[#float]] 0.30103000998497009
; CHECK-DAG: %[[#half_0_30103001:]] = OpConstant %[[#half]] 13521

@logf = global float 0.0, align 4
@logf4 = global <4 x float> zeroinitializer, align 16
@logh = global half 0.0, align 2
@logh4 = global <4 x half> zeroinitializer, align 8

define void @main(float %f, <4 x float> %f4, half %h, <4 x half> %h4) {
entry:
; CHECK-DAG: %[[#f:]] = OpFunctionParameter %[[#float]]
; CHECK-DAG: %[[#f4:]] = OpFunctionParameter %[[#v4float]]
; CHECK-DAG: %[[#h:]] = OpFunctionParameter %[[#half]]
; CHECK-DAG: %[[#h4:]] = OpFunctionParameter %[[#v4half]]

; CHECK: %[[#log2:]] = OpExtInst %[[#float]] %[[#extinst]] Log2 %[[#f]]
; CHECK: %[[#res:]] = OpFMul %[[#float]] %[[#log2]] %[[#float_0_30103001]]
  %elt.log10 = call float @llvm.log10.f32(float %f)
  store float %elt.log10, ptr @logf, align 4

; CHECK: %[[#log2:]] = OpExtInst %[[#v4float]] %[[#extinst]] Log2 %[[#f4]]
; CHECK: %[[#res:]] = OpVectorTimesScalar %[[#v4float]] %[[#log2]] %[[#float_0_30103001]]
  %elt.log101 = call <4 x float> @llvm.log10.v4f32(<4 x float> %f4)
  store <4 x float> %elt.log101, ptr @logf4, align 16

; CHECK: %[[#log2:]] = OpExtInst %[[#half]] %[[#extinst]] Log2 %[[#h]]
; CHECK: %[[#res:]] = OpFMul %[[#half]] %[[#log2]] %[[#half_0_30103001]]
  %elt.log10h = call half @llvm.log10.f16(half %h)
  store half %elt.log10h, ptr @logh, align 2

; CHECK: %[[#log2:]] = OpExtInst %[[#v4half]] %[[#extinst]] Log2 %[[#h4]]
; CHECK: %[[#res:]] = OpVectorTimesScalar %[[#v4half]] %[[#log2]] %[[#half_0_30103001]]
  %elt.log10h4 = call <4 x half> @llvm.log10.v4f16(<4 x half> %h4)
  store <4 x half> %elt.log10h4, ptr @logh4, align 8

  ret void
}

declare float @llvm.log10.f32(float)
declare <4 x float> @llvm.log10.v4f32(<4 x float>)
declare half @llvm.log10.f16(half)
declare <4 x half> @llvm.log10.v4f16(<4 x half>)
