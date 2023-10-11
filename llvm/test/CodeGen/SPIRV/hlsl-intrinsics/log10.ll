; RUN: llc -O0 -mtriple=spirv-unknown-linux %s -o - | FileCheck %s

; CHECK: %[[#extinst:]] = OpExtInstImport "GLSL.std.450"

; CHECK: %[[#float:]] = OpTypeFloat 32
; CHECK: %[[#v4float:]] = OpTypeVector %[[#float]] 4
; CHECK: %[[#float_0_30103001:]] = OpConstant %[[#float]] 0.30103000998497009
; CHECK: %[[#_ptr_Function_v4float:]] = OpTypePointer Function %[[#v4float]]
; CHECK: %[[#_ptr_Function_float:]] = OpTypePointer Function %[[#float]]

define void @main() {
entry:
; CHECK: %[[#f:]] = OpVariable %[[#_ptr_Function_float]] Function
; CHECK: %[[#logf:]] = OpVariable %[[#_ptr_Function_float]] Function
; CHECK: %[[#f4:]] = OpVariable %[[#_ptr_Function_v4float]] Function
; CHECK: %[[#logf4:]] = OpVariable %[[#_ptr_Function_v4float]] Function
  %f = alloca float, align 4
  %logf = alloca float, align 4
  %f4 = alloca <4 x float>, align 16
  %logf4 = alloca <4 x float>, align 16

; CHECK: %[[#load:]] = OpLoad %[[#float]] %[[#f]] Aligned 4
; CHECK: %[[#log2:]] = OpExtInst %[[#float]] %[[#extinst]] Log2 %[[#load]]
; CHECK: %[[#res:]] = OpFMul %[[#float]] %[[#log2]] %[[#float_0_30103001]]
; CHECK: OpStore %[[#logf]] %[[#res]] Aligned 4
  %0 = load float, ptr %f, align 4
  %elt.log10 = call float @llvm.log10.f32(float %0)
  store float %elt.log10, ptr %logf, align 4

; CHECK: %[[#load:]] = OpLoad %[[#v4float]] %[[#f4]] Aligned 16
; CHECK: %[[#log2:]] = OpExtInst %[[#v4float]] %[[#extinst]] Log2 %[[#load]]
; CHECK: %[[#res:]] = OpVectorTimesScalar %[[#v4float]] %[[#log2]] %[[#float_0_30103001]]
; CHECK: OpStore %[[#logf4]] %[[#res]] Aligned 16
  %1 = load <4 x float>, ptr %f4, align 16
  %elt.log101 = call <4 x float> @llvm.log10.v4f32(<4 x float> %1)
  store <4 x float> %elt.log101, ptr %logf4, align 16

  ret void
}

declare float @llvm.log10.f32(float)
declare <4 x float> @llvm.log10.v4f32(<4 x float>)
