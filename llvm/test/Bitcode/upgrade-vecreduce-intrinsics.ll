; RUN: opt -S < %s | FileCheck %s
; RUN: llvm-dis < %s.bc | FileCheck %s

define float @fadd_acc(<4 x float> %in, float %acc) {
; CHECK-LABEL: @fadd_acc
; CHECK: %res = call float @llvm.experimental.vector.reduce.v2.fadd.f32.v4f32(float %acc, <4 x float> %in)
  %res = call float @llvm.experimental.vector.reduce.fadd.f32.v4f32(float %acc, <4 x float> %in)
  ret float %res
}

define float @fadd_undef(<4 x float> %in) {
; CHECK-LABEL: @fadd_undef
; CHECK: %res = call float @llvm.experimental.vector.reduce.v2.fadd.f32.v4f32(float undef, <4 x float> %in)
  %res = call float @llvm.experimental.vector.reduce.fadd.f32.v4f32(float undef, <4 x float> %in)
  ret float %res
}

define float @fadd_fast_acc(<4 x float> %in, float %acc) {
; CHECK-LABEL: @fadd_fast_acc
; CHECK: %res = call fast float @llvm.experimental.vector.reduce.v2.fadd.f32.v4f32(float 0.000000e+00, <4 x float> %in)
  %res = call fast float @llvm.experimental.vector.reduce.fadd.f32.v4f32(float %acc, <4 x float> %in)
  ret float %res
}

define float @fadd_fast_undef(<4 x float> %in) {
; CHECK-LABEL: @fadd_fast_undef
; CHECK: %res = call fast float @llvm.experimental.vector.reduce.v2.fadd.f32.v4f32(float 0.000000e+00, <4 x float> %in)
  %res = call fast float @llvm.experimental.vector.reduce.fadd.f32.v4f32(float undef, <4 x float> %in)
  ret float %res
}

define float @fmul_acc(<4 x float> %in, float %acc) {
; CHECK-LABEL: @fmul_acc
; CHECK: %res = call float @llvm.experimental.vector.reduce.v2.fmul.f32.v4f32(float %acc, <4 x float> %in)
  %res = call float @llvm.experimental.vector.reduce.fmul.f32.v4f32(float %acc, <4 x float> %in)
  ret float %res
}

define float @fmul_undef(<4 x float> %in) {
; CHECK-LABEL: @fmul_undef
; CHECK: %res = call float @llvm.experimental.vector.reduce.v2.fmul.f32.v4f32(float undef, <4 x float> %in)
  %res = call float @llvm.experimental.vector.reduce.fmul.f32.v4f32(float undef, <4 x float> %in)
  ret float %res
}

define float @fmul_fast_acc(<4 x float> %in, float %acc) {
; CHECK-LABEL: @fmul_fast_acc
; CHECK: %res = call fast float @llvm.experimental.vector.reduce.v2.fmul.f32.v4f32(float 1.000000e+00, <4 x float> %in)
  %res = call fast float @llvm.experimental.vector.reduce.fmul.f32.v4f32(float %acc, <4 x float> %in)
  ret float %res
}

define float @fmul_fast_undef(<4 x float> %in) {
; CHECK-LABEL: @fmul_fast_undef
; CHECK: %res = call fast float @llvm.experimental.vector.reduce.v2.fmul.f32.v4f32(float 1.000000e+00, <4 x float> %in)
  %res = call fast float @llvm.experimental.vector.reduce.fmul.f32.v4f32(float undef, <4 x float> %in)
  ret float %res
}

declare float @llvm.experimental.vector.reduce.fadd.f32.v4f32(float, <4 x float>)
; CHECK: declare float @llvm.experimental.vector.reduce.v2.fadd.f32.v4f32(float, <4 x float>)

declare float @llvm.experimental.vector.reduce.fmul.f32.v4f32(float, <4 x float>)
; CHECK: declare float @llvm.experimental.vector.reduce.v2.fmul.f32.v4f32(float, <4 x float>)
