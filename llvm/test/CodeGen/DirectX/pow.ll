; RUN: opt -S  -dxil-intrinsic-expansion -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s --check-prefixes=CHECK,EXPCHECK
; RUN: opt -S  -dxil-intrinsic-expansion -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s --check-prefixes=CHECK,DOPCHECK

; Make sure dxil operation function calls for pow are generated.

define noundef float @pow_float(float noundef %a, float noundef %b) {
entry:
; DOPCHECK: call float @dx.op.unary.f32(i32 23, float %a)
; EXPCHECK: call float @llvm.log2.f32(float %a)
; CHECK: fmul float %{{.*}}, %b
; DOPCHECK: call float @dx.op.unary.f32(i32 21, float %{{.*}})
; EXPCHECK: call float @llvm.exp2.f32(float %{{.*}})
  %elt.pow = call float @llvm.pow.f32(float %a, float %b)
  ret float %elt.pow
}

define noundef half @pow_half(half noundef %a, half noundef %b) {
entry:
; DOPCHECK: call half @dx.op.unary.f16(i32 23, half %a)
; EXPCHECK: call half @llvm.log2.f16(half %a)
; CHECK: fmul half %{{.*}}, %b
; DOPCHECK: call half @dx.op.unary.f16(i32 21, half %{{.*}})
; EXPCHECK: call half @llvm.exp2.f16(half %{{.*}})
  %elt.pow = call half @llvm.pow.f16(half %a, half %b)
  ret half %elt.pow
}

declare half @llvm.pow.f16(half,half)
declare float @llvm.pow.f32(float,float)
