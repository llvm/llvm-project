; RUN: opt -S  -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Make sure dxil operation function calls for ddy_fine are generated for half/float and matching vectors

define noundef half @deriv_fine_y_half(half noundef %a) {
; CHECK: call half @dx.op.unary.f16(i32 86, half %{{.*}})
entry:
  %dx.ddy.fine = call half @llvm.dx.ddy.fine.f16(half %a)
  ret half %dx.ddy.fine
}

define noundef float @deriv_fine_y_float(float noundef %a) {
; CHECK: call float @dx.op.unary.f32(i32 86, float %{{.*}})
entry:
  %dx.ddy.fine = call float @llvm.dx.ddy.fine.f32(float %a)
  ret float %dx.ddy.fine
}

define noundef <4 x float> @deriv_fine_y_float4(<4 x float> noundef %a) {
; CHECK: [[ee0:%.*]] = extractelement <4 x float> %a, i64 0
; CHECK: [[ie0:%.*]] = call float @dx.op.unary.f32(i32 86, float [[ee0]])
; CHECK: [[ee1:%.*]] = extractelement <4 x float> %a, i64 1
; CHECK: [[ie1:%.*]] = call float @dx.op.unary.f32(i32 86, float [[ee1]])
; CHECK: [[ee2:%.*]] = extractelement <4 x float> %a, i64 2
; CHECK: [[ie2:%.*]] = call float @dx.op.unary.f32(i32 86, float [[ee2]])
; CHECK: [[ee3:%.*]] = extractelement <4 x float> %a, i64 3
; CHECK: [[ie3:%.*]] = call float @dx.op.unary.f32(i32 86, float [[ee3]])
; CHECK: insertelement <4 x float> poison, float [[ie0]], i64 0
; CHECK: insertelement <4 x float> %{{.*}}, float [[ie1]], i64 1
; CHECK: insertelement <4 x float> %{{.*}}, float [[ie2]], i64 2
; CHECK: insertelement <4 x float> %{{.*}}, float [[ie3]], i64 3
; CHECK: ret <4 x float> %{{.*}}
entry:
  %dx.ddy.fine = call <4 x float> @llvm.dx.ddy.fine.v4f32(<4 x float> %a)
  ret <4 x float> %dx.ddy.fine
}

declare half @llvm.dx.ddy.fine.f16(half)
declare float @llvm.dx.ddy.fine.f32(float)
declare <4 x float> @llvm.dx.ddy.fine.v4f32(<4 x float>)
