; RUN: opt -S -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Make sure dxil operation function calls for fwidth are generated for float, half vec, float, an32float v
; Make sure dxil operation function calls for fwidth are generated for float, half vec, flv4oat, an32float vec


define noundef half @deriv_coarse_x_half(half noundef %a) {
; CHECK: call half @dx.op.unary.f16(i32 83, half %{{.*}})
entry:
  %dx.deriv.coarse.x = call half @llvm.dx.deriv.coarse.x.f16(half %a)
  ret half %dx.deriv.coarse.x
}

define noundef float @deriv_coarse_x_float(float noundef %a) {
; CHECK: call float @dx.op.unary.f32(i32 83, float %{{.*}})
entry:
  %dx.deriv.coarse.x = call float @llvm.dx.deriv.coarse.x.f32(float %a)
  ret float %dx.deriv.coarse.x
}

define noundef <4 x float> @deriv_coarse_x_float4(<4 x float> noundef %a) {
; CHECK: [[ee0:%.*]] = extractelement <4 x float> %a, i64 0
; CHECK: [[ie0:%.*]] = call float @dx.op.unary.f32(i32 83, float [[ee0]])
; CHECK: [[ee1:%.*]] = extractelement <4 x float> %a, i64 1
; CHECK: [[ie1:%.*]] = call float @dx.op.unary.f32(i32 83, float [[ee1]])
; CHECK: [[ee2:%.*]] = extractelement <4 x float> %a, i64 2
; CHECK: [[ie2:%.*]] = call float @dx.op.unary.f32(i32 83, float [[ee2]])
; CHECK: [[ee3:%.*]] = extractelement <4 x float> %a, i64 3
; CHECK: [[ie3:%.*]] = call float @dx.op.unary.f32(i32 83, float [[ee3]])
; CHECK: insertelement <4 x float> poison, float [[ie0]], i64 0
; CHECK: insertelement <4 x float> %{{.*}}, float [[ie1]], i64 1
; CHECK: insertelement <4 x float> %{{.*}}, float [[ie2]], i64 2
; CHECK: insertelement <4 x float> %{{.*}}, float [[ie3]], i64 3
; CHECK: ret <4 x float> %{{.*}}
entry:
  %dx.deriv.coarse.x = call <4 x float> @llvm.dx.deriv.coarse.x.v4f32(<4 x float> %a)
  ret <4 x float> %dx.deriv.coarse.x
}

declare half @llvm.dx.deriv.coarse.x.f16(half)
declare float @llvm.dx.deriv.coarse.x.f32(float)
declare <4 x float> @llvm.dx.deriv.coarse.x.v4f32(<4 x float>)

