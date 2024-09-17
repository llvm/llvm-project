; RUN: opt -S -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Make sure dxil operation function calls for sinh are generated for float and half.

define noundef float @sinh_float(float noundef %a) {
entry:
; CHECK:call float @dx.op.unary.f32(i32 19, float %{{.*}})
  %elt.sinh = call float @llvm.sinh.f32(float %a)
  ret float %elt.sinh
}

define noundef half @sinh_half(half noundef %a) {
entry:
; CHECK:call half @dx.op.unary.f16(i32 19, half %{{.*}})
  %elt.sinh = call half @llvm.sinh.f16(half %a)
  ret half %elt.sinh
}

define noundef <4 x float> @sinh_float4(<4 x float> noundef %a) {
entry:
  ; CHECK: [[ee0:%.*]] = extractelement <4 x float> %a, i64 0
  ; CHECK: [[ie0:%.*]] = call float @dx.op.unary.f32(i32 19, float [[ee0]])
  ; CHECK: [[ee1:%.*]] = extractelement <4 x float> %a, i64 1
  ; CHECK: [[ie1:%.*]] = call float @dx.op.unary.f32(i32 19, float [[ee1]])
  ; CHECK: [[ee2:%.*]] = extractelement <4 x float> %a, i64 2
  ; CHECK: [[ie2:%.*]] = call float @dx.op.unary.f32(i32 19, float [[ee2]])
  ; CHECK: [[ee3:%.*]] = extractelement <4 x float> %a, i64 3
  ; CHECK: [[ie3:%.*]] = call float @dx.op.unary.f32(i32 19, float [[ee3]])
  ; CHECK: insertelement <4 x float> poison, float [[ie0]], i64 0
  ; CHECK: insertelement <4 x float> %{{.*}}, float [[ie1]], i64 1
  ; CHECK: insertelement <4 x float> %{{.*}}, float [[ie2]], i64 2
  ; CHECK: insertelement <4 x float> %{{.*}}, float [[ie3]], i64 3
  %2 = call <4 x float> @llvm.sinh.v4f32(<4 x float> %a) 
  ret <4 x float> %2
}

declare half @llvm.sinh.f16(half)
declare float @llvm.sinh.f32(float)
declare <4 x float> @llvm.sinh.v4f32(<4 x float>)
