; RUN: opt -S -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Make sure dxil operation function calls for sqrt are generated for float and half.

define noundef float @sqrt_float(float noundef %a) {
entry:
; CHECK:call float @dx.op.unary.f32(i32 24, float %{{.*}}) #[[#ATTR:]]
  %elt.sqrt = call float @llvm.sqrt.f32(float %a)
  ret float %elt.sqrt
}

define noundef half @sqrt_half(half noundef %a) {
entry:
; CHECK:call half @dx.op.unary.f16(i32 24, half %{{.*}}) #[[#ATTR]]
  %elt.sqrt = call half @llvm.sqrt.f16(half %a)
  ret half %elt.sqrt
}

define noundef <4 x float> @sqrt_float4(<4 x float> noundef %a) {
entry:
  ; CHECK: [[ee0:%.*]] = extractelement <4 x float> %a, i64 0
  ; CHECK: [[ie0:%.*]] = call float @dx.op.unary.f32(i32 24, float [[ee0]]) #[[#ATTR]]
  ; CHECK: [[ee1:%.*]] = extractelement <4 x float> %a, i64 1
  ; CHECK: [[ie1:%.*]] = call float @dx.op.unary.f32(i32 24, float [[ee1]]) #[[#ATTR]]
  ; CHECK: [[ee2:%.*]] = extractelement <4 x float> %a, i64 2
  ; CHECK: [[ie2:%.*]] = call float @dx.op.unary.f32(i32 24, float [[ee2]]) #[[#ATTR]]
  ; CHECK: [[ee3:%.*]] = extractelement <4 x float> %a, i64 3
  ; CHECK: [[ie3:%.*]] = call float @dx.op.unary.f32(i32 24, float [[ee3]]) #[[#ATTR]]
  ; CHECK: insertelement <4 x float> poison, float [[ie0]], i64 0
  ; CHECK: insertelement <4 x float> %{{.*}}, float [[ie1]], i64 1
  ; CHECK: insertelement <4 x float> %{{.*}}, float [[ie2]], i64 2
  ; CHECK: insertelement <4 x float> %{{.*}}, float [[ie3]], i64 3
  %2 = call <4 x float> @llvm.sqrt.v4f32(<4 x float> %a) 
  ret <4 x float> %2
}

; CHECK: attributes #[[#ATTR]] = {{{.*}} memory(none) {{.*}}}

declare half @llvm.sqrt.f16(half)
declare float @llvm.sqrt.f32(float)
declare <4 x float> @llvm.sqrt.v4f32(<4 x float>)
