; RUN: opt -S -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Make sure dxil operation function calls for round are generated for float and half.

; CHECK-LABEL: round_half
define noundef half @round_half(half noundef %a) {
entry:
; CHECK: call half @dx.op.unary.f16(i32 26, half %{{.*}}) #[[#ATTR:]]
  %elt.roundeven = call half @llvm.roundeven.f16(half %a)
  ret half %elt.roundeven
}

; CHECK-LABEL: round_float
define noundef float @round_float(float noundef %a) {
entry:
; CHECK: call float @dx.op.unary.f32(i32 26, float %{{.*}}) #[[#ATTR]]
  %elt.roundeven = call float @llvm.roundeven.f32(float %a)
  ret float %elt.roundeven
}

define noundef <4 x float> @round_float4(<4 x float> noundef %a) #0 {
entry:
  ; CHECK: [[ee0:%.*]] = extractelement <4 x float> %a, i64 0
  ; CHECK: [[ie0:%.*]] = call float @dx.op.unary.f32(i32 26, float [[ee0]]) #[[#ATTR]]
  ; CHECK: [[ee1:%.*]] = extractelement <4 x float> %a, i64 1
  ; CHECK: [[ie1:%.*]] = call float @dx.op.unary.f32(i32 26, float [[ee1]]) #[[#ATTR]]
  ; CHECK: [[ee2:%.*]] = extractelement <4 x float> %a, i64 2
  ; CHECK: [[ie2:%.*]] = call float @dx.op.unary.f32(i32 26, float [[ee2]]) #[[#ATTR]]
  ; CHECK: [[ee3:%.*]] = extractelement <4 x float> %a, i64 3
  ; CHECK: [[ie3:%.*]] = call float @dx.op.unary.f32(i32 26, float [[ee3]]) #[[#ATTR]]
  ; CHECK: insertelement <4 x float> poison, float [[ie0]], i64 0
  ; CHECK: insertelement <4 x float> %{{.*}}, float [[ie1]], i64 1
  ; CHECK: insertelement <4 x float> %{{.*}}, float [[ie2]], i64 2
  ; CHECK: insertelement <4 x float> %{{.*}}, float [[ie3]], i64 3
  %2 = call <4 x float> @llvm.roundeven.v4f32(<4 x float> %a) 
  ret <4 x float> %2
}

; CHECK: attributes #[[#ATTR]] = {{{.*}} memory(none) {{.*}}}

declare half @llvm.roundeven.f16(half)
declare float @llvm.roundeven.f32(float)
declare <4 x float> @llvm.roundeven.v4f32(<4 x float>)
