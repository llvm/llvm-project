; RUN: opt -S -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Make sure dxil operation function calls for abs are generated for float, half, and double.


; CHECK-LABEL: fabs_half
define noundef half @fabs_half(half noundef %a) {
entry:
  ; CHECK: call half @dx.op.unary.f16(i32 6, half %{{.*}})
  %elt.abs = call half @llvm.fabs.f16(half %a)
  ret half %elt.abs
}

; CHECK-LABEL: fabs_float
define noundef float @fabs_float(float noundef %a) {
entry:
; CHECK: call float @dx.op.unary.f32(i32 6, float %{{.*}})
  %elt.abs = call float @llvm.fabs.f32(float %a)
  ret float %elt.abs
}

; CHECK-LABEL: fabs_double
define noundef double @fabs_double(double noundef %a) {
entry:
; CHECK: call double @dx.op.unary.f64(i32 6, double %{{.*}})
  %elt.abs = call double @llvm.fabs.f64(double %a)
  ret double %elt.abs
}

; CHECK-LABEL: fabs_float4
define noundef <4 x float> @fabs_float4(<4 x float> noundef %a) {
entry:
  ; CHECK: [[ee0:%.*]] = extractelement <4 x float> %a, i64 0
  ; CHECK: [[ie0:%.*]] = call float @dx.op.unary.f32(i32 6, float [[ee0]])
  ; CHECK: [[ee1:%.*]] = extractelement <4 x float> %a, i64 1
  ; CHECK: [[ie1:%.*]] = call float @dx.op.unary.f32(i32 6, float [[ee1]])
  ; CHECK: [[ee2:%.*]] = extractelement <4 x float> %a, i64 2
  ; CHECK: [[ie2:%.*]] = call float @dx.op.unary.f32(i32 6, float [[ee2]])
  ; CHECK: [[ee3:%.*]] = extractelement <4 x float> %a, i64 3
  ; CHECK: [[ie3:%.*]] = call float @dx.op.unary.f32(i32 6, float [[ee3]])
  ; CHECK: insertelement <4 x float> poison, float [[ie0]], i64 0
  ; CHECK: insertelement <4 x float> %{{.*}}, float [[ie1]], i64 1
  ; CHECK: insertelement <4 x float> %{{.*}}, float [[ie2]], i64 2
  ; CHECK: insertelement <4 x float> %{{.*}}, float [[ie3]], i64 3
  %2 = call <4 x float> @llvm.fabs.v4f32(<4 x float> %a) 
  ret <4 x float> %2
}

declare half @llvm.fabs.f16(half)
declare float @llvm.fabs.f32(float)
declare double @llvm.fabs.f64(double)
declare <4 x float> @llvm.fabs.v4f32(<4 x float>)
