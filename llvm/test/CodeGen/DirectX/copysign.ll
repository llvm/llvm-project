; RUN: opt -S -dxil-intrinsic-expansion -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Make sure the copysign intrinsic is expanded to bitwise operations.

; CHECK-LABEL: copysign_half
define noundef half @copysign_half(half noundef %a, half noundef %b) {
entry:
  ; CHECK: [[MAGNITUDE_INT:%.*]] = bitcast half %{{.*}} to i16
  ; CHECK: [[SIGN_INT:%.*]] = bitcast half %{{.*}} to i16
  ; CHECK: [[MAGNITUDE_BITS:%.*]] = and i16 [[MAGNITUDE_INT]], 32767
  ; CHECK: [[SIGN_BITS:%.*]] = and i16 [[SIGN_INT]], -32768
  ; CHECK: [[COMBINED_INT:%.*]] = or i16 [[MAGNITUDE_BITS]], [[SIGN_BITS]]
  ; CHECK: bitcast i16 [[COMBINED_INT]] to half
  %r = call half @llvm.copysign.f16(half %a, half %b)
  ret half %r
}

; CHECK-LABEL: copysign_float
define noundef float @copysign_float(float noundef %a, float noundef %b) {
entry:
  ; CHECK: [[MAGNITUDE_INT:%.*]] = bitcast float %{{.*}} to i32
  ; CHECK: [[SIGN_INT:%.*]] = bitcast float %{{.*}} to i32
  ; CHECK: [[MAGNITUDE_BITS:%.*]] = and i32 [[MAGNITUDE_INT]], 2147483647
  ; CHECK: [[SIGN_BITS:%.*]] = and i32 [[SIGN_INT]], -2147483648
  ; CHECK: [[COMBINED_INT:%.*]] = or i32 [[MAGNITUDE_BITS]], [[SIGN_BITS]]
  ; CHECK: bitcast i32 [[COMBINED_INT]] to float
  %r = call float @llvm.copysign.f32(float %a, float %b)
  ret float %r
}

; CHECK-LABEL: copysign_double
define noundef double @copysign_double(double noundef %a, double noundef %b) {
entry:
  ; CHECK: [[MAGNITUDE_HALVES:%.*]] = call %dx.types.splitdouble @dx.op.splitDouble.f64(i32 102, double %{{.*}})
  ; CHECK: [[SIGN_HALVES:%.*]] = call %dx.types.splitdouble @dx.op.splitDouble.f64(i32 102, double %{{.*}})
  ; CHECK: [[MAGNITUDE_LOW:%.*]] = extractvalue %dx.types.splitdouble [[MAGNITUDE_HALVES]], 0
  ; CHECK: [[MAGNITUDE_HIGH:%.*]] = extractvalue %dx.types.splitdouble [[MAGNITUDE_HALVES]], 1
  ; CHECK: [[SIGN_HIGH:%.*]] = extractvalue %dx.types.splitdouble [[SIGN_HALVES]], 1
  ; CHECK: [[MAGNITUDE_BITS:%.*]] = and i32 [[MAGNITUDE_HIGH]], 2147483647
  ; CHECK: [[SIGN_BITS:%.*]] = and i32 [[SIGN_HIGH]], -2147483648
  ; CHECK: [[COMBINED_HIGH:%.*]] = or i32 [[MAGNITUDE_BITS]], [[SIGN_BITS]]
  ; CHECK: call double @dx.op.makeDouble.f64(i32 101, i32 [[MAGNITUDE_LOW]], i32 [[COMBINED_HIGH]])
  %r = call double @llvm.copysign.f64(double %a, double %b)
  ret double %r
}

; CHECK-LABEL: copysign_float4
define noundef <4 x float> @copysign_float4(<4 x float> noundef %a, <4 x float> noundef %b) {
entry:
  ; CHECK: [[MAGNITUDE_INT:%.*]] = bitcast <4 x float> %{{.*}} to <4 x i32>
  ; CHECK: [[SIGN_INT:%.*]] = bitcast <4 x float> %{{.*}} to <4 x i32>
  ; CHECK: [[MAGNITUDE_BITS:%.*]] = and <4 x i32> [[MAGNITUDE_INT]], splat (i32 2147483647)
  ; CHECK: [[SIGN_BITS:%.*]] = and <4 x i32> [[SIGN_INT]], splat (i32 -2147483648)
  ; CHECK: [[COMBINED_INT:%.*]] = or <4 x i32> [[MAGNITUDE_BITS]], [[SIGN_BITS]]
  ; CHECK: bitcast <4 x i32> [[COMBINED_INT]] to <4 x float>
  %r = call <4 x float> @llvm.copysign.v4f32(<4 x float> %a, <4 x float> %b)
  ret <4 x float> %r
}

declare half @llvm.copysign.f16(half, half)
declare float @llvm.copysign.f32(float, float)
declare double @llvm.copysign.f64(double, double)
declare <4 x float> @llvm.copysign.v4f32(<4 x float>, <4 x float>)
