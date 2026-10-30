; RUN: opt -S -dxil-intrinsic-expansion -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.9-library %s | FileCheck %s --check-prefixes=CHECK,SM69CHECK
; RUN: opt -S -dxil-intrinsic-expansion -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.8-library %s | FileCheck %s --check-prefixes=CHECK,SMOLDCHECK

; Make sure dxil operation function calls for isfinite are generated for float and half.

define noundef i1 @isfinite_float(float noundef %a) {
entry:
  ; CHECK: call i1 @dx.op.isSpecialFloat.f32(i32 10, float %{{.*}})
  %dx.isfinite = call i1 @llvm.dx.isfinite.f32(float %a)
  ret i1 %dx.isfinite
}

define noundef i1 @isfinite_half(half noundef %a) {
entry:
  ; SM69CHECK: call i1 @dx.op.isSpecialFloat.f16(i32 10, half %{{.*}})
  ; SMOLDCHECK: [[BITCAST:%.*]] = bitcast half %a to i16
  ; SMOLDCHECK: [[AND:%.*]] = and i16 [[BITCAST]], 31744
  ; SMOLDCHECK: [[CMP:%.*]] = icmp ne i16 [[AND]], 31744
  %dx.isfinite = call i1 @llvm.dx.isfinite.f16(half %a)
  ret i1 %dx.isfinite
}

define noundef <4 x i1> @isfinite_half4(<4 x half> noundef %p0) {
entry:
  ; SM69CHECK: call i1 @dx.op.isSpecialFloat.f16(i32 10, half
  ; SM69CHECK: call i1 @dx.op.isSpecialFloat.f16(i32 10, half
  ; SM69CHECK: call i1 @dx.op.isSpecialFloat.f16(i32 10, half
  ; SM69CHECK: call i1 @dx.op.isSpecialFloat.f16(i32 10, half

  ; SMOLDCHECK: [[ee0:%.*]] = extractelement <4 x half> %p0, i64 0
  ; SMOLDCHECK: [[BITCAST0:%.*]] = bitcast half [[ee0]] to i16
  ; SMOLDCHECK: [[ee1:%.*]] = extractelement <4 x half> %p0, i64 1
  ; SMOLDCHECK: [[BITCAST1:%.*]] = bitcast half [[ee1]] to i16
  ; SMOLDCHECK: [[ee2:%.*]] = extractelement <4 x half> %p0, i64 2
  ; SMOLDCHECK: [[BITCAST2:%.*]] = bitcast half [[ee2]] to i16
  ; SMOLDCHECK: [[ee3:%.*]] = extractelement <4 x half> %p0, i64 3
  ; SMOLDCHECK: [[BITCAST3:%.*]] = bitcast half [[ee3]] to i16
  ; SMOLDCHECK: [[AND0:%.*]] = and i16 [[BITCAST0]], 31744
  ; SMOLDCHECK: [[AND1:%.*]] = and i16 [[BITCAST1]], 31744
  ; SMOLDCHECK: [[AND2:%.*]] = and i16 [[BITCAST2]], 31744
  ; SMOLDCHECK: [[AND3:%.*]] = and i16 [[BITCAST3]], 31744
  ; SMOLDCHECK: [[CMP0:%.*]] = icmp ne i16 [[AND0]], 31744
  ; SMOLDCHECK: [[CMP1:%.*]] = icmp ne i16 [[AND1]], 31744
  ; SMOLDCHECK: [[CMP2:%.*]] = icmp ne i16 [[AND2]], 31744
  ; SMOLDCHECK: [[CMP3:%.*]] = icmp ne i16 [[AND3]], 31744

  %hlsl.isfinite = call <4 x i1> @llvm.dx.isfinite.v4f16(<4 x half> %p0)
  ret <4 x i1> %hlsl.isfinite
}

define noundef <3 x i1> @isfinite_float3(<3 x float> noundef %p0) {
entry:
  ; CHECK: call i1 @dx.op.isSpecialFloat.f32(i32 10, float
  ; CHECK: call i1 @dx.op.isSpecialFloat.f32(i32 10, float
  ; CHECK: call i1 @dx.op.isSpecialFloat.f32(i32 10, float
  %hlsl.isfinite = call <3 x i1> @llvm.dx.isfinite.v3f32(<3 x float> %p0)
  ret <3 x i1> %hlsl.isfinite
}

; CHECK-DAG: declare i1 @dx.op.isSpecialFloat.f32(i32, float) #[[#ATTR0:]]
; SM69CHECK-DAG: declare i1 @dx.op.isSpecialFloat.f16(i32, half) #[[#ATTR0]]
; CHECK: attributes #[[#ATTR0]] = { nounwind memory(none) }
