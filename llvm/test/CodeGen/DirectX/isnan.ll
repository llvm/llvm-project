; RUN: opt -S -dxil-intrinsic-expansion -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.9-library %s | FileCheck %s --check-prefixes=CHECK,SM69CHECK
; RUN: opt -S -dxil-intrinsic-expansion -mtriple=dxil-pc-shadermodel6.8-library %s | FileCheck %s --check-prefixes=CHECK,SMOLDCHECK

; Make sure dxil operation function calls for isnan are generated for float and half.

define noundef i1 @isnan_float(float noundef %a) {
entry:
  ; SM69CHECK: call i1 @dx.op.isSpecialFloat.f32(i32 8, float %{{.*}}) #[[#ATTR:]]
  ; SMOLDCHECK: call i1 @llvm.dx.isnan.f32(float %{{.*}})
  %dx.isnan = call i1 @llvm.dx.isnan.f32(float %a)
  ret i1 %dx.isnan
}

define noundef i1 @isnan_half(half noundef %a) {
entry:
  ; SM69CHECK: call i1 @dx.op.isSpecialFloat.f16(i32 8, half %{{.*}}) #[[#ATTR]]
  ; SMOLDCHECK: [[BITCAST:%.*]] = bitcast half %{{.*}} to i16
  ; SMOLDCHECK: [[ANDHIGH:%.*]] = and i16 [[BITCAST]], 31744
  ; SMOLDCHECK: [[CMPHIGH:%.*]] = icmp eq i16 [[ANDHIGH]], 31744
  ; SMOLDCHECK: [[ANDLOW:%.*]] = and i16 [[BITCAST]], 1023
  ; SMOLDCHECK: [[CMPLOW:%.*]] = icmp ne i16 [[ANDLOW]], 0
  ; SMOLDCHECK: [[AND:%.*]] = and i1 [[CMPHIGH]], [[CMPLOW]]
  %dx.isnan = call i1 @llvm.dx.isnan.f16(half %a)
  ret i1 %dx.isnan
}

define noundef <4 x i1> @isnan_half4(<4 x half> noundef %p0) {
entry:
  ; SM69CHECK: call i1 @dx.op.isSpecialFloat.f16(i32 8, half
  ; SM69CHECK: call i1 @dx.op.isSpecialFloat.f16(i32 8, half
  ; SM69CHECK: call i1 @dx.op.isSpecialFloat.f16(i32 8, half
  ; SM69CHECK: call i1 @dx.op.isSpecialFloat.f16(i32 8, half
  ; SMOLDCHECK: [[BITCAST:%.*]] = bitcast <4 x half> %{{.*}} to <4 x i16>
  ; SMOLDCHECK: [[ANDHIGH:%.*]] = and <4 x i16> [[BITCAST]], splat (i16 31744)
  ; SMOLDCHECK: [[CMPHIGH:%.*]] = icmp eq <4 x i16> [[ANDHIGH]], splat (i16 31744)
  ; SMOLDCHECK: [[ANDLOW:%.*]] = and <4 x i16> [[BITCAST]], splat (i16 1023)
  ; SMOLDCHECK: [[CMPLOW:%.*]] = icmp ne <4 x i16> [[ANDLOW]], zeroinitializer
  ; SMOLDCHECK: [[AND:%.*]] = and <4 x i1> [[CMPHIGH]], [[CMPLOW]]
  %hlsl.isnan = call <4 x i1> @llvm.dx.isnan.v4f16(<4 x half> %p0)
  ret <4 x i1> %hlsl.isnan
}

define noundef <3 x i1> @isnan_float3(<3 x float> noundef %p0) {
entry:
  ; SM69CHECK: call i1 @dx.op.isSpecialFloat.f32(i32 8, float
  ; SM69CHECK: call i1 @dx.op.isSpecialFloat.f32(i32 8, float
  ; SM69CHECK: call i1 @dx.op.isSpecialFloat.f32(i32 8, float
  ; SMOLDCHECK: = call <3 x i1> @llvm.dx.isnan.v3f32(<3 x float>
  %hlsl.isnan = call <3 x i1> @llvm.dx.isnan.v3f32(<3 x float> %p0)
  ret <3 x i1> %hlsl.isnan
}

; CHECK: attributes #{{[0-9]*}} = {{{.*}} memory(none) {{.*}}}
