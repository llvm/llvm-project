; RUN: opt -S -dxil-intrinsic-expansion -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.9-library %s | FileCheck %s --check-prefixes=CHECK,SM69CHECK
; RUN: opt -S -dxil-intrinsic-expansion -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.8-library %s | FileCheck %s --check-prefixes=CHECK,SMOLDCHECK

; Make sure dxil operation function calls for isinf are generated for float and half.

define noundef i1 @isinf_float(float noundef %a) {
entry:
  ; CHECK: call i1 @dx.op.isSpecialFloat.f32(i32 9, float %{{.*}}) #[[#ATTR:]]
  %dx.isinf = call i1 @llvm.dx.isinf.f32(float %a)
  ret i1 %dx.isinf
}

define noundef i1 @isinf_half(half noundef %a) {
entry:
  ; SM69CHECK: call i1 @dx.op.isSpecialFloat.f16(i32 9, half %{{.*}}) #[[#ATTR]]
  ; SMOLDCHECK: [[BITCAST:%.*]] = bitcast half %a to i16
  ; SMOLDCHECK: [[CMPHIGH:%.*]] = icmp eq i16 [[BITCAST]], 31744
  ; SMOLDCHECK: [[CMPLOW:%.*]] = icmp eq i16 [[BITCAST]], -1024
  ; SMOLDCHECK: [[OR:%.*]] = or i1 [[CMPHIGH]], [[CMPLOW]]
  %dx.isinf = call i1 @llvm.dx.isinf.f16(half %a)
  ret i1 %dx.isinf
}

define noundef <4 x i1> @isinf_half4(<4 x half> noundef %p0) {
entry:
  ; SM69CHECK: call i1 @dx.op.isSpecialFloat.f16(i32 9, half
  ; SM69CHECK: call i1 @dx.op.isSpecialFloat.f16(i32 9, half
  ; SM69CHECK: call i1 @dx.op.isSpecialFloat.f16(i32 9, half
  ; SM69CHECK: call i1 @dx.op.isSpecialFloat.f16(i32 9, half

  ; SMOLDCHECK: [[ee0:%.*]] = extractelement <4 x half> %p0, i64 0
  ; SMOLDCHECK: [[BITCAST0:%.*]] = bitcast half [[ee0]] to i16
  ; SMOLDCHECK: [[ee1:%.*]] = extractelement <4 x half> %p0, i64 1
  ; SMOLDCHECK: [[BITCAST1:%.*]] = bitcast half [[ee1]] to i16
  ; SMOLDCHECK:[[ee2:%.*]] = extractelement <4 x half> %p0, i64 2
  ; SMOLDCHECK: [[BITCAST2:%.*]] = bitcast half [[ee2]] to i16
  ; SMOLDCHECK: [[ee3:%.*]] = extractelement <4 x half> %p0, i64 3
  ; SMOLDCHECK: [[BITCAST3:%.*]] = bitcast half [[ee3]] to i16
  ; SMOLDCHECK: [[ICMPHIGH0:%.*]] = icmp eq i16 [[BITCAST0]], 31744
  ; SMOLDCHECK: [[ICMPHIGH1:%.*]] = icmp eq i16 [[BITCAST1]], 31744
  ; SMOLDCHECK: [[ICMPHIGH2:%.*]] = icmp eq i16 [[BITCAST2]], 31744
  ; SMOLDCHECK: [[ICMPHIGH3:%.*]] = icmp eq i16 [[BITCAST3]], 31744
  ; SMOLDCHECK: [[ICMPLOW0:%.*]] = icmp eq i16 [[BITCAST0]], -1024
  ; SMOLDCHECK: [[ICMPLOW1:%.*]] = icmp eq i16 [[BITCAST1]], -1024
  ; SMOLDCHECK: [[ICMPLOW2:%.*]] = icmp eq i16 [[BITCAST2]], -1024
  ; SMOLDCHECK: [[ICMPLOW3:%.*]] = icmp eq i16 [[BITCAST3]], -1024
  ; SMOLDCHECK: [[OR0:%.*]] = or i1 [[ICMPHIGH0]], [[ICMPLOW0]]
  ; SMOLDCHECK: [[OR1:%.*]] = or i1 [[ICMPHIGH1]], [[ICMPLOW1]]
  ; SMOLDCHECK: [[OR2:%.*]] = or i1 [[ICMPHIGH2]], [[ICMPLOW2]]
  ; SMOLDCHECK: [[OR3:%.*]] = or i1 [[ICMPHIGH3]], [[ICMPLOW3]]
  ; SMOLDCHECK: %.upto019 = insertelement <4 x i1> poison, i1 [[OR0]], i64 0
  ; SMOLDCHECK: %.upto120 = insertelement <4 x i1> %.upto019, i1 [[OR1]], i64 1
  ; SMOLDCHECK: %.upto221 = insertelement <4 x i1> %.upto120, i1 [[OR2]], i64 2
  ; SMOLDCHECK: %0 = insertelement <4 x i1> %.upto221, i1 [[OR3]], i64 3

  %hlsl.isinf = call <4 x i1> @llvm.dx.isinf.v4f16(<4 x half> %p0)
  ret <4 x i1> %hlsl.isinf
}

define noundef <3 x i1> @isinf_float3(<3 x float> noundef %p0) {
entry:
  ; CHECK: call i1 @dx.op.isSpecialFloat.f32(i32 9, float
  ; CHECK: call i1 @dx.op.isSpecialFloat.f32(i32 9, float
  ; CHECK: call i1 @dx.op.isSpecialFloat.f32(i32 9, float
  %hlsl.isinf = call <3 x i1> @llvm.dx.isinf.v3f32(<3 x float> %p0)
  ret <3 x i1> %hlsl.isinf
}

; CHECK: attributes #[[#ATTR]] = {{{.*}} memory(none) {{.*}}}
