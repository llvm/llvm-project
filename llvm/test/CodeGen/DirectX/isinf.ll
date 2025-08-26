; RUN: opt -S -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Make sure dxil operation function calls for isinf are generated for float and half.

define noundef i1 @isinf_float(float noundef %a) {
entry:
  ; CHECK: call i1 @dx.op.isSpecialFloat.f32(i32 9, float %{{.*}}) #[[#ATTR:]]
  %dx.isinf = call i1 @llvm.dx.isinf.f32(float %a)
  ret i1 %dx.isinf
}

define noundef i1 @isinf_half(half noundef %a) {
entry:
  ; CHECK: call i1 @dx.op.isSpecialFloat.f16(i32 9, half %{{.*}}) #[[#ATTR]]
  %dx.isinf = call i1 @llvm.dx.isinf.f16(half %a)
  ret i1 %dx.isinf
}

define noundef <4 x i1> @isinf_half4(<4 x half> noundef %p0) {
entry:
  ; CHECK: call i1 @dx.op.isSpecialFloat.f16(i32 9, half
  ; CHECK: call i1 @dx.op.isSpecialFloat.f16(i32 9, half
  ; CHECK: call i1 @dx.op.isSpecialFloat.f16(i32 9, half
  ; CHECK: call i1 @dx.op.isSpecialFloat.f16(i32 9, half
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
