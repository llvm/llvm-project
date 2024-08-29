; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Make sure dxil operation function calls for sin are generated for float and half.
; CHECK:call float @dx.op.unary.f32(i32 13, float %{{.*}})
; CHECK:call half @dx.op.unary.f16(i32 13, half %{{.*}})

; Function Attrs: noinline nounwind optnone
define noundef float @sin_float(float noundef %a) #0 {
entry:
  %1 = call float @llvm.sin.f32(float %a)
  ret float %1
}

; Function Attrs: noinline nounwind optnone
define noundef half @sin_half(half noundef %a) #0 {
entry:
  %1 = call half @llvm.sin.f16(half %a)
  ret half %1
}

define noundef <4 x float> @sin_float4(<4 x float> noundef %a) #0 {
entry:
  %2 = call <4 x float> @llvm.sin.v4f32(<4 x float> %a) 
  ret <4 x float> %2
}
