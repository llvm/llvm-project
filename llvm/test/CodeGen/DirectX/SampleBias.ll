; RUN: opt -S -dxil-op-lower %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-pixel"

declare void @use_float4(<4 x float>)

; CHECK-LABEL: define void @samplebias_texture2d_float4(
define void @samplebias_texture2d_float4(<2 x float> %coords, float %bias) {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_v4f32_0_0_0_2t(
          i32 0, i32 0, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding.tdx.Sampler_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: %[[COORD0:.*]] = extractelement <2 x float> %coords, i64 0
  ; CHECK: %[[COORD1:.*]] = extractelement <2 x float> %coords, i64 1
  ; CHECK: %[[SAMPLE:.*]] = call %dx.types.ResRet.f32 @dx.op.sampleBias.f32(i32 61, %dx.types.Handle %{{.*}}, %dx.types.Handle %{{.*}}, float %[[COORD0]], float %[[COORD1]], float undef, float undef, i32 undef, i32 undef, i32 undef, float %bias, float undef)
  %data = call <4 x float>
      @llvm.dx.resource.samplebias.v4f32.tdx.Texture_v4f32_0_0_0_2t.tdx.Sampler_0t.v2f32.v2i32(
          target("dx.Texture", <4 x float>, 0, 0, 0, 2) %texture,
          target("dx.Sampler", 0) %sampler,
          <2 x float> %coords, float %bias, <2 x i32> zeroinitializer)

  ; CHECK: extractvalue %dx.types.ResRet.f32 %[[SAMPLE]], 0
  call void @use_float4(<4 x float> %data)
  ret void
}

; CHECK-LABEL: define void @samplebias_texture2d_with_clamp(
define void @samplebias_texture2d_with_clamp(<2 x float> %coords, float %bias, float %clamp) {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_v4f32_0_0_0_2t(
          i32 0, i32 0, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding.tdx.Sampler_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: %[[COORD0:.*]] = extractelement <2 x float> %coords, i64 0
  ; CHECK: %[[COORD1:.*]] = extractelement <2 x float> %coords, i64 1
  ; CHECK: %[[SAMPLE:.*]] = call %dx.types.ResRet.f32 @dx.op.sampleBias.f32(i32 61, %dx.types.Handle %{{.*}}, %dx.types.Handle %{{.*}}, float %[[COORD0]], float %[[COORD1]], float undef, float undef, i32 undef, i32 undef, i32 undef, float %bias, float %clamp)
  %data = call <4 x float>
      @llvm.dx.resource.samplebias.clamp.v4f32.tdx.Texture_v4f32_0_0_0_2t.tdx.Sampler_0t.v2f32.v2i32(
          target("dx.Texture", <4 x float>, 0, 0, 0, 2) %texture,
          target("dx.Sampler", 0) %sampler,
          <2 x float> %coords, float %bias, <2 x i32> zeroinitializer, float %clamp)

  ; CHECK: extractvalue %dx.types.ResRet.f32 %[[SAMPLE]], 0
  call void @use_float4(<4 x float> %data)
  ret void
}
