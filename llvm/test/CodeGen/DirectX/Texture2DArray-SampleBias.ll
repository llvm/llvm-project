; RUN: opt -S -dxil-op-lower %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-pixel"

declare void @use_float4(<4 x float>)

; Test SampleBias on a Texture2DArray with float4 result.
; CHECK-LABEL: define void @samplebias_texture2darray_float4(
define void @samplebias_texture2darray_float4(<3 x float> %coords, float %bias) {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 7)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_v4f32_0_0_0_7t(
          i32 0, i32 0, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding.tdx.Sampler_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: %[[COORD0:.*]] = extractelement <3 x float> %coords, i32 0
  ; CHECK: %[[COORD1:.*]] = extractelement <3 x float> %coords, i32 1
  ; CHECK: %[[COORD2:.*]] = extractelement <3 x float> %coords, i32 2
  ; CHECK: %[[SAMPLE:.*]] = call %dx.types.ResRet.f32
  ; CHECK-SAME: @dx.op.sampleBias.f32(i32 61,
  ; CHECK-SAME: %dx.types.Handle %{{[^,]*}},
  ; CHECK-SAME: %dx.types.Handle %{{[^,]*}},
  ; CHECK-SAME: float %[[COORD0]], float %[[COORD1]], float %[[COORD2]], float undef,
  ; CHECK-SAME: i32 undef, i32 undef, i32 undef,
  ; CHECK-SAME: float %bias,
  ; CHECK-SAME: float undef)
  %data = call <4 x float>
      @llvm.dx.resource.samplebias.v4f32.tdx.Texture_v4f32_0_0_0_7t.tdx.Sampler_0t.v3f32.v2i32(
          target("dx.Texture", <4 x float>, 0, 0, 0, 7) %texture,
          target("dx.Sampler", 0) %sampler,
          <3 x float> %coords, float %bias, <2 x i32> zeroinitializer)

  ; CHECK: extractvalue %dx.types.ResRet.f32 %[[SAMPLE]], 0
  call void @use_float4(<4 x float> %data)
  ret void
}

; Test SampleBias with constant non-zero offsets on a Texture2DArray.
; CHECK-LABEL: define void @samplebias_texture2darray_with_offset(
define void @samplebias_texture2darray_with_offset(<3 x float> %coords, float %bias) {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 7)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_v4f32_0_0_0_7t(
          i32 0, i32 0, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding.tdx.Sampler_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: %[[COORD0:.*]] = extractelement <3 x float> %coords, i32 0
  ; CHECK: %[[COORD1:.*]] = extractelement <3 x float> %coords, i32 1
  ; CHECK: %[[COORD2:.*]] = extractelement <3 x float> %coords, i32 2
  ; CHECK: %[[SAMPLE:.*]] = call %dx.types.ResRet.f32
  ; CHECK-SAME: @dx.op.sampleBias.f32(i32 61,
  ; CHECK-SAME: %dx.types.Handle %{{[^,]*}},
  ; CHECK-SAME: %dx.types.Handle %{{[^,]*}},
  ; CHECK-SAME: float %[[COORD0]], float %[[COORD1]], float %[[COORD2]], float undef,
  ; CHECK-SAME: i32 1, i32 -2, i32 undef,
  ; CHECK-SAME: float %bias,
  ; CHECK-SAME: float undef)
  %data = call <4 x float>
      @llvm.dx.resource.samplebias.v4f32.tdx.Texture_v4f32_0_0_0_7t.tdx.Sampler_0t.v3f32.v2i32(
          target("dx.Texture", <4 x float>, 0, 0, 0, 7) %texture,
          target("dx.Sampler", 0) %sampler,
          <3 x float> %coords, float %bias, <2 x i32> <i32 1, i32 -2>)

  ; CHECK: extractvalue %dx.types.ResRet.f32 %[[SAMPLE]], 0
  call void @use_float4(<4 x float> %data)
  ret void
}

; Test SampleBias with clamp on a Texture2DArray.
; CHECK-LABEL: define void @samplebias_texture2darray_with_clamp(
define void @samplebias_texture2darray_with_clamp(<3 x float> %coords, float %bias, float %clamp) {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 7)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_v4f32_0_0_0_7t(
          i32 0, i32 0, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding.tdx.Sampler_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: %[[SAMPLE:.*]] = call %dx.types.ResRet.f32
  ; CHECK-SAME: @dx.op.sampleBias.f32(i32 61,
  ; CHECK-SAME: float %{{[^,]*}}, float %{{[^,]*}}, float %{{[^,]*}}, float undef,
  ; CHECK-SAME: i32 undef, i32 undef, i32 undef,
  ; CHECK-SAME: float %bias,
  ; CHECK-SAME: float %clamp)
  %data = call <4 x float>
      @llvm.dx.resource.samplebias.clamp.v4f32.tdx.Texture_v4f32_0_0_0_7t.tdx.Sampler_0t.v3f32.v2i32(
          target("dx.Texture", <4 x float>, 0, 0, 0, 7) %texture,
          target("dx.Sampler", 0) %sampler,
          <3 x float> %coords, float %bias, <2 x i32> zeroinitializer, float %clamp)

  ; CHECK: extractvalue %dx.types.ResRet.f32 %[[SAMPLE]], 0
  call void @use_float4(<4 x float> %data)
  ret void
}

; The scalarizer re-gathers the coordinates just to pass them to the intrinsic.
; DXIL has no vector instructions, so the scalars the vector was built from are
; forwarded into the sample rather than extracted again.
; CHECK-LABEL: define void @samplebias_texture2darray_scalarized(
define void @samplebias_texture2darray_scalarized(float %u, float %v, float %w,
                                                  float %bias) {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 7)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_v4f32_0_0_0_7t(
          i32 0, i32 0, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding.tdx.Sampler_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  %coords.0 = insertelement <3 x float> poison, float %u, i32 0
  %coords.1 = insertelement <3 x float> %coords.0, float %v, i32 1
  %coords.2 = insertelement <3 x float> %coords.1, float %w, i32 2

  ; CHECK-NOT: insertelement
  ; CHECK-NOT: extractelement
  ; CHECK: %[[SAMPLE:.*]] = call %dx.types.ResRet.f32
  ; CHECK-SAME: @dx.op.sampleBias.f32(i32 61,
  ; CHECK-SAME: %dx.types.Handle %{{[^,]*}},
  ; CHECK-SAME: %dx.types.Handle %{{[^,]*}},
  ; CHECK-SAME: float %u, float %v, float %w, float undef,
  ; CHECK-SAME: i32 undef, i32 undef, i32 undef,
  ; CHECK-SAME: float %bias,
  ; CHECK-SAME: float undef)
  %data = call <4 x float>
      @llvm.dx.resource.samplebias.v4f32.tdx.Texture_v4f32_0_0_0_7t.tdx.Sampler_0t.v3f32.v2i32(
          target("dx.Texture", <4 x float>, 0, 0, 0, 7) %texture,
          target("dx.Sampler", 0) %sampler,
          <3 x float> %coords.2, float %bias, <2 x i32> zeroinitializer)

  ; CHECK: extractvalue %dx.types.ResRet.f32 %[[SAMPLE]], 0
  call void @use_float4(<4 x float> %data)
  ret void
}
