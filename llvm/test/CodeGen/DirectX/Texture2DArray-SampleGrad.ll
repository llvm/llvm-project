; RUN: opt -S -dxil-op-lower %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-pixel"

declare void @use_float4(<4 x float>)

; Test basic SampleGrad on a Texture2DArray with float4 result.
; CHECK-LABEL: define void @samplegrad_texture2darray_float4(
define void @samplegrad_texture2darray_float4(<3 x float> %coords, <2 x float> %ddx, <2 x float> %ddy) {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 7)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_v4f32_0_0_0_7t(
          i32 0, i32 0, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding.tdx.Sampler_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: %[[COORD0:.*]] = extractelement <3 x float> %coords, i32 0
  ; CHECK: %[[COORD1:.*]] = extractelement <3 x float> %coords, i32 1
  ; CHECK: %[[COORD2:.*]] = extractelement <3 x float> %coords, i32 2
  ; CHECK: %[[DDX0:.*]] = extractelement <2 x float> %ddx, i32 0
  ; CHECK: %[[DDX1:.*]] = extractelement <2 x float> %ddx, i32 1
  ; CHECK: %[[DDY0:.*]] = extractelement <2 x float> %ddy, i32 0
  ; CHECK: %[[DDY1:.*]] = extractelement <2 x float> %ddy, i32 1
  ; CHECK: %[[SAMPLE:.*]] = call %dx.types.ResRet.f32
  ; CHECK-SAME: @dx.op.sampleGrad.f32(i32 63,
  ; CHECK-SAME: %dx.types.Handle %{{[^,]*}},
  ; CHECK-SAME: %dx.types.Handle %{{[^,]*}},
  ; CHECK-SAME: float %[[COORD0]], float %[[COORD1]], float %[[COORD2]], float undef,
  ; CHECK-SAME: i32 undef, i32 undef, i32 undef,
  ; CHECK-SAME: float %[[DDX0]], float %[[DDX1]], float undef,
  ; CHECK-SAME: float %[[DDY0]], float %[[DDY1]], float undef,
  ; CHECK-SAME: float undef)
  %data = call <4 x float>
      @llvm.dx.resource.samplegrad.v4f32.tdx.Texture_v4f32_0_0_0_7t.tdx.Sampler_0t.v3f32.v2f32.v2f32.v2i32(
          target("dx.Texture", <4 x float>, 0, 0, 0, 7) %texture,
          target("dx.Sampler", 0) %sampler,
          <3 x float> %coords, <2 x float> %ddx, <2 x float> %ddy,
          <2 x i32> zeroinitializer)

  ; CHECK: extractvalue %dx.types.ResRet.f32 %[[SAMPLE]], 0
  call void @use_float4(<4 x float> %data)
  ret void
}

; Test SampleGrad with constant non-zero offsets on a Texture2DArray.
; CHECK-LABEL: define void @samplegrad_texture2darray_with_offset(
define void @samplegrad_texture2darray_with_offset(<3 x float> %coords, <2 x float> %ddx, <2 x float> %ddy) {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 7)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_v4f32_0_0_0_7t(
          i32 0, i32 0, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding.tdx.Sampler_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: %[[SAMPLE:.*]] = call %dx.types.ResRet.f32
  ; CHECK-SAME: @dx.op.sampleGrad.f32(i32 63,
  ; CHECK-SAME: float %{{[^,]*}}, float %{{[^,]*}}, float %{{[^,]*}}, float undef,
  ; CHECK-SAME: i32 1, i32 -2, i32 undef,
  ; CHECK-SAME: float %{{[^,]*}}, float %{{[^,]*}}, float undef,
  ; CHECK-SAME: float %{{[^,]*}}, float %{{[^,]*}}, float undef,
  ; CHECK-SAME: float undef)
  %data = call <4 x float>
      @llvm.dx.resource.samplegrad.v4f32.tdx.Texture_v4f32_0_0_0_7t.tdx.Sampler_0t.v3f32.v2f32.v2f32.v2i32(
          target("dx.Texture", <4 x float>, 0, 0, 0, 7) %texture,
          target("dx.Sampler", 0) %sampler,
          <3 x float> %coords, <2 x float> %ddx, <2 x float> %ddy,
          <2 x i32> <i32 1, i32 -2>)

  ; CHECK: extractvalue %dx.types.ResRet.f32 %[[SAMPLE]], 0
  call void @use_float4(<4 x float> %data)
  ret void
}

; The scalarizer re-gathers the vector operands just to pass them to the
; intrinsic. DXIL has no vector instructions, so the scalars they were built
; from are forwarded into the sample rather than extracted again.
; CHECK-LABEL: define void @samplegrad_texture2darray_scalarized(
define void @samplegrad_texture2darray_scalarized(float %u, float %v, float %w,
                                                  float %ddxu, float %ddxv,
                                                  float %ddyu, float %ddyv) {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 7)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_v4f32_0_0_0_7t(
          i32 0, i32 0, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding.tdx.Sampler_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  %coords.0 = insertelement <3 x float> poison, float %u, i32 0
  %coords.1 = insertelement <3 x float> %coords.0, float %v, i32 1
  %coords.2 = insertelement <3 x float> %coords.1, float %w, i32 2
  %ddx.0 = insertelement <2 x float> poison, float %ddxu, i32 0
  %ddx.1 = insertelement <2 x float> %ddx.0, float %ddxv, i32 1
  %ddy.0 = insertelement <2 x float> poison, float %ddyu, i32 0
  %ddy.1 = insertelement <2 x float> %ddy.0, float %ddyv, i32 1

  ; CHECK-NOT: insertelement
  ; CHECK-NOT: extractelement
  ; CHECK: %[[SAMPLE:.*]] = call %dx.types.ResRet.f32
  ; CHECK-SAME: @dx.op.sampleGrad.f32(i32 63,
  ; CHECK-SAME: %dx.types.Handle %{{[^,]*}},
  ; CHECK-SAME: %dx.types.Handle %{{[^,]*}},
  ; CHECK-SAME: float %u, float %v, float %w, float undef,
  ; CHECK-SAME: i32 undef, i32 undef, i32 undef,
  ; CHECK-SAME: float %ddxu, float %ddxv, float undef,
  ; CHECK-SAME: float %ddyu, float %ddyv, float undef,
  ; CHECK-SAME: float undef)
  %data = call <4 x float>
      @llvm.dx.resource.samplegrad.v4f32.tdx.Texture_v4f32_0_0_0_7t.tdx.Sampler_0t.v3f32.v2f32.v2f32.v2i32(
          target("dx.Texture", <4 x float>, 0, 0, 0, 7) %texture,
          target("dx.Sampler", 0) %sampler,
          <3 x float> %coords.2, <2 x float> %ddx.1, <2 x float> %ddy.1,
          <2 x i32> zeroinitializer)

  ; CHECK: extractvalue %dx.types.ResRet.f32 %[[SAMPLE]], 0
  call void @use_float4(<4 x float> %data)
  ret void
}
