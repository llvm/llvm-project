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

  ; CHECK: %[[COORD0:.*]] = extractelement <3 x float> %coords, i64 0
  ; CHECK: %[[COORD1:.*]] = extractelement <3 x float> %coords, i64 1
  ; CHECK: %[[COORD2:.*]] = extractelement <3 x float> %coords, i64 2
  ; CHECK: %[[DDX0:.*]] = extractelement <2 x float> %ddx, i64 0
  ; CHECK: %[[DDX1:.*]] = extractelement <2 x float> %ddx, i64 1
  ; CHECK: %[[DDY0:.*]] = extractelement <2 x float> %ddy, i64 0
  ; CHECK: %[[DDY1:.*]] = extractelement <2 x float> %ddy, i64 1
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
