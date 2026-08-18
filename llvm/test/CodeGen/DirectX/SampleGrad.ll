; RUN: opt -S -dxil-op-lower %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-pixel"

declare void @use_float4(<4 x float>)
declare void @use_float(float)
declare void @use_half4(<4 x half>)

; Test basic SampleGrad on a Texture2D with float4 result.
; CHECK-LABEL: define void @samplegrad_texture2d_float4(
define void @samplegrad_texture2d_float4(<2 x float> %coords, <2 x float> %ddx, <2 x float> %ddy) {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_v4f32_0_0_0_2t(
          i32 0, i32 0, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding.tdx.Sampler_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: %[[COORD0:.*]] = extractelement <2 x float> %coords, i64 0
  ; CHECK: %[[COORD1:.*]] = extractelement <2 x float> %coords, i64 1
  ; CHECK: %[[DDX0:.*]] = extractelement <2 x float> %ddx, i64 0
  ; CHECK: %[[DDX1:.*]] = extractelement <2 x float> %ddx, i64 1
  ; CHECK: %[[DDY0:.*]] = extractelement <2 x float> %ddy, i64 0
  ; CHECK: %[[DDY1:.*]] = extractelement <2 x float> %ddy, i64 1
  ; CHECK: %[[SAMPLE:.*]] = call %dx.types.ResRet.f32
  ; CHECK-SAME: @dx.op.sampleGrad.f32(i32 63,
  ; CHECK-SAME: %dx.types.Handle %{{[^,]*}},
  ; CHECK-SAME: %dx.types.Handle %{{[^,]*}},
  ; CHECK-SAME: float %[[COORD0]], float %[[COORD1]], float undef, float undef,
  ; CHECK-SAME: i32 undef, i32 undef, i32 undef,
  ; CHECK-SAME: float %[[DDX0]], float %[[DDX1]], float undef,
  ; CHECK-SAME: float %[[DDY0]], float %[[DDY1]], float undef,
  ; CHECK-SAME: float undef)
  %data = call <4 x float>
      @llvm.dx.resource.samplegrad.v4f32.tdx.Texture_v4f32_0_0_0_2t.tdx.Sampler_0t.v2f32.v2f32.v2f32.v2i32(
          target("dx.Texture", <4 x float>, 0, 0, 0, 2) %texture,
          target("dx.Sampler", 0) %sampler,
          <2 x float> %coords, <2 x float> %ddx, <2 x float> %ddy,
          <2 x i32> zeroinitializer)

  ; CHECK: extractvalue %dx.types.ResRet.f32 %[[SAMPLE]], 0
  call void @use_float4(<4 x float> %data)
  ret void
}

; Test SampleGrad with clamp on a Texture2D.
; CHECK-LABEL: define void @samplegrad_texture2d_with_clamp(
define void @samplegrad_texture2d_with_clamp(<2 x float> %coords, <2 x float> %ddx, <2 x float> %ddy, float %clamp) {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_v4f32_0_0_0_2t(
          i32 0, i32 0, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding.tdx.Sampler_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: %[[COORD0:.*]] = extractelement <2 x float> %coords, i64 0
  ; CHECK: %[[COORD1:.*]] = extractelement <2 x float> %coords, i64 1
  ; CHECK: %[[DDX0:.*]] = extractelement <2 x float> %ddx, i64 0
  ; CHECK: %[[DDX1:.*]] = extractelement <2 x float> %ddx, i64 1
  ; CHECK: %[[DDY0:.*]] = extractelement <2 x float> %ddy, i64 0
  ; CHECK: %[[DDY1:.*]] = extractelement <2 x float> %ddy, i64 1
  ; CHECK: %[[SAMPLE:.*]] = call %dx.types.ResRet.f32
  ; CHECK-SAME: @dx.op.sampleGrad.f32(i32 63,
  ; CHECK-SAME: %dx.types.Handle %{{[^,]*}},
  ; CHECK-SAME: %dx.types.Handle %{{[^,]*}},
  ; CHECK-SAME: float %[[COORD0]], float %[[COORD1]], float undef, float undef,
  ; CHECK-SAME: i32 undef, i32 undef, i32 undef,
  ; CHECK-SAME: float %[[DDX0]], float %[[DDX1]], float undef,
  ; CHECK-SAME: float %[[DDY0]], float %[[DDY1]], float undef,
  ; CHECK-SAME: float %clamp)
  %data = call <4 x float>
      @llvm.dx.resource.samplegrad.clamp.v4f32.tdx.Texture_v4f32_0_0_0_2t.tdx.Sampler_0t.v2f32.v2f32.v2f32.v2i32(
          target("dx.Texture", <4 x float>, 0, 0, 0, 2) %texture,
          target("dx.Sampler", 0) %sampler,
          <2 x float> %coords, <2 x float> %ddx, <2 x float> %ddy,
          <2 x i32> zeroinitializer, float %clamp)

  ; CHECK: extractvalue %dx.types.ResRet.f32 %[[SAMPLE]], 0
  call void @use_float4(<4 x float> %data)
  ret void
}

; Test SampleGrad with constant non-zero offsets on a Texture2D.
; CHECK-LABEL: define void @samplegrad_texture2d_with_offset(
define void @samplegrad_texture2d_with_offset(<2 x float> %coords, <2 x float> %ddx, <2 x float> %ddy) {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_v4f32_0_0_0_2t(
          i32 0, i32 0, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding.tdx.Sampler_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: %[[COORD0:.*]] = extractelement <2 x float> %coords, i64 0
  ; CHECK: %[[COORD1:.*]] = extractelement <2 x float> %coords, i64 1
  ; CHECK: %[[DDX0:.*]] = extractelement <2 x float> %ddx, i64 0
  ; CHECK: %[[DDX1:.*]] = extractelement <2 x float> %ddx, i64 1
  ; CHECK: %[[DDY0:.*]] = extractelement <2 x float> %ddy, i64 0
  ; CHECK: %[[DDY1:.*]] = extractelement <2 x float> %ddy, i64 1
  ; CHECK: %[[SAMPLE:.*]] = call %dx.types.ResRet.f32
  ; CHECK-SAME: @dx.op.sampleGrad.f32(i32 63,
  ; CHECK-SAME: %dx.types.Handle %{{[^,]*}},
  ; CHECK-SAME: %dx.types.Handle %{{[^,]*}},
  ; CHECK-SAME: float %[[COORD0]], float %[[COORD1]], float undef, float undef,
  ; CHECK-SAME: i32 1, i32 -2, i32 undef,
  ; CHECK-SAME: float %[[DDX0]], float %[[DDX1]], float undef,
  ; CHECK-SAME: float %[[DDY0]], float %[[DDY1]], float undef,
  ; CHECK-SAME: float undef)
  %data = call <4 x float>
      @llvm.dx.resource.samplegrad.v4f32.tdx.Texture_v4f32_0_0_0_2t.tdx.Sampler_0t.v2f32.v2f32.v2f32.v2i32(
          target("dx.Texture", <4 x float>, 0, 0, 0, 2) %texture,
          target("dx.Sampler", 0) %sampler,
          <2 x float> %coords, <2 x float> %ddx, <2 x float> %ddy,
          <2 x i32> <i32 1, i32 -2>)

  ; CHECK: extractvalue %dx.types.ResRet.f32 %[[SAMPLE]], 0
  call void @use_float4(<4 x float> %data)
  ret void
}

; Test SampleGrad with dynamic (non-constant) offsets on a Texture2D.
; CHECK-LABEL: define void @samplegrad_texture2d_with_dynamic_offset(
define void @samplegrad_texture2d_with_dynamic_offset(<2 x float> %coords, <2 x float> %ddx, <2 x float> %ddy, <2 x i32> %offsets) {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_v4f32_0_0_0_2t(
          i32 0, i32 0, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding.tdx.Sampler_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: %[[COORD0:.*]] = extractelement <2 x float> %coords, i64 0
  ; CHECK: %[[COORD1:.*]] = extractelement <2 x float> %coords, i64 1
  ; CHECK: %[[OFF0:.*]] = extractelement <2 x i32> %offsets, i64 0
  ; CHECK: %[[OFF1:.*]] = extractelement <2 x i32> %offsets, i64 1
  ; CHECK: %[[DDX0:.*]] = extractelement <2 x float> %ddx, i64 0
  ; CHECK: %[[DDX1:.*]] = extractelement <2 x float> %ddx, i64 1
  ; CHECK: %[[DDY0:.*]] = extractelement <2 x float> %ddy, i64 0
  ; CHECK: %[[DDY1:.*]] = extractelement <2 x float> %ddy, i64 1
  ; CHECK: %[[SAMPLE:.*]] = call %dx.types.ResRet.f32
  ; CHECK-SAME: @dx.op.sampleGrad.f32(i32 63,
  ; CHECK-SAME: %dx.types.Handle %{{[^,]*}},
  ; CHECK-SAME: %dx.types.Handle %{{[^,]*}},
  ; CHECK-SAME: float %[[COORD0]], float %[[COORD1]], float undef, float undef,
  ; CHECK-SAME: i32 %[[OFF0]], i32 %[[OFF1]], i32 undef,
  ; CHECK-SAME: float %[[DDX0]], float %[[DDX1]], float undef,
  ; CHECK-SAME: float %[[DDY0]], float %[[DDY1]], float undef,
  ; CHECK-SAME: float undef)
  %data = call <4 x float>
      @llvm.dx.resource.samplegrad.v4f32.tdx.Texture_v4f32_0_0_0_2t.tdx.Sampler_0t.v2f32.v2f32.v2f32.v2i32(
          target("dx.Texture", <4 x float>, 0, 0, 0, 2) %texture,
          target("dx.Sampler", 0) %sampler,
          <2 x float> %coords, <2 x float> %ddx, <2 x float> %ddy,
          <2 x i32> %offsets)

  ; CHECK: extractvalue %dx.types.ResRet.f32 %[[SAMPLE]], 0
  call void @use_float4(<4 x float> %data)
  ret void
}

; Test SampleGrad with both offset and clamp on a Texture2D.
; CHECK-LABEL: define void @samplegrad_texture2d_with_offset_and_clamp(
define void @samplegrad_texture2d_with_offset_and_clamp(<2 x float> %coords, <2 x float> %ddx, <2 x float> %ddy, float %clamp) {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_v4f32_0_0_0_2t(
          i32 0, i32 0, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding.tdx.Sampler_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: %[[COORD0:.*]] = extractelement <2 x float> %coords, i64 0
  ; CHECK: %[[COORD1:.*]] = extractelement <2 x float> %coords, i64 1
  ; CHECK: %[[DDX0:.*]] = extractelement <2 x float> %ddx, i64 0
  ; CHECK: %[[DDX1:.*]] = extractelement <2 x float> %ddx, i64 1
  ; CHECK: %[[DDY0:.*]] = extractelement <2 x float> %ddy, i64 0
  ; CHECK: %[[DDY1:.*]] = extractelement <2 x float> %ddy, i64 1
  ; CHECK: %[[SAMPLE:.*]] = call %dx.types.ResRet.f32
  ; CHECK-SAME: @dx.op.sampleGrad.f32(i32 63,
  ; CHECK-SAME: %dx.types.Handle %{{[^,]*}},
  ; CHECK-SAME: %dx.types.Handle %{{[^,]*}},
  ; CHECK-SAME: float %[[COORD0]], float %[[COORD1]], float undef, float undef,
  ; CHECK-SAME: i32 3, i32 -1, i32 undef,
  ; CHECK-SAME: float %[[DDX0]], float %[[DDX1]], float undef,
  ; CHECK-SAME: float %[[DDY0]], float %[[DDY1]], float undef,
  ; CHECK-SAME: float %clamp)
  %data = call <4 x float>
      @llvm.dx.resource.samplegrad.clamp.v4f32.tdx.Texture_v4f32_0_0_0_2t.tdx.Sampler_0t.v2f32.v2f32.v2f32.v2i32(
          target("dx.Texture", <4 x float>, 0, 0, 0, 2) %texture,
          target("dx.Sampler", 0) %sampler,
          <2 x float> %coords, <2 x float> %ddx, <2 x float> %ddy,
          <2 x i32> <i32 3, i32 -1>, float %clamp)

  ; CHECK: extractvalue %dx.types.ResRet.f32 %[[SAMPLE]], 0
  call void @use_float4(<4 x float> %data)
  ret void
}

; Test SampleGrad on a Texture1D (scalar coordinate/gradient, scalar offset).
; CHECK-LABEL: define void @samplegrad_texture1d_float4(
define void @samplegrad_texture1d_float4(float %coord, float %ddx_scalar, float %ddy_scalar) {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 1)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_v4f32_0_0_0_1t(
          i32 0, i32 3, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding.tdx.Sampler_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: %[[SAMPLE:.*]] = call %dx.types.ResRet.f32
  ; CHECK-SAME: @dx.op.sampleGrad.f32(i32 63,
  ; CHECK-SAME: %dx.types.Handle %{{[^,]*}},
  ; CHECK-SAME: %dx.types.Handle %{{[^,]*}},
  ; CHECK-SAME: float %coord, float undef, float undef, float undef,
  ; CHECK-SAME: i32 undef, i32 undef, i32 undef,
  ; CHECK-SAME: float %ddx_scalar, float undef, float undef,
  ; CHECK-SAME: float %ddy_scalar, float undef, float undef,
  ; CHECK-SAME: float undef)
  %data = call <4 x float>
      @llvm.dx.resource.samplegrad.v4f32.tdx.Texture_v4f32_0_0_0_1t.tdx.Sampler_0t.f32.f32.f32.i32(
          target("dx.Texture", <4 x float>, 0, 0, 0, 1) %texture,
          target("dx.Sampler", 0) %sampler,
          float %coord, float %ddx_scalar, float %ddy_scalar, i32 0)

  ; CHECK: extractvalue %dx.types.ResRet.f32 %[[SAMPLE]], 0
  call void @use_float4(<4 x float> %data)
  ret void
}

; Test SampleGrad on a Texture3D (3-component coordinates/gradients).
; CHECK-LABEL: define void @samplegrad_texture3d_float4(
define void @samplegrad_texture3d_float4(<3 x float> %coords, <3 x float> %ddx, <3 x float> %ddy) {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 4)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_v4f32_0_0_0_4t(
          i32 0, i32 4, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding.tdx.Sampler_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: %[[COORD0:.*]] = extractelement <3 x float> %coords, i64 0
  ; CHECK: %[[COORD1:.*]] = extractelement <3 x float> %coords, i64 1
  ; CHECK: %[[COORD2:.*]] = extractelement <3 x float> %coords, i64 2
  ; CHECK: %[[DDX0:.*]] = extractelement <3 x float> %ddx, i64 0
  ; CHECK: %[[DDX1:.*]] = extractelement <3 x float> %ddx, i64 1
  ; CHECK: %[[DDX2:.*]] = extractelement <3 x float> %ddx, i64 2
  ; CHECK: %[[DDY0:.*]] = extractelement <3 x float> %ddy, i64 0
  ; CHECK: %[[DDY1:.*]] = extractelement <3 x float> %ddy, i64 1
  ; CHECK: %[[DDY2:.*]] = extractelement <3 x float> %ddy, i64 2
  ; CHECK: %[[SAMPLE:.*]] = call %dx.types.ResRet.f32
  ; CHECK-SAME: @dx.op.sampleGrad.f32(i32 63,
  ; CHECK-SAME: %dx.types.Handle %{{[^,]*}},
  ; CHECK-SAME: %dx.types.Handle %{{[^,]*}},
  ; CHECK-SAME: float %[[COORD0]], float %[[COORD1]], float %[[COORD2]], float undef,
  ; CHECK-SAME: i32 undef, i32 undef, i32 undef,
  ; CHECK-SAME: float %[[DDX0]], float %[[DDX1]], float %[[DDX2]],
  ; CHECK-SAME: float %[[DDY0]], float %[[DDY1]], float %[[DDY2]],
  ; CHECK-SAME: float undef)
  %data = call <4 x float>
      @llvm.dx.resource.samplegrad.v4f32.tdx.Texture_v4f32_0_0_0_4t.tdx.Sampler_0t.v3f32.v3f32.v3f32.v3i32(
          target("dx.Texture", <4 x float>, 0, 0, 0, 4) %texture,
          target("dx.Sampler", 0) %sampler,
          <3 x float> %coords, <3 x float> %ddx, <3 x float> %ddy,
          <3 x i32> zeroinitializer)

  ; CHECK: extractvalue %dx.types.ResRet.f32 %[[SAMPLE]], 0
  call void @use_float4(<4 x float> %data)
  ret void
}

; Test SampleGrad with a scalar return type on a Texture2D.
; CHECK-LABEL: define void @samplegrad_texture2d_scalar(
define void @samplegrad_texture2d_scalar(<2 x float> %coords, <2 x float> %ddx, <2 x float> %ddy) {
  %texture = call target("dx.Texture", float, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_f32_0_0_0_2t(
          i32 0, i32 1, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding.tdx.Sampler_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: %[[COORD0:.*]] = extractelement <2 x float> %coords, i64 0
  ; CHECK: %[[COORD1:.*]] = extractelement <2 x float> %coords, i64 1
  ; CHECK: %[[DDX0:.*]] = extractelement <2 x float> %ddx, i64 0
  ; CHECK: %[[DDX1:.*]] = extractelement <2 x float> %ddx, i64 1
  ; CHECK: %[[DDY0:.*]] = extractelement <2 x float> %ddy, i64 0
  ; CHECK: %[[DDY1:.*]] = extractelement <2 x float> %ddy, i64 1
  ; CHECK: %[[SAMPLE:.*]] = call %dx.types.ResRet.f32
  ; CHECK-SAME: @dx.op.sampleGrad.f32(i32 63,
  ; CHECK-SAME: %dx.types.Handle %{{[^,]*}},
  ; CHECK-SAME: %dx.types.Handle %{{[^,]*}},
  ; CHECK-SAME: float %[[COORD0]], float %[[COORD1]], float undef, float undef,
  ; CHECK-SAME: i32 undef, i32 undef, i32 undef,
  ; CHECK-SAME: float %[[DDX0]], float %[[DDX1]], float undef,
  ; CHECK-SAME: float %[[DDY0]], float %[[DDY1]], float undef,
  ; CHECK-SAME: float undef)
  %data = call float
      @llvm.dx.resource.samplegrad.f32.tdx.Texture_f32_0_0_0_2t.tdx.Sampler_0t.v2f32.v2f32.v2f32.v2i32(
          target("dx.Texture", float, 0, 0, 0, 2) %texture,
          target("dx.Sampler", 0) %sampler,
          <2 x float> %coords, <2 x float> %ddx, <2 x float> %ddy,
          <2 x i32> zeroinitializer)

  ; CHECK: extractvalue %dx.types.ResRet.f32 %[[SAMPLE]], 0
  call void @use_float(float %data)
  ret void
}

; Test SampleGrad with half-precision result type on a Texture2D.
; CHECK-LABEL: define void @samplegrad_texture2d_half4(
define void @samplegrad_texture2d_half4(<2 x float> %coords, <2 x float> %ddx, <2 x float> %ddy) {
  %texture = call target("dx.Texture", <4 x half>, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_v4f16_0_0_0_2t(
          i32 0, i32 5, i32 1, i32 0, ptr null)
  %sampler = call target("dx.Sampler", 0)
      @llvm.dx.resource.handlefrombinding.tdx.Sampler_0t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: %[[COORD0:.*]] = extractelement <2 x float> %coords, i64 0
  ; CHECK: %[[COORD1:.*]] = extractelement <2 x float> %coords, i64 1
  ; CHECK: %[[DDX0:.*]] = extractelement <2 x float> %ddx, i64 0
  ; CHECK: %[[DDX1:.*]] = extractelement <2 x float> %ddx, i64 1
  ; CHECK: %[[DDY0:.*]] = extractelement <2 x float> %ddy, i64 0
  ; CHECK: %[[DDY1:.*]] = extractelement <2 x float> %ddy, i64 1
  ; CHECK: %[[SAMPLE:.*]] = call %dx.types.ResRet.f16
  ; CHECK-SAME: @dx.op.sampleGrad.f16(i32 63,
  ; CHECK-SAME: %dx.types.Handle %{{[^,]*}},
  ; CHECK-SAME: %dx.types.Handle %{{[^,]*}},
  ; CHECK-SAME: float %[[COORD0]], float %[[COORD1]], float undef, float undef,
  ; CHECK-SAME: i32 undef, i32 undef, i32 undef,
  ; CHECK-SAME: float %[[DDX0]], float %[[DDX1]], float undef,
  ; CHECK-SAME: float %[[DDY0]], float %[[DDY1]], float undef,
  ; CHECK-SAME: float undef)
  %data = call <4 x half>
      @llvm.dx.resource.samplegrad.v4f16.tdx.Texture_v4f16_0_0_0_2t.tdx.Sampler_0t.v2f32.v2f32.v2f32.v2i32(
          target("dx.Texture", <4 x half>, 0, 0, 0, 2) %texture,
          target("dx.Sampler", 0) %sampler,
          <2 x float> %coords, <2 x float> %ddx, <2 x float> %ddy,
          <2 x i32> zeroinitializer)

  ; CHECK: extractvalue %dx.types.ResRet.f16 %[[SAMPLE]], 0
  call void @use_half4(<4 x half> %data)
  ret void
}
