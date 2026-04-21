; RUN: opt -S -dxil-op-lower %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

declare void @use_float4(<4 x float>)
declare void @use_float(float)
declare void @use_int3(<3 x i32>)

; CHECK-LABEL: define void @load_texture2d_float4(
define void @load_texture2d_float4(<2 x i32> %coords) {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_v4f32_0_0_0_2t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: %[[COORD0:.*]] = extractelement <2 x i32> %coords, i64 0
  ; CHECK: %[[COORD1:.*]] = extractelement <2 x i32> %coords, i64 1
  ; CHECK: %[[LOAD:.*]] = call %dx.types.ResRet.f32 @dx.op.textureLoad.f32(i32 66, %dx.types.Handle %{{.*}}, i32 0, i32 %[[COORD0]], i32 %[[COORD1]], i32 undef, i32 undef, i32 undef, i32 undef)
  %data = call <4 x float> @llvm.dx.resource.load.level.v4f32.tdx.Texture_v4f32_0_0_0_2t.v2i32.i32.v2i32(
      target("dx.Texture", <4 x float>, 0, 0, 0, 2) %texture,
      <2 x i32> %coords, i32 0, <2 x i32> zeroinitializer)

  ; CHECK: extractvalue %dx.types.ResRet.f32 %[[LOAD]], 0
  call void @use_float4(<4 x float> %data)
  ret void
}

; CHECK-LABEL: define void @load_texture2d_float(
define void @load_texture2d_float(<2 x i32> %coords) {
  %texture = call target("dx.Texture", float, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_f32_0_0_0_2t(
          i32 0, i32 1, i32 1, i32 0, ptr null)

  ; CHECK: %[[COORD0:.*]] = extractelement <2 x i32> %coords, i64 0
  ; CHECK: %[[COORD1:.*]] = extractelement <2 x i32> %coords, i64 1
  ; CHECK: %[[LOAD:.*]] = call %dx.types.ResRet.f32 @dx.op.textureLoad.f32(i32 66, %dx.types.Handle %{{.*}}, i32 0, i32 %[[COORD0]], i32 %[[COORD1]], i32 undef, i32 undef, i32 undef, i32 undef)
  %data = call float @llvm.dx.resource.load.level.f32.tdx.Texture_f32_0_0_0_2t.v2i32.i32.v2i32(
      target("dx.Texture", float, 0, 0, 0, 2) %texture,
      <2 x i32> %coords, i32 0, <2 x i32> zeroinitializer)

  ; CHECK: extractvalue %dx.types.ResRet.f32 %[[LOAD]], 0
  call void @use_float(float %data)
  ret void
}

; CHECK-LABEL: define void @load_texture2d_int3(
define void @load_texture2d_int3(<2 x i32> %coords) {
  %texture = call target("dx.Texture", <3 x i32>, 0, 0, 1, 2)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_v3i32_0_0_1_2t(
          i32 0, i32 2, i32 1, i32 0, ptr null)

  ; CHECK: %[[COORD0:.*]] = extractelement <2 x i32> %coords, i64 0
  ; CHECK: %[[COORD1:.*]] = extractelement <2 x i32> %coords, i64 1
  ; CHECK: %[[LOAD:.*]] = call %dx.types.ResRet.i32 @dx.op.textureLoad.i32(i32 66, %dx.types.Handle %{{.*}}, i32 0, i32 %[[COORD0]], i32 %[[COORD1]], i32 undef, i32 undef, i32 undef, i32 undef)
  %data = call <3 x i32> @llvm.dx.resource.load.level.v3i32.tdx.Texture_v3i32_0_0_1_2t.v2i32.i32.v2i32(
      target("dx.Texture", <3 x i32>, 0, 0, 1, 2) %texture,
      <2 x i32> %coords, i32 0, <2 x i32> zeroinitializer)

  call void @use_int3(<3 x i32> %data)
  ret void
}

declare <4 x float> @llvm.dx.resource.load.level.v4f32.tdx.Texture_v4f32_0_0_0_2t.v2i32.i32.v2i32(target("dx.Texture", <4 x float>, 0, 0, 0, 2), <2 x i32>, i32, <2 x i32>)
declare float @llvm.dx.resource.load.level.f32.tdx.Texture_f32_0_0_0_2t.v2i32.i32.v2i32(target("dx.Texture", float, 0, 0, 0, 2), <2 x i32>, i32, <2 x i32>)
declare <3 x i32> @llvm.dx.resource.load.level.v3i32.tdx.Texture_v3i32_0_0_1_2t.v2i32.i32.v2i32(target("dx.Texture", <3 x i32>, 0, 0, 1, 2), <2 x i32>, i32, <2 x i32>)

declare target("dx.Texture", <4 x float>, 0, 0, 0, 2) @llvm.dx.resource.handlefrombinding.tdx.Texture_v4f32_0_0_0_2t(i32, i32, i32, i32, ptr)
declare target("dx.Texture", float, 0, 0, 0, 2) @llvm.dx.resource.handlefrombinding.tdx.Texture_f32_0_0_0_2t(i32, i32, i32, i32, ptr)
declare target("dx.Texture", <3 x i32>, 0, 0, 1, 2) @llvm.dx.resource.handlefrombinding.tdx.Texture_v3i32_0_0_1_2t(i32, i32, i32, i32, ptr)
