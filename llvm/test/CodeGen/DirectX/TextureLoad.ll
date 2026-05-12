; RUN: opt -S -dxil-op-lower %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

declare void @use_float4(<4 x float>)
declare void @use_float(float)
declare void @use_int3(<3 x i32>)
declare void @use_float3(<3 x float>)

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

; CHECK-LABEL: define void @load_texture2d_float4_with_level(
define void @load_texture2d_float4_with_level(<2 x i32> %coords) {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_v4f32_0_0_0_2t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: %[[COORD0:.*]] = extractelement <2 x i32> %coords, i64 0
  ; CHECK: %[[COORD1:.*]] = extractelement <2 x i32> %coords, i64 1
  ; CHECK: %[[LOAD:.*]] = call %dx.types.ResRet.f32 @dx.op.textureLoad.f32(i32 66, %dx.types.Handle %{{.*}}, i32 2, i32 %[[COORD0]], i32 %[[COORD1]], i32 undef, i32 undef, i32 undef, i32 undef)
  %data = call <4 x float> @llvm.dx.resource.load.level.v4f32.tdx.Texture_v4f32_0_0_0_2t.v2i32.i32.v2i32(
      target("dx.Texture", <4 x float>, 0, 0, 0, 2) %texture,
      <2 x i32> %coords, i32 2, <2 x i32> zeroinitializer)

  ; CHECK: extractvalue %dx.types.ResRet.f32 %[[LOAD]], 0
  call void @use_float4(<4 x float> %data)
  ret void
}

; CHECK-LABEL: define void @load_texture2d_float4_with_offset(
define void @load_texture2d_float4_with_offset(<2 x i32> %coords) {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_v4f32_0_0_0_2t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: %[[COORD0:.*]] = extractelement <2 x i32> %coords, i64 0
  ; CHECK: %[[COORD1:.*]] = extractelement <2 x i32> %coords, i64 1
  ; CHECK: %[[LOAD:.*]] = call %dx.types.ResRet.f32 @dx.op.textureLoad.f32(i32 66, %dx.types.Handle %{{.*}}, i32 0, i32 %[[COORD0]], i32 %[[COORD1]], i32 undef, i32 1, i32 -2, i32 undef)
  %data = call <4 x float> @llvm.dx.resource.load.level.v4f32.tdx.Texture_v4f32_0_0_0_2t.v2i32.i32.v2i32(
      target("dx.Texture", <4 x float>, 0, 0, 0, 2) %texture,
      <2 x i32> %coords, i32 0, <2 x i32> <i32 1, i32 -2>)

  ; CHECK: extractvalue %dx.types.ResRet.f32 %[[LOAD]], 0
  call void @use_float4(<4 x float> %data)
  ret void
}

; CHECK-LABEL: define void @load_texture2d_float4_with_level_and_offset(
define void @load_texture2d_float4_with_level_and_offset(<2 x i32> %coords) {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_v4f32_0_0_0_2t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: %[[COORD0:.*]] = extractelement <2 x i32> %coords, i64 0
  ; CHECK: %[[COORD1:.*]] = extractelement <2 x i32> %coords, i64 1
  ; CHECK: %[[LOAD:.*]] = call %dx.types.ResRet.f32 @dx.op.textureLoad.f32(i32 66, %dx.types.Handle %{{.*}}, i32 3, i32 %[[COORD0]], i32 %[[COORD1]], i32 undef, i32 -1, i32 2, i32 undef)
  %data = call <4 x float> @llvm.dx.resource.load.level.v4f32.tdx.Texture_v4f32_0_0_0_2t.v2i32.i32.v2i32(
      target("dx.Texture", <4 x float>, 0, 0, 0, 2) %texture,
      <2 x i32> %coords, i32 3, <2 x i32> <i32 -1, i32 2>)

  ; CHECK: extractvalue %dx.types.ResRet.f32 %[[LOAD]], 0
  call void @use_float4(<4 x float> %data)
  ret void
}

; CHECK-LABEL: define void @load_texture1d_float4(
define void @load_texture1d_float4(i32 %coord) {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 1)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_v4f32_0_0_0_1t(
          i32 0, i32 3, i32 1, i32 0, ptr null)

  ; CHECK: %[[LOAD:.*]] = call %dx.types.ResRet.f32 @dx.op.textureLoad.f32(i32 66, %dx.types.Handle %{{.*}}, i32 0, i32 %coord, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef)
  %data = call <4 x float> @llvm.dx.resource.load.level.v4f32.tdx.Texture_v4f32_0_0_0_1t.i32.i32.i32(
      target("dx.Texture", <4 x float>, 0, 0, 0, 1) %texture,
      i32 %coord, i32 0, i32 0)

  ; CHECK: extractvalue %dx.types.ResRet.f32 %[[LOAD]], 0
  call void @use_float4(<4 x float> %data)
  ret void
}

; CHECK-LABEL: define void @load_texture1d_float4_with_level_and_offset(
define void @load_texture1d_float4_with_level_and_offset(i32 %coord) {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 1)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_v4f32_0_0_0_1t(
          i32 0, i32 3, i32 1, i32 0, ptr null)

  ; CHECK: %[[LOAD:.*]] = call %dx.types.ResRet.f32 @dx.op.textureLoad.f32(i32 66, %dx.types.Handle %{{.*}}, i32 4, i32 %coord, i32 undef, i32 undef, i32 5, i32 undef, i32 undef)
  %data = call <4 x float> @llvm.dx.resource.load.level.v4f32.tdx.Texture_v4f32_0_0_0_1t.i32.i32.i32(
      target("dx.Texture", <4 x float>, 0, 0, 0, 1) %texture,
      i32 %coord, i32 4, i32 5)

  ; CHECK: extractvalue %dx.types.ResRet.f32 %[[LOAD]], 0
  call void @use_float4(<4 x float> %data)
  ret void
}

; CHECK-LABEL: define void @load_texture3d_float3(
define void @load_texture3d_float3(<3 x i32> %coords) {
  %texture = call target("dx.Texture", <3 x float>, 0, 0, 0, 4)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_v3f32_0_0_0_4t(
          i32 0, i32 4, i32 1, i32 0, ptr null)

  ; CHECK: %[[COORD0:.*]] = extractelement <3 x i32> %coords, i64 0
  ; CHECK: %[[COORD1:.*]] = extractelement <3 x i32> %coords, i64 1
  ; CHECK: %[[COORD2:.*]] = extractelement <3 x i32> %coords, i64 2
  ; CHECK: %[[LOAD:.*]] = call %dx.types.ResRet.f32 @dx.op.textureLoad.f32(i32 66, %dx.types.Handle %{{.*}}, i32 0, i32 %[[COORD0]], i32 %[[COORD1]], i32 %[[COORD2]], i32 undef, i32 undef, i32 undef)
  %data = call <3 x float> @llvm.dx.resource.load.level.v3f32.tdx.Texture_v3f32_0_0_0_4t.v3i32.i32.v3i32(
      target("dx.Texture", <3 x float>, 0, 0, 0, 4) %texture,
      <3 x i32> %coords, i32 0, <3 x i32> zeroinitializer)

  ; CHECK: extractvalue %dx.types.ResRet.f32 %[[LOAD]], 0
  call void @use_float3(<3 x float> %data)
  ret void
}

; CHECK-LABEL: define void @load_texture3d_float3_with_level_and_offset(
define void @load_texture3d_float3_with_level_and_offset(<3 x i32> %coords) {
  %texture = call target("dx.Texture", <3 x float>, 0, 0, 0, 4)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_v3f32_0_0_0_4t(
          i32 0, i32 4, i32 1, i32 0, ptr null)

  ; CHECK: %[[COORD0:.*]] = extractelement <3 x i32> %coords, i64 0
  ; CHECK: %[[COORD1:.*]] = extractelement <3 x i32> %coords, i64 1
  ; CHECK: %[[COORD2:.*]] = extractelement <3 x i32> %coords, i64 2
  ; CHECK: %[[LOAD:.*]] = call %dx.types.ResRet.f32 @dx.op.textureLoad.f32(i32 66, %dx.types.Handle %{{.*}}, i32 2, i32 %[[COORD0]], i32 %[[COORD1]], i32 %[[COORD2]], i32 1, i32 -2, i32 3)
  %data = call <3 x float> @llvm.dx.resource.load.level.v3f32.tdx.Texture_v3f32_0_0_0_4t.v3i32.i32.v3i32(
      target("dx.Texture", <3 x float>, 0, 0, 0, 4) %texture,
      <3 x i32> %coords, i32 2, <3 x i32> <i32 1, i32 -2, i32 3>)

  ; CHECK: extractvalue %dx.types.ResRet.f32 %[[LOAD]], 0
  call void @use_float3(<3 x float> %data)
  ret void
}

; CHECK-LABEL: define void @load_texture2darray_float4(
define void @load_texture2darray_float4(<3 x i32> %coords) {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 7)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_v4f32_0_0_0_7t(
          i32 0, i32 5, i32 1, i32 0, ptr null)

  ; CHECK: %[[COORD0:.*]] = extractelement <3 x i32> %coords, i64 0
  ; CHECK: %[[COORD1:.*]] = extractelement <3 x i32> %coords, i64 1
  ; CHECK: %[[COORD2:.*]] = extractelement <3 x i32> %coords, i64 2
  ; CHECK: %[[LOAD:.*]] = call %dx.types.ResRet.f32 @dx.op.textureLoad.f32(i32 66, %dx.types.Handle %{{.*}}, i32 0, i32 %[[COORD0]], i32 %[[COORD1]], i32 %[[COORD2]], i32 undef, i32 undef, i32 undef)
  %data = call <4 x float> @llvm.dx.resource.load.level.v4f32.tdx.Texture_v4f32_0_0_0_7t.v3i32.i32.v2i32(
      target("dx.Texture", <4 x float>, 0, 0, 0, 7) %texture,
      <3 x i32> %coords, i32 0, <2 x i32> zeroinitializer)

  ; CHECK: extractvalue %dx.types.ResRet.f32 %[[LOAD]], 0
  call void @use_float4(<4 x float> %data)
  ret void
}

; CHECK-LABEL: define void @load_texture2darray_float4_with_level_and_offset(
define void @load_texture2darray_float4_with_level_and_offset(<3 x i32> %coords) {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 7)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_v4f32_0_0_0_7t(
          i32 0, i32 5, i32 1, i32 0, ptr null)

  ; CHECK: %[[COORD0:.*]] = extractelement <3 x i32> %coords, i64 0
  ; CHECK: %[[COORD1:.*]] = extractelement <3 x i32> %coords, i64 1
  ; CHECK: %[[COORD2:.*]] = extractelement <3 x i32> %coords, i64 2
  ; CHECK: %[[LOAD:.*]] = call %dx.types.ResRet.f32 @dx.op.textureLoad.f32(i32 66, %dx.types.Handle %{{.*}}, i32 1, i32 %[[COORD0]], i32 %[[COORD1]], i32 %[[COORD2]], i32 -1, i32 2, i32 undef)
  %data = call <4 x float> @llvm.dx.resource.load.level.v4f32.tdx.Texture_v4f32_0_0_0_7t.v3i32.i32.v2i32(
      target("dx.Texture", <4 x float>, 0, 0, 0, 7) %texture,
      <3 x i32> %coords, i32 1, <2 x i32> <i32 -1, i32 2>)

  ; CHECK: extractvalue %dx.types.ResRet.f32 %[[LOAD]], 0
  call void @use_float4(<4 x float> %data)
  ret void
}
