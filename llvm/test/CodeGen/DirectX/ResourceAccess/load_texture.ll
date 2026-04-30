; RUN: opt -S -dxil-resource-access %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

declare void @use_float4(<4 x float>)
declare void @use_float1(<1 x float>)
declare void @use_float(float)
declare void @use_int3(<3 x i32>)

; CHECK-LABEL: define void @load_texture2d_float4(
define void @load_texture2d_float4(<2 x i32> %coords) {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_v4f32_0_0_0_2t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.Texture", <4 x float>, 0, 0, 0, 2) %texture, <2 x i32> %coords)

  ; CHECK: %[[VALUE:.*]] = call <4 x float> @llvm.dx.resource.load.level.v4f32.tdx.Texture_v4f32_0_0_0_2t.v2i32.i32.v2i32(target("dx.Texture", <4 x float>, 0, 0, 0, 2) %texture, <2 x i32> %coords, i32 0, <2 x i32> zeroinitializer)
  %data = load <4 x float>, ptr %ptr
  call void @use_float4(<4 x float> %data)

  ret void
}

; CHECK-LABEL: define void @load_texture2d_float(
define void @load_texture2d_float(<2 x i32> %coords) {
  %texture = call target("dx.Texture", float, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_f32_0_0_0_2t(
          i32 0, i32 1, i32 1, i32 0, ptr null)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.Texture", float, 0, 0, 0, 2) %texture, <2 x i32> %coords)

  ; CHECK: %[[VALUE:.*]] = call float @llvm.dx.resource.load.level.f32.tdx.Texture_f32_0_0_0_2t.v2i32.i32.v2i32(target("dx.Texture", float, 0, 0, 0, 2) %texture, <2 x i32> %coords, i32 0, <2 x i32> zeroinitializer)
  %data = load float, ptr %ptr
  call void @use_float(float %data)

  ret void
}

; CHECK-LABEL: define void @load_texture2d_int3(
define void @load_texture2d_int3(<2 x i32> %coords) {
  %texture = call target("dx.Texture", <3 x i32>, 0, 0, 1, 2)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_v3i32_0_0_1_2t(
          i32 0, i32 2, i32 1, i32 0, ptr null)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.Texture", <3 x i32>, 0, 0, 1, 2) %texture, <2 x i32> %coords)

  ; CHECK: %[[VALUE:.*]] = call <3 x i32> @llvm.dx.resource.load.level.v3i32.tdx.Texture_v3i32_0_0_1_2t.v2i32.i32.v2i32(target("dx.Texture", <3 x i32>, 0, 0, 1, 2) %texture, <2 x i32> %coords, i32 0, <2 x i32> zeroinitializer)
  %data = load <3 x i32>, ptr %ptr
  call void @use_int3(<3 x i32> %data)

  ret void
}

; CHECK-LABEL: define void @load_texture2d_float_as_float1(
define void @load_texture2d_float_as_float1(<2 x i32> %coords) {
  %texture = call target("dx.Texture", float, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_f32_0_0_0_2t(
          i32 0, i32 1, i32 1, i32 0, ptr null)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.Texture", float, 0, 0, 0, 2) %texture, <2 x i32> %coords)

  ; CHECK: %[[VALUE:.*]] = call float @llvm.dx.resource.load.level.f32.tdx.Texture_f32_0_0_0_2t.v2i32.i32.v2i32(target("dx.Texture", float, 0, 0, 0, 2) %texture, <2 x i32> %coords, i32 0, <2 x i32> zeroinitializer)
  ; CHECK: %[[VEC:.*]] = insertelement <1 x float> poison, float %[[VALUE]], i32 0
  ; CHECK: call void @use_float1(<1 x float> %[[VEC]])
  %vec_data = load <1 x float>, ptr %ptr
  call void @use_float1(<1 x float> %vec_data)

  ret void
}

; CHECK-LABEL: define void @load_texture2d_float4_extract_element(
define void @load_texture2d_float4_extract_element(<2 x i32> %coords) {
  %texture = call target("dx.Texture", <4 x float>, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_v4f32_0_0_0_2t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.Texture", <4 x float>, 0, 0, 0, 2) %texture, <2 x i32> %coords)

  ; CHECK: %[[VALUE:.*]] = call <4 x float> @llvm.dx.resource.load.level.v4f32.tdx.Texture_v4f32_0_0_0_2t.v2i32.i32.v2i32(target("dx.Texture", <4 x float>, 0, 0, 0, 2) %texture, <2 x i32> %coords, i32 0, <2 x i32> zeroinitializer)
  ; CHECK: extractelement <4 x float> %[[VALUE]], i32 1
  %y_ptr = getelementptr inbounds <4 x float>, ptr %ptr, i32 0, i32 1
  %y_data = load float, ptr %y_ptr
  call void @use_float(float %y_data)

  ret void
}
