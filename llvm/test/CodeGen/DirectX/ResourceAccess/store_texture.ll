; RUN: opt -S -dxil-resource-access %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

; CHECK-LABEL: define void @store_texture1d_float4
define void @store_texture1d_float4(<4 x float> %data, i32 %coord) {
  %texture = call target("dx.Texture", <4 x float>, 1, 0, 0, 1)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.Texture", <4 x float>, 1, 0, 0, 1) %texture, i32 %coord)

  ; CHECK: call void @llvm.dx.resource.store.texture.tdx.Texture_v4f32_1_0_0_1t.i32.v4f32(target("dx.Texture", <4 x float>, 1, 0, 0, 1) %texture, i32 %coord, <4 x float> %data)
  store <4 x float> %data, ptr %ptr

  ret void
}

; CHECK-LABEL: define void @store_texture2d_float4
define void @store_texture2d_float4(<4 x float> %data, <2 x i32> %coords, i32 %elemindex) {
  %texture = call target("dx.Texture", <4 x float>, 1, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.Texture", <4 x float>, 1, 0, 0, 2) %texture, <2 x i32> %coords)

  ; Store the whole value
  ; CHECK: call void @llvm.dx.resource.store.texture.tdx.Texture_v4f32_1_0_0_2t.v2i32.v4f32(target("dx.Texture", <4 x float>, 1, 0, 0, 2) %texture, <2 x i32> %coords, <4 x float> %data)
  store <4 x float> %data, ptr %ptr

  ; Store just the .x component
  %scalar = extractelement <4 x float> %data, i32 0
  ; CHECK: %[[LOAD:.*]] = call <4 x float> @llvm.dx.resource.load.level.v4f32.tdx.Texture_v4f32_1_0_0_2t.v2i32.i32.v2i32(target("dx.Texture", <4 x float>, 1, 0, 0, 2) %texture, <2 x i32> %coords, i32 0, <2 x i32> zeroinitializer)
  ; CHECK: %[[INSERT:.*]] = insertelement <4 x float> %[[LOAD]], float %scalar, i32 0
  ; CHECK: call void @llvm.dx.resource.store.texture.tdx.Texture_v4f32_1_0_0_2t.v2i32.v4f32(target("dx.Texture", <4 x float>, 1, 0, 0, 2) %texture, <2 x i32> %coords, <4 x float> %[[INSERT]])
  store float %scalar, ptr %ptr

  ; Store just the .y component
  ; CHECK: %[[LOAD:.*]] = call <4 x float> @llvm.dx.resource.load.level.v4f32.tdx.Texture_v4f32_1_0_0_2t.v2i32.i32.v2i32(target("dx.Texture", <4 x float>, 1, 0, 0, 2) %texture, <2 x i32> %coords, i32 0, <2 x i32> zeroinitializer)
  ; CHECK: %[[INSERT:.*]] = insertelement <4 x float> %[[LOAD]], float %scalar, i32 1
  ; CHECK: call void @llvm.dx.resource.store.texture.tdx.Texture_v4f32_1_0_0_2t.v2i32.v4f32(target("dx.Texture", <4 x float>, 1, 0, 0, 2) %texture, <2 x i32> %coords, <4 x float> %[[INSERT]])
  %y_ptr = getelementptr inbounds i8, ptr %ptr, i32 4
  store float %scalar, ptr %y_ptr

  ; Store to one of the elements dynamically
  ; CHECK: %[[LOAD:.*]] = call <4 x float> @llvm.dx.resource.load.level.v4f32.tdx.Texture_v4f32_1_0_0_2t.v2i32.i32.v2i32(target("dx.Texture", <4 x float>, 1, 0, 0, 2) %texture, <2 x i32> %coords, i32 0, <2 x i32> zeroinitializer)
  ; CHECK: %[[INSERT:.*]] = insertelement <4 x float> %[[LOAD]], float %scalar, i32 %elemindex
  ; CHECK: call void @llvm.dx.resource.store.texture.tdx.Texture_v4f32_1_0_0_2t.v2i32.v4f32(target("dx.Texture", <4 x float>, 1, 0, 0, 2) %texture, <2 x i32> %coords, <4 x float> %[[INSERT]])
  %dynamic = getelementptr inbounds float, ptr %ptr, i32 %elemindex
  store float %scalar, ptr %dynamic

  ret void
}

; CHECK-LABEL: define void @store_texture2d_float
define void @store_texture2d_float(float %data, <2 x i32> %coords) {
  %texture = call target("dx.Texture", float, 1, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 1, i32 1, i32 0, ptr null)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.Texture", float, 1, 0, 0, 2) %texture, <2 x i32> %coords)

  ; CHECK: call void @llvm.dx.resource.store.texture.tdx.Texture_f32_1_0_0_2t.v2i32.f32(target("dx.Texture", float, 1, 0, 0, 2) %texture, <2 x i32> %coords, float %data)
  store float %data, ptr %ptr

  ret void
}

; CHECK-LABEL: define void @store_texture2d_int3
define void @store_texture2d_int3(<3 x i32> %data, <2 x i32> %coords) {
  %texture = call target("dx.Texture", <3 x i32>, 1, 0, 1, 2)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 2, i32 1, i32 0, ptr null)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.Texture", <3 x i32>, 1, 0, 1, 2) %texture, <2 x i32> %coords)

  ; CHECK: call void @llvm.dx.resource.store.texture.tdx.Texture_v3i32_1_0_1_2t.v2i32.v3i32(target("dx.Texture", <3 x i32>, 1, 0, 1, 2) %texture, <2 x i32> %coords, <3 x i32> %data)
  store <3 x i32> %data, ptr %ptr

  ret void
}

; CHECK-LABEL: define void @store_texture1darray_float4
define void @store_texture1darray_float4(<4 x float> %data, <2 x i32> %coords) {
  %texture = call target("dx.Texture", <4 x float>, 1, 0, 0, 6)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 3, i32 1, i32 0, ptr null)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.Texture", <4 x float>, 1, 0, 0, 6) %texture, <2 x i32> %coords)

  ; CHECK: call void @llvm.dx.resource.store.texture.tdx.Texture_v4f32_1_0_0_6t.v2i32.v4f32(target("dx.Texture", <4 x float>, 1, 0, 0, 6) %texture, <2 x i32> %coords, <4 x float> %data)
  store <4 x float> %data, ptr %ptr

  ret void
}

; CHECK-LABEL: define void @store_texture2darray_float4
define void @store_texture2darray_float4(<4 x float> %data, <3 x i32> %coords) {
  %texture = call target("dx.Texture", <4 x float>, 1, 0, 0, 7)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 4, i32 1, i32 0, ptr null)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.Texture", <4 x float>, 1, 0, 0, 7) %texture, <3 x i32> %coords)

  ; CHECK: call void @llvm.dx.resource.store.texture.tdx.Texture_v4f32_1_0_0_7t.v3i32.v4f32(target("dx.Texture", <4 x float>, 1, 0, 0, 7) %texture, <3 x i32> %coords, <4 x float> %data)
  store <4 x float> %data, ptr %ptr

  ret void
}

; CHECK-LABEL: define void @store_texture3d_float3
define void @store_texture3d_float3(<3 x float> %data, <3 x i32> %coords) {
  %texture = call target("dx.Texture", <3 x float>, 1, 0, 0, 4)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 5, i32 1, i32 0, ptr null)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.Texture", <3 x float>, 1, 0, 0, 4) %texture, <3 x i32> %coords)

  ; CHECK: call void @llvm.dx.resource.store.texture.tdx.Texture_v3f32_1_0_0_4t.v3i32.v3f32(target("dx.Texture", <3 x float>, 1, 0, 0, 4) %texture, <3 x i32> %coords, <3 x float> %data)
  store <3 x float> %data, ptr %ptr

  ret void
}

; CHECK-LABEL: define void @store_texture2d_half4
define void @store_texture2d_half4(<4 x half> %data, <2 x i32> %coords) {
  %texture = call target("dx.Texture", <4 x half>, 1, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 6, i32 1, i32 0, ptr null)

  ; CHECK-NOT: @llvm.dx.resource.getpointer
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.Texture", <4 x half>, 1, 0, 0, 2) %texture, <2 x i32> %coords)

  ; CHECK: call void @llvm.dx.resource.store.texture.tdx.Texture_v4f16_1_0_0_2t.v2i32.v4f16(target("dx.Texture", <4 x half>, 1, 0, 0, 2) %texture, <2 x i32> %coords, <4 x half> %data)
  store <4 x half> %data, ptr %ptr

  ; Store just the .z component
  ; CHECK: %[[LOAD:.*]] = call <4 x half> @llvm.dx.resource.load.level.v4f16.tdx.Texture_v4f16_1_0_0_2t.v2i32.i32.v2i32(target("dx.Texture", <4 x half>, 1, 0, 0, 2) %texture, <2 x i32> %coords, i32 0, <2 x i32> zeroinitializer)
  ; CHECK: %[[INSERT:.*]] = insertelement <4 x half> %[[LOAD]], half %scalar, i32 2
  ; CHECK: call void @llvm.dx.resource.store.texture.tdx.Texture_v4f16_1_0_0_2t.v2i32.v4f16(target("dx.Texture", <4 x half>, 1, 0, 0, 2) %texture, <2 x i32> %coords, <4 x half> %[[INSERT]])
  %scalar = extractelement <4 x half> %data, i32 2
  %z_ptr = getelementptr inbounds i8, ptr %ptr, i32 4
  store half %scalar, ptr %z_ptr

  ret void
}
