; RUN: opt -S -dxil-op-lower %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

; CHECK-LABEL: define void @store_texture1d_float4(
define void @store_texture1d_float4(<4 x float> %data, i32 %coord) {
  %texture = call target("dx.Texture", <4 x float>, 1, 0, 0, 1)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: %[[DATA0:.*]] = extractelement <4 x float> %data, i32 0
  ; CHECK: %[[DATA1:.*]] = extractelement <4 x float> %data, i32 1
  ; CHECK: %[[DATA2:.*]] = extractelement <4 x float> %data, i32 2
  ; CHECK: %[[DATA3:.*]] = extractelement <4 x float> %data, i32 3
  ; CHECK: call void @dx.op.textureStore.f32(i32 67, %dx.types.Handle %{{.*}}, i32 %coord, i32 undef, i32 undef, float %[[DATA0]], float %[[DATA1]], float %[[DATA2]], float %[[DATA3]], i8 15)
  call void @llvm.dx.resource.store.texture(
      target("dx.Texture", <4 x float>, 1, 0, 0, 1) %texture,
      i32 %coord, <4 x float> %data)

  ret void
}

; CHECK-LABEL: define void @store_texture2d_float4(
define void @store_texture2d_float4(<4 x float> %data, <2 x i32> %coords) {
  %texture = call target("dx.Texture", <4 x float>, 1, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 1, i32 1, i32 0, ptr null)

  ; CHECK: %[[DATA0:.*]] = extractelement <4 x float> %data, i32 0
  ; CHECK: %[[DATA1:.*]] = extractelement <4 x float> %data, i32 1
  ; CHECK: %[[DATA2:.*]] = extractelement <4 x float> %data, i32 2
  ; CHECK: %[[DATA3:.*]] = extractelement <4 x float> %data, i32 3
  ; CHECK: %[[COORD0:.*]] = extractelement <2 x i32> %coords, i64 0
  ; CHECK: %[[COORD1:.*]] = extractelement <2 x i32> %coords, i64 1
  ; CHECK: call void @dx.op.textureStore.f32(i32 67, %dx.types.Handle %{{.*}}, i32 %[[COORD0]], i32 %[[COORD1]], i32 undef, float %[[DATA0]], float %[[DATA1]], float %[[DATA2]], float %[[DATA3]], i8 15)
  call void @llvm.dx.resource.store.texture(
      target("dx.Texture", <4 x float>, 1, 0, 0, 2) %texture,
      <2 x i32> %coords, <4 x float> %data)

  ret void
}

; CHECK-LABEL: define void @store_texture3d_float4(
define void @store_texture3d_float4(<4 x float> %data, <3 x i32> %coords) {
  %texture = call target("dx.Texture", <4 x float>, 1, 0, 0, 4)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 2, i32 1, i32 0, ptr null)

  ; CHECK: %[[COORD0:.*]] = extractelement <3 x i32> %coords, i64 0
  ; CHECK: %[[COORD1:.*]] = extractelement <3 x i32> %coords, i64 1
  ; CHECK: %[[COORD2:.*]] = extractelement <3 x i32> %coords, i64 2
  ; CHECK: call void @dx.op.textureStore.f32(i32 67, %dx.types.Handle %{{.*}}, i32 %[[COORD0]], i32 %[[COORD1]], i32 %[[COORD2]], float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, i8 15)
  call void @llvm.dx.resource.store.texture(
      target("dx.Texture", <4 x float>, 1, 0, 0, 4) %texture,
      <3 x i32> %coords, <4 x float> %data)

  ret void
}

; CHECK-LABEL: define void @store_texture1darray_float4(
define void @store_texture1darray_float4(<4 x float> %data, <2 x i32> %coords) {
  %texture = call target("dx.Texture", <4 x float>, 1, 0, 0, 6)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 3, i32 1, i32 0, ptr null)

  ; CHECK: %[[COORD0:.*]] = extractelement <2 x i32> %coords, i64 0
  ; CHECK: %[[COORD1:.*]] = extractelement <2 x i32> %coords, i64 1
  ; CHECK: call void @dx.op.textureStore.f32(i32 67, %dx.types.Handle %{{.*}}, i32 %[[COORD0]], i32 %[[COORD1]], i32 undef, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, i8 15)
  call void @llvm.dx.resource.store.texture(
      target("dx.Texture", <4 x float>, 1, 0, 0, 6) %texture,
      <2 x i32> %coords, <4 x float> %data)

  ret void
}

; CHECK-LABEL: define void @store_texture2darray_float4(
define void @store_texture2darray_float4(<4 x float> %data, <3 x i32> %coords) {
  %texture = call target("dx.Texture", <4 x float>, 1, 0, 0, 7)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 4, i32 1, i32 0, ptr null)

  ; CHECK: %[[COORD0:.*]] = extractelement <3 x i32> %coords, i64 0
  ; CHECK: %[[COORD1:.*]] = extractelement <3 x i32> %coords, i64 1
  ; CHECK: %[[COORD2:.*]] = extractelement <3 x i32> %coords, i64 2
  ; CHECK: call void @dx.op.textureStore.f32(i32 67, %dx.types.Handle %{{.*}}, i32 %[[COORD0]], i32 %[[COORD1]], i32 %[[COORD2]], float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, i8 15)
  call void @llvm.dx.resource.store.texture(
      target("dx.Texture", <4 x float>, 1, 0, 0, 7) %texture,
      <3 x i32> %coords, <4 x float> %data)

  ret void
}

; A scalar texture still has to write all four components, so the value is
; repeated to fill out the store.
; CHECK-LABEL: define void @store_texture2d_float(
define void @store_texture2d_float(float %data, <2 x i32> %coords) {
  %texture = call target("dx.Texture", float, 1, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 5, i32 1, i32 0, ptr null)

  ; CHECK: %[[COORD0:.*]] = extractelement <2 x i32> %coords, i64 0
  ; CHECK: %[[COORD1:.*]] = extractelement <2 x i32> %coords, i64 1
  ; CHECK: call void @dx.op.textureStore.f32(i32 67, %dx.types.Handle %{{.*}}, i32 %[[COORD0]], i32 %[[COORD1]], i32 undef, float %data, float %data, float %data, float %data, i8 15)
  call void @llvm.dx.resource.store.texture(
      target("dx.Texture", float, 1, 0, 0, 2) %texture,
      <2 x i32> %coords, float %data)

  ret void
}

; A three component texture repeats the first element to fill out the store.
; CHECK-LABEL: define void @store_texture2d_int3(
define void @store_texture2d_int3(<3 x i32> %data, <2 x i32> %coords) {
  %texture = call target("dx.Texture", <3 x i32>, 1, 0, 1, 2)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 6, i32 1, i32 0, ptr null)

  ; CHECK: %[[DATA0:.*]] = extractelement <3 x i32> %data, i32 0
  ; CHECK: %[[DATA1:.*]] = extractelement <3 x i32> %data, i32 1
  ; CHECK: %[[DATA2:.*]] = extractelement <3 x i32> %data, i32 2
  ; CHECK: %[[COORD0:.*]] = extractelement <2 x i32> %coords, i64 0
  ; CHECK: %[[COORD1:.*]] = extractelement <2 x i32> %coords, i64 1
  ; CHECK: call void @dx.op.textureStore.i32(i32 67, %dx.types.Handle %{{.*}}, i32 %[[COORD0]], i32 %[[COORD1]], i32 undef, i32 %[[DATA0]], i32 %[[DATA1]], i32 %[[DATA2]], i32 %[[DATA0]], i8 15)
  call void @llvm.dx.resource.store.texture(
      target("dx.Texture", <3 x i32>, 1, 0, 1, 2) %texture,
      <2 x i32> %coords, <3 x i32> %data)

  ret void
}

; CHECK-LABEL: define void @store_texture2d_half4(
define void @store_texture2d_half4(<4 x half> %data, <2 x i32> %coords) {
  %texture = call target("dx.Texture", <4 x half>, 1, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 7, i32 1, i32 0, ptr null)

  ; CHECK: call void @dx.op.textureStore.f16(i32 67, %dx.types.Handle %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 undef, half %{{.*}}, half %{{.*}}, half %{{.*}}, half %{{.*}}, i8 15)
  call void @llvm.dx.resource.store.texture(
      target("dx.Texture", <4 x half>, 1, 0, 0, 2) %texture,
      <2 x i32> %coords, <4 x half> %data)

  ret void
}

; The scalarizer leaves behind an insertelement chain that we can forward
; directly into the store arguments.
; CHECK-LABEL: define void @store_texture2d_scalarized(
define void @store_texture2d_scalarized(float %x, float %y, float %z, float %w, <2 x i32> %coords) {
  %texture = call target("dx.Texture", <4 x float>, 1, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 8, i32 1, i32 0, ptr null)

  %vec.0 = insertelement <4 x float> poison, float %x, i32 0
  %vec.1 = insertelement <4 x float> %vec.0, float %y, i32 1
  %vec.2 = insertelement <4 x float> %vec.1, float %z, i32 2
  %vec.3 = insertelement <4 x float> %vec.2, float %w, i32 3

  ; CHECK-NOT: insertelement
  ; CHECK: call void @dx.op.textureStore.f32(i32 67, %dx.types.Handle %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 undef, float %x, float %y, float %z, float %w, i8 15)
  call void @llvm.dx.resource.store.texture(
      target("dx.Texture", <4 x float>, 1, 0, 0, 2) %texture,
      <2 x i32> %coords, <4 x float> %vec.3)

  ret void
}
