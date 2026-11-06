; RUN: opt -S -dxil-intrinsic-expansion < %s | FileCheck %s

; DXIL has no 64-bit TextureStore overload, so 64-bit texture stores are split
; into stores of interleaved low and high i32 halves.

target triple = "dxil-pc-shadermodel6.6-compute"

; CHECK-LABEL: define void @store_rwtexture2d_i64(
define void @store_rwtexture2d_i64(<2 x i32> %coords, i64 %data) {
  %texture = call target("dx.Texture", i64, 1, 0, 1, 2)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_i64_1_0_1_2t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK-NOT: call void @llvm.dx.resource.store.texture.{{.*}}.i64(
  ; CHECK: [[Lo:%.*]] = trunc i64 %data to i32
  ; CHECK: [[Sh:%.*]] = lshr i64 %data, 32
  ; CHECK: [[Hi:%.*]] = trunc i64 [[Sh]] to i32
  ; CHECK: [[Vec0:%.*]] = insertelement <2 x i32> poison, i32 [[Lo]], i32 0
  ; CHECK: [[Vec1:%.*]] = insertelement <2 x i32> [[Vec0]], i32 [[Hi]], i32 1
  ; CHECK: call void @llvm.dx.resource.store.texture.tdx.Texture_i64_1_0_1_2t.v2i32.v2i32(
  ; CHECK-SAME: target("dx.Texture", i64, 1, 0, 1, 2) %texture,
  ; CHECK-SAME: <2 x i32> %coords, <2 x i32> [[Vec1]])
  call void @llvm.dx.resource.store.texture(
      target("dx.Texture", i64, 1, 0, 1, 2) %texture,
      <2 x i32> %coords, i64 %data)

  ; CHECK-NEXT: ret void
  ret void
}

; A RWTexture2DArray has the array slice as its third coordinate.
; CHECK-LABEL: define void @store_rwtexture2darray_v2i64(
define void @store_rwtexture2darray_v2i64(<3 x i32> %coords, <2 x i64> %data) {
  %texture = call target("dx.Texture", <2 x i64>, 1, 0, 0, 7)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_v2i64_1_0_0_7t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: [[Lo:%.*]] = trunc <2 x i64> %data to <2 x i32>
  ; CHECK: [[Sh:%.*]] = lshr <2 x i64> %data, splat (i64 32)
  ; CHECK: [[Hi:%.*]] = trunc <2 x i64> [[Sh]] to <2 x i32>
  ; CHECK: [[Vec:%.*]] = shufflevector <2 x i32> [[Lo]], <2 x i32> [[Hi]], <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  ; CHECK: call void @llvm.dx.resource.store.texture.tdx.Texture_v2i64_1_0_0_7t.v3i32.v4i32(
  ; CHECK-SAME: target("dx.Texture", <2 x i64>, 1, 0, 0, 7) %texture,
  ; CHECK-SAME: <3 x i32> %coords, <4 x i32> [[Vec]])
  call void @llvm.dx.resource.store.texture(
      target("dx.Texture", <2 x i64>, 1, 0, 0, 7) %texture,
      <3 x i32> %coords, <2 x i64> %data)

  ; CHECK-NEXT: ret void
  ret void
}

; A RWTexture1D has a scalar coordinate.
; CHECK-LABEL: define void @store_rwtexture1d_double(
define void @store_rwtexture1d_double(i32 %coord, double %data) {
  %texture = call target("dx.Texture", double, 1, 0, 0, 1)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_f64_1_0_0_1t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: [[SD:%.*]] = call { i32, i32 } @llvm.dx.splitdouble.i32(double %data)
  ; CHECK: [[Lo:%.*]] = extractvalue { i32, i32 } [[SD]], 0
  ; CHECK: [[Hi:%.*]] = extractvalue { i32, i32 } [[SD]], 1
  ; CHECK: [[Vec0:%.*]] = insertelement <2 x i32> poison, i32 [[Lo]], i32 0
  ; CHECK: [[Vec1:%.*]] = insertelement <2 x i32> [[Vec0]], i32 [[Hi]], i32 1
  ; CHECK: call void @llvm.dx.resource.store.texture.tdx.Texture_f64_1_0_0_1t.i32.v2i32(
  ; CHECK-SAME: target("dx.Texture", double, 1, 0, 0, 1) %texture,
  ; CHECK-SAME: i32 %coord, <2 x i32> [[Vec1]])
  call void @llvm.dx.resource.store.texture(
      target("dx.Texture", double, 1, 0, 0, 1) %texture,
      i32 %coord, double %data)

  ; CHECK-NEXT: ret void
  ret void
}

; CHECK-LABEL: define void @store_rwtexture2d_v2double(
define void @store_rwtexture2d_v2double(<2 x i32> %coords, <2 x double> %data) {
  %texture = call target("dx.Texture", <2 x double>, 1, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_v2f64_1_0_0_2t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: [[SD:%.*]] = call { <2 x i32>, <2 x i32> } @llvm.dx.splitdouble.v2i32(<2 x double> %data)
  ; CHECK: [[Lo:%.*]] = extractvalue { <2 x i32>, <2 x i32> } [[SD]], 0
  ; CHECK: [[Hi:%.*]] = extractvalue { <2 x i32>, <2 x i32> } [[SD]], 1
  ; CHECK: [[Vec:%.*]] = shufflevector <2 x i32> [[Lo]], <2 x i32> [[Hi]], <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  ; CHECK: call void @llvm.dx.resource.store.texture.tdx.Texture_v2f64_1_0_0_2t.v2i32.v4i32(
  ; CHECK-SAME: target("dx.Texture", <2 x double>, 1, 0, 0, 2) %texture,
  ; CHECK-SAME: <2 x i32> %coords, <4 x i32> [[Vec]])
  call void @llvm.dx.resource.store.texture(
      target("dx.Texture", <2 x double>, 1, 0, 0, 2) %texture,
      <2 x i32> %coords, <2 x double> %data)

  ; CHECK-NEXT: ret void
  ret void
}
