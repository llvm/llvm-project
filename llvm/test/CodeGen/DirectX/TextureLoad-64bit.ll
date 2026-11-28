; RUN: opt -S -dxil-intrinsic-expansion < %s | FileCheck %s

; DXIL has no 64-bit TextureLoad overload, so 64-bit texture loads are split
; into loads of i32 pairs, which are recombined into the 64-bit values.

target triple = "dxil-pc-shadermodel6.6-compute"

declare void @use_i64(i64)
declare void @use_v2i64(<2 x i64>)
declare void @use_double(double)
declare void @use_v2double(<2 x double>)

; CHECK-LABEL: define void @load_texture2d_i64(
define void @load_texture2d_i64(<2 x i32> %coords) {
  %texture = call target("dx.Texture", i64, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_i64_0_0_0_2t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK-NOT: call i64 @llvm.dx.resource.load.level
  ; CHECK: [[L:%.*]] = call <2 x i32>
  ; CHECK-SAME: @llvm.dx.resource.load.level.v2i32.tdx.Texture_i64_0_0_0_2t.v2i32.i32.v2i32(
  ; CHECK-SAME: target("dx.Texture", i64, 0, 0, 0, 2) %texture,
  ; CHECK-SAME: <2 x i32> %coords, i32 0, <2 x i32> zeroinitializer)
  ; CHECK: [[Lo:%.*]] = extractelement <2 x i32> [[L]], i32 0
  ; CHECK: [[Hi:%.*]] = extractelement <2 x i32> [[L]], i32 1
  ; CHECK: [[ZLo:%.*]] = zext i32 [[Lo]] to i64
  ; CHECK: [[ZHi:%.*]] = zext i32 [[Hi]] to i64
  ; CHECK: [[ShHi:%.*]] = shl i64 [[ZHi]], 32
  ; CHECK: [[V:%.*]] = or i64 [[ZLo]], [[ShHi]]
  %data = call i64 @llvm.dx.resource.load.level.i64.tdx.Texture_i64_0_0_0_2t.v2i32.i32.v2i32(
      target("dx.Texture", i64, 0, 0, 0, 2) %texture,
      <2 x i32> %coords, i32 0, <2 x i32> zeroinitializer)

  ; CHECK: call void @use_i64(i64 [[V]])
  call void @use_i64(i64 %data)
  ret void
}

; CHECK-LABEL: define void @load_rwtexture2d_v2i64(
define void @load_rwtexture2d_v2i64(<2 x i32> %coords) {
  %texture = call target("dx.Texture", <2 x i64>, 1, 0, 1, 2)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_v2i64_1_0_1_2t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: [[L:%.*]] = call <4 x i32>
  ; CHECK-SAME: @llvm.dx.resource.load.level.v4i32.tdx.Texture_v2i64_1_0_1_2t.v2i32.i32.v2i32(
  ; CHECK-SAME: target("dx.Texture", <2 x i64>, 1, 0, 1, 2) %texture,
  ; CHECK-SAME: <2 x i32> %coords, i32 0, <2 x i32> zeroinitializer)
  ; CHECK: [[Lo0:%.*]] = extractelement <4 x i32> [[L]], i32 0
  ; CHECK: [[Hi0:%.*]] = extractelement <4 x i32> [[L]], i32 1
  ; CHECK: [[Lo1:%.*]] = extractelement <4 x i32> [[L]], i32 2
  ; CHECK: [[Hi1:%.*]] = extractelement <4 x i32> [[L]], i32 3
  ; CHECK: [[ZLo0:%.*]] = zext i32 [[Lo0]] to i64
  ; CHECK: [[ZHi0:%.*]] = zext i32 [[Hi0]] to i64
  ; CHECK: [[ShHi0:%.*]] = shl i64 [[ZHi0]], 32
  ; CHECK: [[V0:%.*]] = or i64 [[ZLo0]], [[ShHi0]]
  ; CHECK: [[Vec0:%.*]] = insertelement <2 x i64> poison, i64 [[V0]], i32 0
  ; CHECK: [[ZLo1:%.*]] = zext i32 [[Lo1]] to i64
  ; CHECK: [[ZHi1:%.*]] = zext i32 [[Hi1]] to i64
  ; CHECK: [[ShHi1:%.*]] = shl i64 [[ZHi1]], 32
  ; CHECK: [[V1:%.*]] = or i64 [[ZLo1]], [[ShHi1]]
  ; CHECK: [[Vec1:%.*]] = insertelement <2 x i64> [[Vec0]], i64 [[V1]], i32 1
  %data = call <2 x i64> @llvm.dx.resource.load.level.v2i64.tdx.Texture_v2i64_1_0_1_2t.v2i32.i32.v2i32(
      target("dx.Texture", <2 x i64>, 1, 0, 1, 2) %texture,
      <2 x i32> %coords, i32 0, <2 x i32> zeroinitializer)

  ; CHECK: call void @use_v2i64(<2 x i64> [[Vec1]])
  call void @use_v2i64(<2 x i64> %data)
  ret void
}

; The mip level and offsets are forwarded unchanged.
; CHECK-LABEL: define void @load_texture2d_double_level_offset(
define void @load_texture2d_double_level_offset(<2 x i32> %coords, i32 %mip) {
  %texture = call target("dx.Texture", double, 0, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_f64_0_0_0_2t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: [[L:%.*]] = call <2 x i32>
  ; CHECK-SAME: @llvm.dx.resource.load.level.v2i32.tdx.Texture_f64_0_0_0_2t.v2i32.i32.v2i32(
  ; CHECK-SAME: target("dx.Texture", double, 0, 0, 0, 2) %texture,
  ; CHECK-SAME: <2 x i32> %coords, i32 %mip, <2 x i32> <i32 1, i32 -1>)
  ; CHECK: [[Lo:%.*]] = extractelement <2 x i32> [[L]], i32 0
  ; CHECK: [[Hi:%.*]] = extractelement <2 x i32> [[L]], i32 1
  ; CHECK: [[V:%.*]] = call double @llvm.dx.asdouble.i32(i32 [[Lo]], i32 [[Hi]])
  %data = call double @llvm.dx.resource.load.level.f64.tdx.Texture_f64_0_0_0_2t.v2i32.i32.v2i32(
      target("dx.Texture", double, 0, 0, 0, 2) %texture,
      <2 x i32> %coords, i32 %mip, <2 x i32> <i32 1, i32 -1>)

  ; CHECK: call void @use_double(double [[V]])
  call void @use_double(double %data)
  ret void
}

; A Texture1D has a scalar coordinate and offset.
; CHECK-LABEL: define void @load_texture1d_v2double(
define void @load_texture1d_v2double(i32 %coord) {
  %texture = call target("dx.Texture", <2 x double>, 0, 0, 0, 1)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_v2f64_0_0_0_1t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: [[L:%.*]] = call <4 x i32>
  ; CHECK-SAME: @llvm.dx.resource.load.level.v4i32.tdx.Texture_v2f64_0_0_0_1t.i32.i32.i32(
  ; CHECK-SAME: target("dx.Texture", <2 x double>, 0, 0, 0, 1) %texture,
  ; CHECK-SAME: i32 %coord, i32 2, i32 3)
  ; CHECK: [[Lo0:%.*]] = extractelement <4 x i32> [[L]], i32 0
  ; CHECK: [[Hi0:%.*]] = extractelement <4 x i32> [[L]], i32 1
  ; CHECK: [[Lo1:%.*]] = extractelement <4 x i32> [[L]], i32 2
  ; CHECK: [[Hi1:%.*]] = extractelement <4 x i32> [[L]], i32 3
  ; CHECK: [[V0:%.*]] = call double @llvm.dx.asdouble.i32(i32 [[Lo0]], i32 [[Hi0]])
  ; CHECK: [[Vec0:%.*]] = insertelement <2 x double> poison, double [[V0]], i32 0
  ; CHECK: [[V1:%.*]] = call double @llvm.dx.asdouble.i32(i32 [[Lo1]], i32 [[Hi1]])
  ; CHECK: [[Vec1:%.*]] = insertelement <2 x double> [[Vec0]], double [[V1]], i32 1
  %data = call fast <2 x double> @llvm.dx.resource.load.level.v2f64.tdx.Texture_v2f64_0_0_0_1t.i32.i32.i32(
      target("dx.Texture", <2 x double>, 0, 0, 0, 1) %texture,
      i32 %coord, i32 2, i32 3)

  ; CHECK: call void @use_v2double(<2 x double> [[Vec1]])
  call void @use_v2double(<2 x double> %data)
  ret void
}

; A Texture2DArray has the array slice as its third coordinate.
; CHECK-LABEL: define void @load_texture2darray_i64(
define void @load_texture2darray_i64(<3 x i32> %coords) {
  %texture = call target("dx.Texture", i64, 0, 0, 1, 7)
      @llvm.dx.resource.handlefrombinding.tdx.Texture_i64_0_0_1_7t(
          i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK: [[L:%.*]] = call <2 x i32>
  ; CHECK-SAME: @llvm.dx.resource.load.level.v2i32.tdx.Texture_i64_0_0_1_7t.v3i32.i32.v2i32(
  ; CHECK-SAME: target("dx.Texture", i64, 0, 0, 1, 7) %texture,
  ; CHECK-SAME: <3 x i32> %coords, i32 0, <2 x i32> splat (i32 1))
  %data = call i64 @llvm.dx.resource.load.level.i64.tdx.Texture_i64_0_0_1_7t.v3i32.i32.v2i32(
      target("dx.Texture", i64, 0, 0, 1, 7) %texture,
      <3 x i32> %coords, i32 0, <2 x i32> splat (i32 1))

  ; CHECK-NOT: call i64 @llvm.dx.resource.load.level
  call void @use_i64(i64 %data)
  ret void
}
