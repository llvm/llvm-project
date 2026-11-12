; RUN: opt -S -scalarizer -dxil-op-lower \
; RUN:   -mtriple=dxil-pc-shadermodel6.5-library < %s | FileCheck %s

; CHECK-LABEL: define float @test_float(
; CHECK: call float @dx.op.waveMultiPrefixOp.f32(i32 166, float %value, i32 %m0, i32 %m1, i32 %m2, i32 %m3, i8 4, i8 0)
define float @test_float(float %value, i32 %m0, i32 %m1,
                        i32 %m2, i32 %m3) {
  %ret = call float @llvm.dx.wave.multi.prefix.product.f32(
      float %value, i32 %m0, i32 %m1, i32 %m2, i32 %m3)
  ret float %ret
}

; CHECK-LABEL: define half @test_half(
; CHECK: call half @dx.op.waveMultiPrefixOp.f16(i32 166, half %value, i32 1, i32 2, i32 4, i32 8, i8 4, i8 0)
define half @test_half(half %value) {
  %ret = call half @llvm.dx.wave.multi.prefix.product.f16(
      half %value, i32 1, i32 2, i32 4, i32 8)
  ret half %ret
}

; CHECK-LABEL: define double @test_double(
; CHECK: call double @dx.op.waveMultiPrefixOp.f64(i32 166, double %value, i32 1, i32 2, i32 4, i32 8, i8 4, i8 0)
define double @test_double(double %value) {
  %ret = call double @llvm.dx.wave.multi.prefix.product.f64(
      double %value, i32 1, i32 2, i32 4, i32 8)
  ret double %ret
}

; CHECK-LABEL: define i16 @test_i16(
; CHECK: call i16 @dx.op.waveMultiPrefixOp.i16(i32 166, i16 %value, i32 1, i32 2, i32 4, i32 8, i8 4, i8 0)
define i16 @test_i16(i16 %value) {
  %ret = call i16 @llvm.dx.wave.multi.prefix.product.i16(
      i16 %value, i32 1, i32 2, i32 4, i32 8)
  ret i16 %ret
}

; CHECK-LABEL: define i32 @test_i32(
; CHECK: call i32 @dx.op.waveMultiPrefixOp.i32(i32 166, i32 %value, i32 1, i32 2, i32 4, i32 8, i8 4, i8 0)
define i32 @test_i32(i32 %value) {
  %ret = call i32 @llvm.dx.wave.multi.prefix.product.i32(
      i32 %value, i32 1, i32 2, i32 4, i32 8)
  ret i32 %ret
}

; CHECK-LABEL: define i64 @test_i64(
; CHECK: call i64 @dx.op.waveMultiPrefixOp.i64(i32 166, i64 %value, i32 1, i32 2, i32 4, i32 8, i8 4, i8 0)
define i64 @test_i64(i64 %value) {
  %ret = call i64 @llvm.dx.wave.multi.prefix.product.i64(
      i64 %value, i32 1, i32 2, i32 4, i32 8)
  ret i64 %ret
}

; CHECK-LABEL: define i32 @test_unsigned(
; CHECK: call i32 @dx.op.waveMultiPrefixOp.i32(i32 166, i32 %value, i32 1, i32 2, i32 4, i32 8, i8 4, i8 1)
define i32 @test_unsigned(i32 %value) {
  %ret = call i32 @llvm.dx.wave.multi.prefix.uproduct.i32(
      i32 %value, i32 1, i32 2, i32 4, i32 8)
  ret i32 %ret
}

; CHECK-LABEL: define <2 x i32> @test_unsigned_vector(
; CHECK: call i32 @dx.op.waveMultiPrefixOp.i32(i32 166, i32 %value.i0, i32 %m0, i32 %m1, i32 %m2, i32 %m3, i8 4, i8 1)
; CHECK: call i32 @dx.op.waveMultiPrefixOp.i32(i32 166, i32 %value.i1, i32 %m0, i32 %m1, i32 %m2, i32 %m3, i8 4, i8 1)
define <2 x i32> @test_unsigned_vector(<2 x i32> %value, i32 %m0, i32 %m1,
                                     i32 %m2, i32 %m3) {
  %ret = call <2 x i32> @llvm.dx.wave.multi.prefix.uproduct.v2i32(
      <2 x i32> %value, i32 %m0, i32 %m1, i32 %m2, i32 %m3)
  ret <2 x i32> %ret
}

; CHECK-LABEL: define <5 x float> @test_long_vector(
; CHECK: call float @dx.op.waveMultiPrefixOp.f32(i32 166, float %value.i0, i32 1, i32 2, i32 4, i32 8, i8 4, i8 0)
; CHECK: call float @dx.op.waveMultiPrefixOp.f32(i32 166, float %value.i1, i32 1, i32 2, i32 4, i32 8, i8 4, i8 0)
; CHECK: call float @dx.op.waveMultiPrefixOp.f32(i32 166, float %value.i2, i32 1, i32 2, i32 4, i32 8, i8 4, i8 0)
; CHECK: call float @dx.op.waveMultiPrefixOp.f32(i32 166, float %value.i3, i32 1, i32 2, i32 4, i32 8, i8 4, i8 0)
; CHECK: call float @dx.op.waveMultiPrefixOp.f32(i32 166, float %value.i4, i32 1, i32 2, i32 4, i32 8, i8 4, i8 0)
define <5 x float> @test_long_vector(<5 x float> %value) {
  %ret = call <5 x float> @llvm.dx.wave.multi.prefix.product.v5f32(
      <5 x float> %value, i32 1, i32 2, i32 4, i32 8)
  ret <5 x float> %ret
}
