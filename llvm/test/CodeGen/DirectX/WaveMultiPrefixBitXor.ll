; RUN: opt -S -scalarizer -dxil-op-lower \
; RUN:   -mtriple=dxil-pc-shadermodel6.5-library < %s | FileCheck %s

; CHECK-LABEL: define i16 @test_i16(
; CHECK: call i16 @dx.op.waveMultiPrefixOp.i16(i32 166, i16 %value, i32 %m0, i32 %m1, i32 %m2, i32 %m3, i8 3, i8 0)
define i16 @test_i16(i16 %value, i32 %m0, i32 %m1, i32 %m2, i32 %m3) {
  %ret = call i16 @llvm.dx.wave.multi.prefix.bit.xor.i16(
      i16 %value, i32 %m0, i32 %m1, i32 %m2, i32 %m3)
  ret i16 %ret
}

; CHECK-LABEL: define i32 @test_i32(
; CHECK: call i32 @dx.op.waveMultiPrefixOp.i32(i32 166, i32 %value, i32 %m0, i32 %m1, i32 %m2, i32 %m3, i8 3, i8 0)
define i32 @test_i32(i32 %value, i32 %m0, i32 %m1, i32 %m2, i32 %m3) {
  %ret = call i32 @llvm.dx.wave.multi.prefix.bit.xor.i32(
      i32 %value, i32 %m0, i32 %m1, i32 %m2, i32 %m3)
  ret i32 %ret
}

; CHECK-LABEL: define i64 @test_i64(
; CHECK: call i64 @dx.op.waveMultiPrefixOp.i64(i32 166, i64 %value, i32 %m0, i32 %m1, i32 %m2, i32 %m3, i8 3, i8 0)
define i64 @test_i64(i64 %value, i32 %m0, i32 %m1, i32 %m2, i32 %m3) {
  %ret = call i64 @llvm.dx.wave.multi.prefix.bit.xor.i64(
      i64 %value, i32 %m0, i32 %m1, i32 %m2, i32 %m3)
  ret i64 %ret
}

; CHECK-LABEL: define <2 x i32> @test_vector(
; CHECK: call i32 @dx.op.waveMultiPrefixOp.i32(i32 166, i32 %value.i0, i32 %m0, i32 %m1, i32 %m2, i32 %m3, i8 3, i8 0)
; CHECK: call i32 @dx.op.waveMultiPrefixOp.i32(i32 166, i32 %value.i1, i32 %m0, i32 %m1, i32 %m2, i32 %m3, i8 3, i8 0)
define <2 x i32> @test_vector(<2 x i32> %value, i32 %m0, i32 %m1,
                              i32 %m2, i32 %m3) {
  %ret = call <2 x i32> @llvm.dx.wave.multi.prefix.bit.xor.v2i32(
      <2 x i32> %value, i32 %m0, i32 %m1, i32 %m2, i32 %m3)
  ret <2 x i32> %ret
}

; CHECK-LABEL: define <5 x i32> @test_long_vector(
; CHECK: call i32 @dx.op.waveMultiPrefixOp.i32(i32 166, i32 %value.i0, i32 1, i32 2, i32 4, i32 8, i8 3, i8 0)
; CHECK: call i32 @dx.op.waveMultiPrefixOp.i32(i32 166, i32 %value.i1, i32 1, i32 2, i32 4, i32 8, i8 3, i8 0)
; CHECK: call i32 @dx.op.waveMultiPrefixOp.i32(i32 166, i32 %value.i2, i32 1, i32 2, i32 4, i32 8, i8 3, i8 0)
; CHECK: call i32 @dx.op.waveMultiPrefixOp.i32(i32 166, i32 %value.i3, i32 1, i32 2, i32 4, i32 8, i8 3, i8 0)
; CHECK: call i32 @dx.op.waveMultiPrefixOp.i32(i32 166, i32 %value.i4, i32 1, i32 2, i32 4, i32 8, i8 3, i8 0)
define <5 x i32> @test_long_vector(<5 x i32> %value) {
  %ret = call <5 x i32> @llvm.dx.wave.multi.prefix.bit.xor.v5i32(
      <5 x i32> %value, i32 1, i32 2, i32 4, i32 8)
  ret <5 x i32> %ret
}
