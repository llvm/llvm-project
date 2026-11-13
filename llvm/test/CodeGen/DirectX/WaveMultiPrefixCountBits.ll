; RUN: opt -S -dxil-op-lower \
; RUN:   -mtriple=dxil-pc-shadermodel6.5-library < %s | FileCheck %s

define i32 @test_dynamic(i1 %value, i32 %m0, i32 %m1, i32 %m2, i32 %m3) {
  ; CHECK-LABEL: define i32 @test_dynamic(
  ; CHECK: %[[RET:.*]] = call i32 @dx.op.waveMultiPrefixBitCount(i32 167, i1 %value, i32 %m0, i32 %m1, i32 %m2, i32 %m3)
  ; CHECK: ret i32 %[[RET]]
  %ret = call i32 @llvm.dx.wave.multi.prefix.bit.count(
      i1 %value, i32 %m0, i32 %m1, i32 %m2, i32 %m3)
  ret i32 %ret
}

define i32 @test_constant() {
  ; CHECK-LABEL: define i32 @test_constant(
  ; CHECK: %[[RET:.*]] = call i32 @dx.op.waveMultiPrefixBitCount(i32 167, i1 true, i32 1, i32 2, i32 4, i32 8)
  ; CHECK: ret i32 %[[RET]]
  %ret = call i32 @llvm.dx.wave.multi.prefix.bit.count(
      i1 true, i32 1, i32 2, i32 4, i32 8)
  ret i32 %ret
}

define i32 @test_false() {
  ; CHECK-LABEL: define i32 @test_false(
  ; CHECK: %[[RET:.*]] = call i32 @dx.op.waveMultiPrefixBitCount(
  ; CHECK-SAME: i32 167, i1 false, i32 16, i32 32, i32 64, i32 128)
  ; CHECK: ret i32 %[[RET]]
  %ret = call i32 @llvm.dx.wave.multi.prefix.bit.count(
      i1 false, i32 16, i32 32, i32 64, i32 128)
  ret i32 %ret
}
