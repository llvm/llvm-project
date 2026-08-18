; RUN: llc -mtriple=dxil-pc-shadermodel6.8-library -o - %s | FileCheck %s

; CHECK-LABEL: define <17 x i32> @test_wave_reduce_or(
; CHECK-COUNT-17: call i32 @dx.op.waveActiveBit.i32(i32 120, i32 {{.*}}, i8 1)
define <17 x i32> @test_wave_reduce_or(<17 x i32> %a) {
  %result = call <17 x i32> @llvm.dx.wave.reduce.or.v17i32(<17 x i32> %a)
  ret <17 x i32> %result
}
declare <17 x i32> @llvm.dx.wave.reduce.or.v17i32(<17 x i32>)
