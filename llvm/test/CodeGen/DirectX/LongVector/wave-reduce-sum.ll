; RUN: llc -mtriple=dxil-pc-shadermodel6.8-library -o - %s | FileCheck %s

; CHECK-LABEL: define <17 x i32> @test_wave_reduce_sum(
; CHECK-COUNT-17: call i32 @dx.op.waveActiveOp.i32(i32 119, i32 {{.*}}, i8 0, i8 0)
define <17 x i32> @test_wave_reduce_sum(<17 x i32> %a) {
  %result = call <17 x i32> @llvm.dx.wave.reduce.sum.v17i32(<17 x i32> %a)
  ret <17 x i32> %result
}
declare <17 x i32> @llvm.dx.wave.reduce.sum.v17i32(<17 x i32>)
