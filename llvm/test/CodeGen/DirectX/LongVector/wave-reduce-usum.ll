; RUN: llc -mtriple=dxil-pc-shadermodel6.8-library -o - %s | FileCheck %s

; CHECK-LABEL: define <17 x i32> @test_wave_reduce_usum(
; CHECK-COUNT-17: call i32 @dx.op.waveActiveOp.i32(i32 119, i32 {{.*}}, i8 0, i8 1)
define <17 x i32> @test_wave_reduce_usum(<17 x i32> %a) {
  %result = call <17 x i32> @llvm.dx.wave.reduce.usum.v17i32(<17 x i32> %a)
  ret <17 x i32> %result
}
declare <17 x i32> @llvm.dx.wave.reduce.usum.v17i32(<17 x i32>)
