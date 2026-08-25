; RUN: llc -mtriple=dxil-pc-shadermodel6.8-library -o - %s | FileCheck %s
; RUN: llc -mtriple=dxil-pc-shadermodel6.9-library -stop-before=dxil-op-lower -o - %s | FileCheck %s --check-prefix=SM69-CHECK

; CHECK-LABEL: define <17 x i32> @test_wave_reduce_umax(
; CHECK-COUNT-17: call i32 @dx.op.waveActiveOp.i32(i32 119, i32 {{.*}}, i8 3, i8 1)
; SM69-CHECK-LABEL: define <17 x i32> @test_wave_reduce_umax(
; SM69-CHECK: call <17 x i32> @llvm.dx.wave.reduce.umax.v17i32
define <17 x i32> @test_wave_reduce_umax(<17 x i32> %a) {
  %result = call <17 x i32> @llvm.dx.wave.reduce.umax.v17i32(<17 x i32> %a)
  ret <17 x i32> %result
}
declare <17 x i32> @llvm.dx.wave.reduce.umax.v17i32(<17 x i32>)
