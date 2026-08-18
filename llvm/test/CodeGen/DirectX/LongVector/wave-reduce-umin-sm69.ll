; XFAIL: *
; RUN: llc -mtriple=dxil-pc-shadermodel6.9-library -stop-before=dxil-op-lower -o - %s | FileCheck %s

; CHECK-LABEL: define <17 x i32> @test_wave_reduce_umin(
; CHECK: call <17 x i32> @llvm.dx.wave.reduce.umin.v17i32
define <17 x i32> @test_wave_reduce_umin(<17 x i32> %a) {
  %result = call <17 x i32> @llvm.dx.wave.reduce.umin.v17i32(<17 x i32> %a)
  ret <17 x i32> %result
}
declare <17 x i32> @llvm.dx.wave.reduce.umin.v17i32(<17 x i32>)
