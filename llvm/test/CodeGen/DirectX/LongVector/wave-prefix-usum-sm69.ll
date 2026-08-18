; XFAIL: *
; RUN: llc -mtriple=dxil-pc-shadermodel6.9-library -stop-before=dxil-op-lower -o - %s | FileCheck %s

; CHECK-LABEL: define <17 x i32> @test_wave_prefix_usum(
; CHECK: call <17 x i32> @llvm.dx.wave.prefix.usum.v17i32
define <17 x i32> @test_wave_prefix_usum(<17 x i32> %a) {
  %result = call <17 x i32> @llvm.dx.wave.prefix.usum.v17i32(<17 x i32> %a)
  ret <17 x i32> %result
}
declare <17 x i32> @llvm.dx.wave.prefix.usum.v17i32(<17 x i32>)
