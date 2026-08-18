; XFAIL: *
; RUN: llc -mtriple=dxil-pc-shadermodel6.9-library -stop-before=dxil-op-lower -o - %s | FileCheck %s

; CHECK-LABEL: define <17 x i1> @test_wave_all_equal(
; CHECK: call <17 x i1> @llvm.dx.wave.all.equal.v17i32
define <17 x i1> @test_wave_all_equal(<17 x i32> %a) {
  %result = call <17 x i1> @llvm.dx.wave.all.equal.v17i32(<17 x i32> %a)
  ret <17 x i1> %result
}
declare <17 x i1> @llvm.dx.wave.all.equal.v17i32(<17 x i32>)
