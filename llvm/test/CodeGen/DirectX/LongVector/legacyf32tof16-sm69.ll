; XFAIL: *
; RUN: llc -mtriple=dxil-pc-shadermodel6.9-library -stop-before=dxil-op-lower -o - %s | FileCheck %s

; CHECK-LABEL: define <17 x i32> @test_legacyf32tof16(
; CHECK: call <17 x i32> @llvm.dx.legacyf32tof16.v17f32
define <17 x i32> @test_legacyf32tof16(<17 x float> %a) {
  %result = call <17 x i32> @llvm.dx.legacyf32tof16.v17f32(<17 x float> %a)
  ret <17 x i32> %result
}
declare <17 x i32> @llvm.dx.legacyf32tof16.v17f32(<17 x float>)
