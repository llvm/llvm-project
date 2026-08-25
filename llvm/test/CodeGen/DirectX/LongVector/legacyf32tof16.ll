; RUN: llc -mtriple=dxil-pc-shadermodel6.8-library -o - %s | FileCheck %s
; RUN: llc -mtriple=dxil-pc-shadermodel6.9-library -stop-before=dxil-op-lower -o - %s | FileCheck %s --check-prefix=SM69-CHECK

; CHECK-LABEL: define <17 x i32> @test_legacyf32tof16(
; CHECK-COUNT-17: call i32 @dx.op.legacyF32ToF16(i32 130, float {{.*}})
; SM69-CHECK-LABEL: define <17 x i32> @test_legacyf32tof16(
; SM69-CHECK: call <17 x i32> @llvm.dx.legacyf32tof16.v17f32
define <17 x i32> @test_legacyf32tof16(<17 x float> %a) {
  %result = call <17 x i32> @llvm.dx.legacyf32tof16.v17f32(<17 x float> %a)
  ret <17 x i32> %result
}
declare <17 x i32> @llvm.dx.legacyf32tof16.v17f32(<17 x float>)
