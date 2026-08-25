; RUN: llc -mtriple=dxil-pc-shadermodel6.8-library -o - %s | FileCheck %s
; RUN: llc -mtriple=dxil-pc-shadermodel6.9-library -stop-before=dxil-op-lower -o - %s | FileCheck %s --check-prefix=SM69-CHECK

; CHECK-LABEL: define <17 x float> @test_legacyf16tof32(
; CHECK-COUNT-17: call float @dx.op.legacyF16ToF32(i32 131, i32 {{.*}})
; SM69-CHECK-LABEL: define <17 x float> @test_legacyf16tof32(
; SM69-CHECK: call <17 x float> @llvm.dx.legacyf16tof32.v17i32
define <17 x float> @test_legacyf16tof32(<17 x i32> %a) {
  %result = call <17 x float> @llvm.dx.legacyf16tof32.v17i32(<17 x i32> %a)
  ret <17 x float> %result
}
declare <17 x float> @llvm.dx.legacyf16tof32.v17i32(<17 x i32>)
