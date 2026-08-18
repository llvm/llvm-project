; XFAIL: *
; RUN: llc -mtriple=dxil-pc-shadermodel6.9-library -stop-before=dxil-op-lower -o - %s | FileCheck %s

; CHECK-LABEL: define <17 x double> @test_asdouble(
; CHECK: call <17 x double> @llvm.dx.asdouble.v17i32
define <17 x double> @test_asdouble(<17 x i32> %a, <17 x i32> %b) {
  %result = call <17 x double> @llvm.dx.asdouble.v17i32(<17 x i32> %a, <17 x i32> %b)
  ret <17 x double> %result
}
declare <17 x double> @llvm.dx.asdouble.v17i32(<17 x i32>, <17 x i32>)
