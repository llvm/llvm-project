; RUN: llc -mtriple=dxil-pc-shadermodel6.8-library -o - %s | FileCheck %s

; CHECK-LABEL: define <17 x double> @test_asdouble(
; CHECK-COUNT-17: call double @dx.op.makeDouble.f64(i32 101, i32 {{.*}}, i32 {{.*}})
define <17 x double> @test_asdouble(<17 x i32> %a, <17 x i32> %b) {
  %result = call <17 x double> @llvm.dx.asdouble.v17i32(<17 x i32> %a, <17 x i32> %b)
  ret <17 x double> %result
}
declare <17 x double> @llvm.dx.asdouble.v17i32(<17 x i32>, <17 x i32>)
