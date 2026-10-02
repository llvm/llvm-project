; RUN: llc -mtriple=dxil-pc-shadermodel6.8-library -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SCALAR
; RUN: llc -mtriple=dxil-pc-shadermodel6.9-library -stop-before=dxil-op-lower -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-VECTOR

; CHECK-LABEL: define <17 x i1> @test_wave_all_equal(
; CHECK-SCALAR-COUNT-17: call i1 @dx.op.waveActiveAllEqual.i32(i32 115, i32 {{.*}})
; CHECK-VECTOR: call <17 x i1> @llvm.dx.wave.all.equal.v17i32
define <17 x i1> @test_wave_all_equal(<17 x i32> %a) {
  %result = call <17 x i1> @llvm.dx.wave.all.equal.v17i32(<17 x i32> %a)
  ret <17 x i1> %result
}
declare <17 x i1> @llvm.dx.wave.all.equal.v17i32(<17 x i32>)
