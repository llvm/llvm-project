; RUN: llvm-dis %p/aarch64-addp-upgrade.bc -o - | FileCheck %s

; Bitcode was generated from file below, which may or may not even assemble any
; more.

; CHECK: call <2 x float> @llvm.aarch64.neon.faddp.v2f32(<2 x float> %lhs, <2 x float> %rhs)
define <2 x float> @test_addp(<2 x float> %lhs, <2 x float> %rhs) {
  %res = call <2 x float> @llvm.aarch64.neon.addp.v2f32(<2 x float> %lhs, <2 x float> %rhs)
  ret <2 x float> %res
}

; CHECK: call <2 x float> @llvm.aarch64.neon.faddp.v2f32(<2 x float> %lhs, <2 x float> %rhs)
define <2 x float> @test_addp1(<2 x float> %lhs, <2 x float> %rhs) {
  %res = call <2 x float> @llvm.aarch64.neon.addp.v2f32(<2 x float> %lhs, <2 x float> %rhs)
  ret <2 x float> %res
}

declare <2 x float> @llvm.aarch64.neon.addp.v2f32(<2 x float>, <2 x float>)
