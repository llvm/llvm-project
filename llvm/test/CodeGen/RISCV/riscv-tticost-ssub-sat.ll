; REQUIRES: riscv-registered-target
; RUN: llc -mtriple=riscv64 -mattr=+v -O2 %s -o - | FileCheck %s --check-prefix=ASM

declare <4 x i32> @llvm.ssub.sat.v4i32(<4 x i32>, <4 x i32>)

define <4 x i32> @test_ssub_sat(<4 x i32> %a, <4 x i32> %b) {
  %r = call <4 x i32> @llvm.ssub.sat.v4i32(<4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %r
}

; ASM: vssub.vv    {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
; ASM-NOT: vssubu.vv
