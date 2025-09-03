; REQUIRES: riscv-registered-target
; RUN: llc -mtriple=riscv64 -mattr=+v -O2 %s -o - | FileCheck %s

; SHL
define <4 x i32> @test_shl(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_shl:
; CHECK: vsll.vv
  %shl = shl <4 x i32> %a, %b
  ret <4 x i32> %shl
}

; SRL
define <4 x i32> @test_srl(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_srl:
; CHECK: vsrl.vv
  %srl = lshr <4 x i32> %a, %b
  ret <4 x i32> %srl
}

; SRA
define <4 x i32> @test_sra(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_sra:
; CHECK: vsra.vv
  %sra = ashr <4 x i32> %a, %b
  ret <4 x i32> %sra
}
