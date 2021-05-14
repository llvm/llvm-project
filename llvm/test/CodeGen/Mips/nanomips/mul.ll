; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

define i32 @test_mul(i32 %a, i32 %b) {
; CHECK: mul $a0, $a0, $a1
; CHECK: MUL_NM
  %mul = mul i32 %a, %b
  ret i32 %mul
}
