; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

define i32 @test_div(i32 %a, i32 %b) {
; CHECK: div $a0, $a0, $a1
; CHECK: DIV_NM
; CHECK: teq $zero, $a1, 7
; CHECK: TEQ_NM
  %div = sdiv i32 %a, %b
  ret i32 %div
}

define i32 @test_mod(i32 %a, i32 %b) {
; CHECK: mod $a0, $a0, $a1
; CHECK: MOD_NM
; CHECK: teq $zero, $a1, 7
; CHECK: TEQ_NM
  %mod = srem i32 %a, %b
  ret i32 %mod
}
