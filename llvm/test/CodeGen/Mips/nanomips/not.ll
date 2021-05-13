; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

define i32 @test_not(i32 %a) {
; CHECK: not $a0, $a0
; CHECK: NOT_NM
  %not = xor i32 %a, -1
  ret i32 %not
}
