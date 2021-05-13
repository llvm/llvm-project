; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

define i32 @test_xor(i32 %a, i32 %b) {
; CHECK: xor $a0, $a0, $a1
; CHECK: XOR_NM
  %xor = xor i32 %a, %b
  ret i32 %xor
}
