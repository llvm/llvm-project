; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

define i32 @test_nor(i32 %a, i32 %b) {
; CHECK: nor $a0, $a0, $a1
; CHECK: NOR_NM
  %or = or i32 %a, %b
  %nor = xor i32 %or, -1
  ret i32 %nor
}
