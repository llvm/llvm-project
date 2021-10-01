; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

define i32 @test(i32 %a, i32 %b) {
; CHECK: move $a0, $a1
  ret i32 %b
}
