; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

define i32 @test_and(i32 %a, i32 %b) nounwind readnone {
entry:
; CHECK: and $a0, $a0, $a1
; CHECK: AND_NM
  %anded = and i32 %a, %b
  ret i32 %anded
}
