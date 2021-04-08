; RUN: llc -mtriple=nanomips -asm-show-inst < %s | FileCheck %s

define i32 @test_or(i32 %a, i32 %b) nounwind readnone {
entry:
; CHECK: or $a0, $a0, $a1
; CHECK: OR_NM
  %orred = or i32 %a, %b
  ret i32 %orred
}
