; RUN: llc -mtriple=nanomips -asm-show-inst < %s | FileCheck %s

define i32 @test_subu(i32 %a, i32 %b) nounwind readnone {
entry:
; CHECK: subu $a0, $a0, $a1
; CHECK: SUBu_NM
  %subbed = sub i32 %a, %b
  ret i32 %subbed
}
