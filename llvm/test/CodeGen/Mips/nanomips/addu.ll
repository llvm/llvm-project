; RUN: llc -march=nanomips -mcpu=nanomips  < %s | FileCheck %s

define i32 @test_addu(i32 %a, i32 %b) nounwind readnone {
entry:
; CHECK: addu $a0, $a0, $a1
  %added = add i32 %a, %b
; CHECK: jrc $ra
  ret i32 %added
}
