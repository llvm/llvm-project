; RUN: llc -march=nanomips -mcpu=nanomips  < %s | FileCheck %s

define i32 @test_addu(i32 %a, i32 %b) nounwind readnone {
entry:
; CHECK: addu ${{[0-9]+}}, ${{[0-9]+}}, ${{[0-9]+}}
  %added = add i32 %a, %b
  ret i32 %added
}
