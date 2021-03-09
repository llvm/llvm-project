; RUN: llc -march=nanomips -mcpu=nanomips  < %s | FileCheck %s

define i32 @test_addiu0(i32 %a) nounwind readnone {
entry:
; CHECK: addiu ${{[0-9]+}}, ${{[0-9]+}}, 1
  %added = add i32 %a, 1
  ret i32 %added
}

define i32 @test_addiu1(i32 %a) nounwind readnone {
entry:
; CHECK: addiu ${{[0-9]+}}, ${{[0-9]+}}, 32767
  %added = add i32 %a, 32767
  ret i32 %added
}

define i32 @test_addiu2(i32 %a) nounwind readnone {
entry:
; CHECK: addiu ${{[0-9]+}}, ${{[0-9]+}}, -32768
  %added = add i32 %a, -32768
  ret i32 %added
}
