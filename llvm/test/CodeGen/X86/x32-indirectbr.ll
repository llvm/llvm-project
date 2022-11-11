; RUN: llc < %s -mtriple=x86_64-none-none-gnux32 -mcpu=generic | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-none-none-gnux32 -mcpu=generic -fast-isel | FileCheck %s
; Bug 22859
;
; x32 pointers are 32-bits wide. x86-64 indirect branches use the full 64-bit
; registers. Therefore, x32 CodeGen needs to zero extend indirectbr's target to
; 64-bit.

define i8 @test1() nounwind ssp {
entry:
  %0 = select i1 undef,                           ; <ptr> [#uses=1]
              ptr blockaddress(@test1, %bb),
              ptr blockaddress(@test1, %bb6)
  indirectbr ptr %0, [label %bb, label %bb6]
bb:                                               ; preds = %entry
  ret i8 1

bb6:                                              ; preds = %entry
  ret i8 2
}
; CHECK-LABEL: @test1
; We are looking for a movl ???, %r32 followed by a 64-bit jmp through the
; same register.
; CHECK: movl {{.*}}, %{{e|r}}[[REG:.[^d]*]]{{d?}}
; CHECK-NEXT: jmpq *%r[[REG]]

