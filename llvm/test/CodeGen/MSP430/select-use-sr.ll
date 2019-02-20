; RUN: llc < %s -march=msp430 | FileCheck %s
; PR32769

target triple = "msp430"

; Test that CMP instruction is not removed by MachineCSE.
;
; CHECK-LABEL: @f
; CHECK: cmp r15, r13
; CHECK: cmp r15, r13
; CHECK-NEXT: jeq .LBB0_2
define i16 @f(i16, i16, i16, i16) {
entry:
  %4 = icmp ult i16 %1, %3
  %5 = zext i1 %4 to i16
  %6 = icmp ult i16 %0, %2
  %7 = zext i1 %6 to i16
  %8 = icmp eq i16 %1, %3
  %out = select i1 %8, i16 %5, i16 %7
  ret i16 %out
}
