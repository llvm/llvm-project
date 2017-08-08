; Test that the backend does not mess up the I/R in case of a use of an undef
; register. This typically happens while expanding a pseudo or otherwise
; replacing an instruction for another.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 -verify-machineinstrs | FileCheck %s

; LLCRMux
define void @f1(i8*) {
; CHECK-LABEL: f1:
; CHECK-NOT: *** Bad machine code: Using an undefined physical register ***
BB:
  %L5 = load i8, i8* %0
  %B9 = lshr i8 %L5, -1
  br label %CF

CF:                                               ; preds = %CF, %BB
  %Cmp25 = icmp ne i8 27, %B9
  br i1 %Cmp25, label %CF, label %CF34

CF34:                                             ; preds = %CF
  ret void
}
