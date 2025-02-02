; RUN: llc < %s -mtriple=avr | FileCheck %s

; Test case for an assertion error.
;
; Error:
; ```
; Impossible reg-to-reg copy
; UNREACHABLE executed at lib/Target/AVR/AVRInstrInfo.cpp
; ```
;
; This no longer occurs.

declare { i16, i1 } @llvm.umul.with.overflow.i16(i16, i16)

; CHECK-LABEL: foo
define void @foo() {
entry-block:
  %0 = call { i16, i1 } @llvm.umul.with.overflow.i16(i16 undef, i16 undef)
  %1 = extractvalue { i16, i1 } %0, 1
  %2 = icmp eq i1 %1, true
  br i1 %2, label %cond, label %next

next:                                             ; preds = %entry-block
  ret void
cond:                                             ; preds = %entry-block
  unreachable
}
