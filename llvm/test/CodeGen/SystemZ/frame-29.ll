; RUN: llc %s -o - -mtriple=s390x-linux-gnu -mcpu=z16 -print-after=finalize-isel 2>&1 | FileCheck %s
;
; Test that the correct space is allocated for the outgoing stack argument.

declare void @bar(i72 %Arg);

define void @foo() {
; CHECK-LABEL: # Machine code for function foo: IsSSA, TracksLiveness
; CHECK-NEXT: Frame Objects:
; CHECK-NEXT:   fi#0: size=1, align=2, at location [SP]
; CHECK-NEXT:   fi#1: size=16, align=8, at location [SP]

; CHECK-LABEL: foo:
; CHECK: aghi %r15, -184
  %1 = alloca i8, align 2
  tail call fastcc void @bar(i72 2097168)
  ret void
}
