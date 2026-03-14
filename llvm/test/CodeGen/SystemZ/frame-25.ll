; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s
;
; Test that space is allocated for the incoming back chain also in cases
; where no GPRs are saved / restored.

define void @fun0() #0 {
; CHECK-LABEL: fun0:
; CHECK: lgr     %r1, %r15
; CHECK-NEXT: aghi    %r15, -24
; CHECK-NEXT: stg     %r1, 152(%r15)
; CHECK-NEXT: #APP
; CHECK-NEXT: stcke   160(%r15)
; CHECK-NEXT: #NO_APP
; CHECK-NEXT: aghi    %r15, 24
; CHECK-NEXT: br      %r14

entry:
  %b = alloca [16 x i8], align 1
  call void asm "stcke $0", "=*Q"(ptr elementtype([16 x i8]) nonnull %b) #2
  ret void
}

attributes #0 = { nounwind "packed-stack" "backchain" "use-soft-float"="true" }
