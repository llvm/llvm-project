; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s
;
; Test backchain with packed-stack, which requires soft-float.

attributes #0 = { nounwind "backchain" "packed-stack" "use-soft-float"="true" }
define i64 @fun0(i64 %a) #0 {
; CHECK-LABEL: fun0:
; CHECK:      stmg	%r14, %r15, 136(%r15)
; CHECK-NEXT: lgr	%r1, %r15
; CHECK-NEXT: aghi	%r15, -24
; CHECK-NEXT: stg	%r1, 152(%r15)
; CHECK-NEXT: brasl	%r14, foo@PLT
; CHECK-NEXT: lmg	%r14, %r15, 160(%r15)
; CHECK-NEXT: br	%r14
entry:
  %call = call i64 @foo(i64 %a)
  ret i64 %call
}

declare i64 @foo(i64)
