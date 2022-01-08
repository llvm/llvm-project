; Test argument passing and stack frame construction, callee side.
;
; RUN: llc < %s -mtriple=m88k-openbsd -mcpu=mc88100 | FileCheck %s
; RUN: llc < %s -mtriple=m88k-openbsd -mcpu=mc88110 | FileCheck %s

; Registers r2 to r9 used, no parameter passed on stack.
define i32 @f1(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f, i32 %g, i32 %h) {
; CHECK-LABEL: f1:
; CHECK: and %r2, %r2, %r3
; CHECK: and %r2, %r4, %r2
; CHECK: and %r2, %r5, %r2
; CHECK: and %r2, %r6, %r2
; CHECK: and %r2, %r7, %r2
; CHECK: and %r2, %r8, %r2
; CHECK: and %r2, %r9, %r2
; CHECK: jmp %r1
  %sum1 = and i32 %a, %b
  %sum2 = and i32 %c, %sum1
  %sum3 = and i32 %d, %sum2
  %sum4 = and i32 %e, %sum3
  %sum5 = and i32 %f, %sum4
  %sum6 = and i32 %g, %sum5
  %sum7 = and i32 %h, %sum6
  ret i32 %sum7
}

; Registers r2 to r9 used, 2 parameter passed on stack.
define i32 @f2(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %a, i32 %b) {
; CHECK-LABEL: f2:
; CHECK: ld %r2, %r31, 0
; CHECK: ld %r3, %r31, 4
; CHECK: or %r2, %r2, %r3
; CHECK: jmp %r1
  %sum = or i32 %a, %b
  ret i32 %sum
}

; Float arguments passed in r2 and r3.
define float @f3(float %a, float %b) {
; CHECK-LABEL: f3:
; CHECK: fadd.sss %r2, %r2, %r3
; CHECK: jmp %r1
  %sum = fadd float %a, %b
  ret float %sum
}
