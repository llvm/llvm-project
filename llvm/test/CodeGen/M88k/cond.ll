; Test conditionals.
;
; RUN: llc < %s -mtriple=m88k-openbsd -mcpu=mc88100 | FileCheck %s
; RUN: llc < %s -mtriple=m88k-openbsd -mcpu=mc88110 | FileCheck %s

define i32 @f1(i32 %a, i32 %b) {
; CHECK-LABEL: f1:
; CHECK: bcnd 13, %r2, .LBB0_2
; CHECK: br .{{[A-Z0-9]+}}
; CHECK: jmp %r1
  %cmp = icmp eq i32 %a, 0
  br i1 %cmp, label %compute, label %exit

compute:
  %res = and i32 %a, %b
  ret i32 %res

exit:
  ret i32 0
;  ret void
}

define i32 @f2(i32 %a, i32 %b) {
; CHECK-LABEL: f2:
; CHECK: cmp %r4, %r2, %r3
; CHECK: bb1 2, %r4, .{{[A-Z0-9]+}}
; CHECK: jmp %r1
  %cmp = icmp ne i32 %a, %b
  br i1 %cmp, label %compute, label %exit

compute:
  %res = and i32 %a, %b
  ret i32 %res

exit:
  ret i32 0
;  ret void
}
