; Test conditionals.
;
; RUN: llc < %s -mtriple=m88k-openbsd -mcpu=mc88100 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=m88k-openbsd -mcpu=mc88110 -verify-machineinstrs | FileCheck %s

define i32 @f1(i32 %a, i32 %b) {
; CHECK-LABEL: f1:
; CHECK: bcnd.n ne0, %r2, .LBB0_{{[0-9]}}
; CHECK: jmp.n %r1
; CHECK: jmp.n %r1
  %cmp = icmp eq i32 %a, 0
  br i1 %cmp, label %compute, label %exit

compute:
  %res = and i32 %a, %b
  ret i32 %res

exit:
  ret i32 0
}

define i32 @f2(i32 %a, i32 %b) {
; CHECK-LABEL: f2:
; CHECK: cmp [[REG:%r[0-9]+]], %r2, %r3
; CHECK-NEXT: bb1.n 2, [[REG]], .{{[A-Z0-9]+}}
; CHECK: jmp.n %r1
; CHECK: jmp.n %r1
  %cmp = icmp ne i32 %a, %b
  br i1 %cmp, label %compute, label %exit

compute:
  %res = and i32 %a, %b
  ret i32 %res

exit:
  ret i32 0
}
