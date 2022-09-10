; Test addition/subtraction.
;
; RUN: llc < %s -mtriple=m88k-openbsd -mcpu=mc88100 -O0 -verify-machineinstrs -m88k-enable-delay-slot-filler=false | FileCheck %s
; RUN: llc < %s -mtriple=m88k-openbsd -mcpu=mc88110 -O0 -verify-machineinstrs -m88k-enable-delay-slot-filler=false | FileCheck %s

define i32 @f1(i32 %a, i32 %b) {
; CHECK-LABEL: f1:
; CHECK:       cmp %r4, %r2, %r3
; CHECK-NEXT:  ext %r4, %r4, 1<2>
; CHECK-NEXT:  and %r2, %r2, %r4
; CHECK-NEXT:  and.c %r3, %r3, %r4
; CHECK-NEXT:  or %r2, %r2, %r3
; CHECK-NEXT:  jmp %r1
  %cmp = icmp eq i32 %a, %b
  %res = select i1 %cmp, i32 %a, i32 %b
  ret i32 %res
}

define ptr @f2(ptr %a, ptr %b) {
; CHECK-LABEL: f2:
; CHECK:       cmp %r4, %r2, %r3
; CHECK-NEXT:  ext %r4, %r4, 1<2>
; CHECK-NEXT:  and %r2, %r2, %r4
; CHECK-NEXT:  and.c %r3, %r3, %r4
; CHECK-NEXT:  or %r2, %r2, %r3
; CHECK-NEXT:  jmp %r1
  %cmp = icmp eq ptr %a, %b
  %res = select i1 %cmp, ptr %a, ptr %b
  ret ptr %res
}
