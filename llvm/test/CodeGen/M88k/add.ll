; Test addition/subtraction.
;
; RUN: llc < %s -mtriple=m88k-openbsd -mcpu=mc88100 -verify-machineinstrs -m88k-enable-delay-slot-filler=false | FileCheck %s
; RUN: llc < %s -mtriple=m88k-openbsd -mcpu=mc88110 -verify-machineinstrs -m88k-enable-delay-slot-filler=false | FileCheck %s

define i32 @f1(i32 %a) {
; CHECK-LABEL: f1:
; CHECK:       addu %r2, %r2, 512
; CHECK-NEXT:  jmp %r1
  %sum = add i32 %a, 512
  ret i32 %sum
}

define i32 @f2(i32 %a) {
; CHECK-LABEL: f2:
; CHECK:       addu %r2, %r2, 512
; CHECK-NEXT:  jmp %r1
  %sum = add i32 512, %a
  ret i32 %sum
}
