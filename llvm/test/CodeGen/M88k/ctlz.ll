; Test @llvm.ctlz.* instrinsic.
;
; RUN: llc < %s -mtriple=m88k-openbsd -mcpu=mc88100 -m88k-enable-delay-slot-filler=false -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=m88k-openbsd -mcpu=mc88110 -m88k-enable-delay-slot-filler=false -verify-machineinstrs | FileCheck %s

declare i32 @llvm.ctlz.i32(i32, i1)

define i32 @f1(i32 %x) {
; CHECK-LABEL: f1:
; CHECK: ff1 %r2, %r2
; CHECK: xor %r2, %r2, 31
; CHECK: jmp %r1
  %res = call i32 @llvm.ctlz.i32(i32 %x, i1 false)
  ret i32 %res
}

define i32 @f2(i32 %x) {
; CHECK-LABEL: f2:
; CHECK: ff1 %r2, %r2
; CHECK: xor %r2, %r2, 31
; CHECK: jmp %r1
  %res = call i32 @llvm.ctlz.i32(i32 %x, i1 true)
  ret i32 %res
}

define i32 @f3(i32 %x) {
; CHECK-LABEL: f3:
; CHECK: ff1 %r2, %r2
; CHECK-NOT: xor
; CHECK-NOT: sub
; CHECK: jmp %r1
  %cnt = call i32 @llvm.ctlz.i32(i32 %x, i1 true)
  %res = sub i32 31, %cnt
  ret i32 %res
}

define i32 @f4(i32 %x) {
; CHECK-LABEL: f4:
; CHECK: ff1 %r2, %r2
; CHECK-NOT: xor
; CHECK-NOT: sub
; CHECK: jmp %r1
  %cnt = call i32 @llvm.ctlz.i32(i32 %x, i1 true)
  %res = xor i32 %cnt, 31
  ret i32 %res
}
