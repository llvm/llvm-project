; RUN: llc < %s -mtriple=arm64-eabi | FileCheck %s

define ptr @rt0(i32 %x) nounwind readnone {
entry:
; CHECK-LABEL: rt0:
; CHECK: hint #7
; CHECK: mov x0, x30
; CHECK: ret
  %0 = tail call ptr @llvm.returnaddress(i32 0)
  ret ptr %0
}

define ptr @rt2() nounwind readnone {
entry:
; CHECK-LABEL: rt2:
; CHECK: stp x29, x30, [sp, #-16]!
; CHECK: mov x29, sp
; CHECK: ldr x[[REG:[0-9]+]], [x29]
; CHECK: ldr x[[REG2:[0-9]+]], [x[[REG]]]
; CHECK: ldr x30, [x[[REG2]], #8]
; CHECK: hint #7
; CHECK: mov x0, x30
; CHECK: ldp x29, x30, [sp], #16
; CHECK: ret
  %0 = tail call ptr @llvm.returnaddress(i32 2)
  ret ptr %0
}

declare ptr @llvm.returnaddress(i32) nounwind readnone
