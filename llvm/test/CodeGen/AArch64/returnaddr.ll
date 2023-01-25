; RUN: llc -o - %s -mtriple=arm64-apple-ios7.0 | FileCheck %s

define ptr @rt0(i32 %x) nounwind readnone {
entry:
; CHECK-LABEL: rt0:
; CHECK: hint #7
; CHECK: mov x0, x30
  %0 = tail call ptr @llvm.returnaddress(i32 0)
  ret ptr %0
}

define ptr @rt2() nounwind readnone {
entry:
; CHECK-LABEL: rt2:
; CHECK: ldr x[[reg:[0-9]+]], [x29]
; CHECK: ldr x[[reg]], [x[[reg]]]
; CHECK: ldr x30, [x[[reg]], #8]
; CHECK: hint #7
; CHECK: mov x0, x30
  %0 = tail call ptr @llvm.returnaddress(i32 2)
  ret ptr %0
}

declare ptr @llvm.returnaddress(i32) nounwind readnone
