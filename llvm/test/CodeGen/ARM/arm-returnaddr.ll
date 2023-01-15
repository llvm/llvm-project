; RUN: llc < %s -mtriple=arm-apple-ios | FileCheck %s
; RUN: llc < %s -mtriple=thumbv6-apple-ios | FileCheck %s
; RUN: llc < %s -mtriple=arm-apple-ios -regalloc=basic | FileCheck %s
; RUN: llc < %s -mtriple=thumbv6-apple-ios -regalloc=basic | FileCheck %s
; rdar://8015977
; rdar://8020118

define ptr @rt0(i32 %x) nounwind readnone {
entry:
; CHECK-LABEL: rt0:
; CHECK: mov r0, lr
  %0 = tail call ptr @llvm.returnaddress(i32 0)
  ret ptr %0
}

define ptr @rt2() nounwind readnone {
entry:
; CHECK-LABEL: rt2:
; CHECK: ldr r[[R0:[0-9]+]], [r7]
; CHECK: ldr r0, [r[[R0]]]
; CHECK: ldr r0, [r[[R0]], #4]
  %0 = tail call ptr @llvm.returnaddress(i32 2)
  ret ptr %0
}

declare ptr @llvm.returnaddress(i32) nounwind readnone
