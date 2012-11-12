; RUN: llc < %s -mtriple=arm-apple-ios | FileCheck %s
; RUN: llc < %s -mtriple=thumbv6-apple-ios | FileCheck %s
; RUN: llc < %s -mtriple=arm-apple-ios -regalloc=basic | FileCheck %s
; RUN: llc < %s -mtriple=thumbv6-apple-ios -regalloc=basic | FileCheck %s
; rdar://8015977
; rdar://8020118

define i8* @rt0(i32 %x) nounwind readnone {
entry:
; CHECK: rt0:
; CHECK: {r7, lr}
; CHECK: mov r0, lr
  %0 = tail call i8* @llvm.returnaddress(i32 0)
  ret i8* %0
}

define i8* @rt2() nounwind readnone {
entry:
; CHECK: rt2:
; CHECK: {r7, lr}
; CHECK: ldr r[[R0:[0-9]+]], [r7]
; CHECK: ldr r0, [r0]
; CHECK: ldr r0, [r0, #4]
  %0 = tail call i8* @llvm.returnaddress(i32 2)
  ret i8* %0
}

declare i8* @llvm.returnaddress(i32) nounwind readnone
