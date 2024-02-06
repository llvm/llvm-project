; Test support for the llvm.returnaddress intrinsic with packed-stack.

; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; The current function's return address is in the link register.
attributes #0 = { nounwind "packed-stack" "backchain" "use-soft-float"="true" }
define ptr @rt0() #0 {
entry:
; CHECK-LABEL: rt0:
; CHECK: lgr  %r2, %r14
; CHECK: br   %r14
  %0 = tail call ptr @llvm.returnaddress(i32 0)
  ret ptr %0
}

; Check the caller's return address.
define ptr @rtcaller() #0 {
entry:
; CHECK-LABEL: rtcaller:
; CHECK: lg   %r1, 152(%r15)
; CHECK  lg   %r2, 136(%r1)
; CHECK: br   %r14
  %0 = tail call ptr @llvm.returnaddress(i32 1)
  ret ptr %0
}

; Check the caller's caller's return address.
define ptr @rtcallercaller() #0 {
entry:
; CHECK-LABEL: rtcallercaller:
; CHECK: lg   %r1, 152(%r15)
; CHECK: lg   %r1, 152(%r1)
; CHECK  lg   %r2, 136(%r1)
; CHECK: br   %r14
  %0 = tail call ptr @llvm.returnaddress(i32 2)
  ret ptr %0
}

declare ptr @llvm.returnaddress(i32) nounwind readnone
