; Test support for the llvm.returnaddress intrinsic.
; 
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; The current function's return address is in the link register.
define ptr @rt0() norecurse nounwind readnone {
entry:
; CHECK-LABEL: rt0:
; CHECK: lgr  %r2, %r14
; CHECK: br   %r14
  %0 = tail call ptr @llvm.returnaddress(i32 0)
  ret ptr %0
}

declare ptr @llvm.returnaddress(i32) nounwind readnone
