; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; The current function's frame address is the address of
; the optional back chain slot.
define i8* @fp0() nounwind {
entry:
; CHECK-LABEL: fp0:
; CHECK: la   %r2, 0(%r15)
; CHECK: br   %r14
  %0 = tail call i8* @llvm.frameaddress(i32 0)
  ret i8* %0
}

; Check that the frame address is correct in a presence
; of a stack frame.
define i8* @fp0f() nounwind {
entry:
; CHECK-LABEL: fp0f:
; CHECK: aghi %r15, -168
; CHECK: la   %r2, 168(%r15)
; CHECK: aghi %r15, 168
; CHECK: br   %r14
  %0 = alloca i64, align 8
  %1 = tail call i8* @llvm.frameaddress(i32 0)
  ret i8* %1
}

declare i8* @llvm.frameaddress(i32) nounwind readnone
