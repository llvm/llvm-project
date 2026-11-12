; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+reserve-r14 | FileCheck %s

; Check that user-reserved registers (+reserve-r14) are excluded from
; callee-saved registers on the very first query to getCalleeSavedRegs,
; ensuring correct setjmp preservation and base pointer restoration offsets.

@buf = internal global [5 x ptr] zeroinitializer, align 16

declare i32 @llvm.eh.sjlj.setjmp(ptr)

define i32 @test_setjmp_r14() {
; CHECK-LABEL: test_setjmp_r14:
; CHECK-NOT:   pushq %r14
; CHECK-NOT:   popq %r14
; CHECK:       retq
entry:
  %0 = call i32 @llvm.eh.sjlj.setjmp(ptr @buf)
  ret i32 %0
}

define i32 @test_setjmp_baseptr_r14(i64 %n) {
; CHECK-LABEL: test_setjmp_baseptr_r14:
; CHECK:       movq %rsp, -40(%rbp)
; CHECK:       #EH_SjLj_Setup
; CHECK:       movq -40(%rbp), %rbx
entry:
  %q = alloca i8, align 64
  %s = alloca i8, i64 %n, align 1
  %t = call i32 @llvm.eh.sjlj.setjmp(ptr @buf)
  ret i32 %t
}

define coldcc i32 @test_setjmp_coldcc_r14() {
; CHECK-LABEL: test_setjmp_coldcc_r14:
; CHECK-NOT:   pushq %r14
; CHECK-NOT:   popq %r14
; CHECK:       retq
entry:
  %0 = call i32 @llvm.eh.sjlj.setjmp(ptr @buf)
  ret i32 %0
}
