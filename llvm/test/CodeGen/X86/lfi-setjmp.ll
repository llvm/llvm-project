; RUN: llc < %s -mtriple=x86_64_lfi -filetype=obj -o /dev/null
; RUN: llc < %s -mtriple=x86_64_lfi | FileCheck %s

; Check that setjmp on x86_64_lfi does not save and restore reserved registers
; %r14 (sandbox base) and %r15 (context register).

@buf = internal global [5 x ptr] zeroinitializer, align 16

declare i32 @llvm.eh.sjlj.setjmp(ptr)
declare void @llvm.eh.sjlj.longjmp(ptr)

define i32 @test_setjmp() {
; CHECK-LABEL: test_setjmp:
; CHECK-NOT:   pushq %r14
; CHECK-NOT:   pushq %r15
; CHECK-NOT:   popq %r14
; CHECK-NOT:   popq %r15
entry:
  %0 = call i32 @llvm.eh.sjlj.setjmp(ptr @buf)
  ret i32 %0
}

define void @test_longjmp() {
; CHECK-LABEL: test_longjmp:
; CHECK:       jmpq *
entry:
  call void @llvm.eh.sjlj.longjmp(ptr @buf)
  unreachable
}

; Also check that base pointer restoration offset accounts for R14/R15 not being spilled.
define i32 @test_setjmp_baseptr(i64 %n) {
; CHECK-LABEL: test_setjmp_baseptr:
; CHECK:       movq -32(%rbp), %rbx
entry:
  %q = alloca i8, align 64
  %s = alloca i8, i64 %n, align 1
  %t = call i32 @llvm.eh.sjlj.setjmp(ptr @buf)
  ret i32 %t
}

; The reserved registers must not be saved or restored by other calling
; conventions either, e.g. the cold calling convention which is used by
; hot/cold splitting and partial inlining.
define coldcc i32 @test_setjmp_coldcc() {
; CHECK-LABEL: test_setjmp_coldcc:
; CHECK-NOT:   %r14
; CHECK-NOT:   %r15
entry:
  %0 = call i32 @llvm.eh.sjlj.setjmp(ptr @buf)
  ret i32 %0
}

define preserve_mostcc i32 @test_setjmp_preserve_most() {
; CHECK-LABEL: test_setjmp_preserve_most:
; CHECK-NOT:   %r14
; CHECK-NOT:   %r15
entry:
  %0 = call i32 @llvm.eh.sjlj.setjmp(ptr @buf)
  ret i32 %0
}

define preserve_allcc i32 @test_setjmp_preserve_all() {
; CHECK-LABEL: test_setjmp_preserve_all:
; CHECK-NOT:   %r14
; CHECK-NOT:   %r15
entry:
  %0 = call i32 @llvm.eh.sjlj.setjmp(ptr @buf)
  ret i32 %0
}

define x86_regcallcc i32 @test_setjmp_regcall() {
; CHECK-LABEL: test_setjmp_regcall:
; CHECK-NOT:   %r14
; CHECK-NOT:   %r15
entry:
  %0 = call i32 @llvm.eh.sjlj.setjmp(ptr @buf)
  ret i32 %0
}

define anyregcc void @test_setjmp_anyreg() {
; CHECK-LABEL: test_setjmp_anyreg:
; CHECK-NOT:   %r14
; CHECK-NOT:   %r15
entry:
  %0 = call i32 @llvm.eh.sjlj.setjmp(ptr @buf)
  ret void
}

define i32 @test_setjmp_nocsr() #0 {
; CHECK-LABEL: test_setjmp_nocsr:
; CHECK-NOT:   %r14
; CHECK-NOT:   %r15
entry:
  %0 = call i32 @llvm.eh.sjlj.setjmp(ptr @buf)
  ret i32 %0
}

define swifttailcc void @test_setjmp_swifttail(ptr swifterror %err) {
; CHECK-LABEL: test_setjmp_swifttail:
; CHECK-NOT:   %r14
; CHECK-NOT:   %r15
entry:
  %0 = call i32 @llvm.eh.sjlj.setjmp(ptr @buf)
  ret void
}

define swiftcc void @test_setjmp_swift(ptr swifterror %err) {
; CHECK-LABEL: test_setjmp_swift:
; CHECK-NOT:   %r14
; CHECK-NOT:   %r15
entry:
  %0 = call i32 @llvm.eh.sjlj.setjmp(ptr @buf)
  ret void
}

attributes #0 = { "no_caller_saved_registers" }
