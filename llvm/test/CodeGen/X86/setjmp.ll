; RUN: llc < %s -mtriple=i386-pc-linux | FileCheck --check-prefix=X86 %s
; RUN: llc < %s -mtriple=x86_64-pc-linux | FileCheck --check-prefix=X64 %s
; RUN: llc < %s -mtriple=x86_64-windows-gnu | FileCheck --check-prefix=WIN64 %s

; Verify that @llvm.eh.sjlj.setjmp stores FP and SP into the buffer
; (in addition to IP), so the frontend doesn't need to do it separately.

@buf = internal global [5 x ptr] zeroinitializer

declare i32 @llvm.eh.sjlj.setjmp(ptr) nounwind

define i32 @setjmp_test() nounwind "frame-pointer"="all" {
  %r = tail call i32 @llvm.eh.sjlj.setjmp(ptr @buf)
  ret i32 %r
}

; X86-LABEL: setjmp_test:
; X86:       movl %ebp, buf
; X86:       movl %esp, buf+8

; X64-LABEL: setjmp_test:
; X64:       movq %rbp, buf(%rip)
; X64:       movq %rsp, buf+16(%rip)

; WIN64-LABEL: setjmp_test:
; WIN64:       movq %rbp, buf(%rip)
; WIN64:       movq %rsp, buf+16(%rip)
