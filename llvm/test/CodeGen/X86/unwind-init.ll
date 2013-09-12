; RUN: llc -mtriple=x86_64-unknown-linux < %s | FileCheck -check-prefix X8664 %s
; RUN: llc -mtriple=i686-unknown-linux < %s | FileCheck -check-prefix X8632 %s
; Check that all callee-saved registers are saved and restored in functions
; that call __builtin_unwind_init(). This is its undocumented behavior in gcc,
; and it is used in compiling libgcc_eh.
; See also PR8541

declare void @llvm.eh.unwind.init()

define void @calls_unwind_init() {
  call void @llvm.eh.unwind.init()
  ret void
}

; X8664: calls_unwind_init:
; X8664: pushq %rbp
; X8664: pushq %r15
; X8664: pushq %r14
; X8664: pushq %r13
; X8664: pushq %r12
; X8664: pushq %rbx
; X8664: popq %rbx
; X8664: popq %r12
; X8664: popq %r13
; X8664: popq %r14
; X8664: popq %r15

; X8632: calls_unwind_init:
; X8632: pushl %ebp
; X8632: pushl %ebx
; X8632: pushl %edi
; X8632: pushl %esi
; X8632: popl %esi
; X8632: popl %edi
; X8632: popl %ebx
; X8632: popl %ebp
