; RUN: llc -mtriple x86_64-unknown-windows-msvc %s -o - | FileCheck %s -check-prefix CHECK64
; RUN: llc -mtriple i686-windows-msvc %s -o - | FileCheck %s -check-prefix CHECK32

define void @simple(ptr swiftasync %context) "frame-pointer"="all" {
  ret void
}

; CHECK64-LABEL: simple:
; CHECK64: pushq   %rbp
; CHECK64: pushq   %rax
; CHECK64: movq    %rsp, %rbp
; CHECK64: movq    %r14, (%rbp)
; CHECK64: addq    $8, %rsp
; CHECK64: popq    %rbp
; CHECK64: retq

; CHECK32-LABEL: simple:
; CHECK32: movl    8(%ebp), [[TMP:%.*]]
; CHECK32: movl    [[TMP]], {{.*}}(%ebp)

define void @more_csrs(ptr swiftasync %context) "frame-pointer"="all" {
  call void asm sideeffect "", "~{r15}"()
  ret void
}

; CHECK64-LABEL: more_csrs:
; CHECK64: pushq   %rbp
; CHECK64: .seh_pushreg %rbp
; CHECK64: pushq   %r15
; CHECK64: .seh_pushreg %r15
; CHECK64: pushq   %rax
; CHECK64: .seh_stackalloc 8
; CHECK64: movq    %rsp, %rbp
; CHECK64: .seh_setframe %rbp, 0
; CHECK64: .seh_endprologue
; CHECK64: movq    %r14, (%rbp)
; [...]
; CHECK64: addq    $8, %rsp
; CHECK64: popq    %r15
; CHECK64: popq    %rbp
; CHECK64: retq

declare void @f(ptr)

define void @locals(ptr swiftasync %context) "frame-pointer"="all" {
  %var = alloca i32, i32 10
  call void @f(ptr %var)
  ret void
}

; CHECK64-LABEL: locals:
; CHECK64: pushq   %rbp
; CHECK64: .seh_pushreg %rbp
; CHECK64: subq    $80, %rsp
; CHECK64: movq	%r14, -8(%rbp)

; CHECK64: leaq    -48(%rbp), %rcx
; CHECK64: callq   f

; CHECK64: addq    $80, %rsp
; CHECK64: popq    %rbp
; CHECK64: retq

define void @use_input_context(ptr swiftasync %context, ptr %ptr) "frame-pointer"="all" {
  store ptr %context, ptr %ptr
  ret void
}

; CHECK64-LABEL: use_input_context:
; CHECK64: movq    %r14, (%rcx)

declare ptr @llvm.swift.async.context.addr()

define ptr @context_in_func() "frmae-pointer"="non-leaf" {
  %ptr = call ptr @llvm.swift.async.context.addr()
  ret ptr %ptr
}

; CHECK64-LABEL: context_in_func:
; CHECK64: movq   %rsp, %rax

; CHECK32-LABEL: context_in_func:
; CHECK32: movl    %esp, %eax

define void @write_frame_context(ptr swiftasync %context, ptr %new_context) "frame-pointer"="non-leaf" {
  %ptr = call ptr @llvm.swift.async.context.addr()
  store ptr %new_context, ptr %ptr
  ret void
}

; CHECK64-LABEL: write_frame_context:
; CHECK64: movq    %rcx, (%rsp)

define void @simple_fp_elim(ptr swiftasync %context) "frame-pointer"="non-leaf" {
  ret void
}

; CHECK64-LABEL: simple_fp_elim:
; CHECK64-NOT: btsq

define void @manylocals_and_overwritten_context(ptr swiftasync %context, ptr %new_context) "frame-pointer"="all" {
  %ptr = call ptr @llvm.swift.async.context.addr()
  store ptr %new_context, ptr %ptr
  %var1 = alloca i64, i64 1
  call void @f(ptr %var1)
  %var2 = alloca i64, i64 16
  call void @f(ptr %var2)
  %ptr2 = call ptr @llvm.swift.async.context.addr()
  store ptr %new_context, ptr %ptr2
  ret void
}

; CHECK64-LABEL: manylocals_and_overwritten_context:
; CHECK64:       pushq	%rbp
; CHECK64:       subq	$184, %rsp
; CHECK64:       leaq	128(%rsp), %rbp
; CHECK64:       movq	%rcx, %rsi
; CHECK64:       movq	%rcx, 48(%rbp)
; CHECK64:       callq	f
; CHECK64:       callq	f
; CHECK64:       movq	%rsi, 48(%rbp)
