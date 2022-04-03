; RUN: llc -mtriple x86_64-unknown-windows-msvc %s -o - | FileCheck %s -check-prefix CHECK64
; RUN: llc -mtriple i686-windows-msvc %s -o - | FileCheck %s -check-prefix CHECK32

define void @simple(i8* swiftasync %context) "frame-pointer"="all" {
  ret void
}

; CHECK64-LABEL: simple:
; CHECK64: btsq    $60, %rbp
; CHECK64: pushq   %rbp
; CHECK64: pushq   %r14
; CHECK64: leaq    8(%rsp), %rbp
; [...]
; CHECK64: addq    $16, %rsp
; CHECK64: popq    %rbp
; CHECK64: btrq    $60, %rbp
; CHECK64: retq

; CHECK32-LABEL: simple:
; CHECK32: movl    8(%ebp), [[TMP:%.*]]
; CHECK32: movl    [[TMP]], {{.*}}(%ebp)

define void @more_csrs(i8* swiftasync %context) "frame-pointer"="all" {
  call void asm sideeffect "", "~{r15}"()
  ret void
}

; CHECK64-LABEL: more_csrs:
; CHECK64: btsq    $60, %rbp
; CHECK64: pushq   %rbp
; CHECK64: .seh_pushreg %rbp
; CHECK64: pushq   %r14
; CHECK64: .seh_pushreg %r14
; CHECK64: leaq    8(%rsp), %rbp
; CHECK64: subq    $8, %rsp
; CHECK64: pushq   %r15
; CHECK64: .seh_pushreg %r15
; [...]
; CHECK64: popq    %r15
; CHECK64: addq    $16, %rsp
; CHECK64: popq    %rbp
; CHECK64: btrq    $60, %rbp
; CHECK64: retq

declare void @f(i32*)

define void @locals(i8* swiftasync %context) "frame-pointer"="all" {
  %var = alloca i32, i32 10
  call void @f(i32* %var)
  ret void
}

; CHECK64-LABEL: locals:
; CHECK64: btsq    $60, %rbp
; CHECK64: pushq   %rbp
; CHECK64: .seh_pushreg %rbp
; CHECK64: pushq   %r14
; CHECK64: .seh_pushreg %r14
; CHECK64: leaq    8(%rsp), %rbp
; CHECK64: subq    $88, %rsp

; CHECK64: leaq    -48(%rbp), %rcx
; CHECK64: callq   f

; CHECK64: addq    $80, %rsp
; CHECK64: addq    $16, %rsp
; CHECK64: popq    %rbp
; CHECK64: btrq    $60, %rbp
; CHECK64: retq

define void @use_input_context(i8* swiftasync %context, i8** %ptr) "frame-pointer"="all" {
  store i8* %context, i8** %ptr
  ret void
}

; CHECK64-LABEL: use_input_context:
; CHECK64: movq    %r14, (%rcx)

declare i8** @llvm.swift.async.context.addr()

define i8** @context_in_func() "frmae-pointer"="non-leaf" {
  %ptr = call i8** @llvm.swift.async.context.addr()
  ret i8** %ptr
}

; CHECK64-LABEL: context_in_func:
; CHECK64: leaq    -8(%rbp), %rax

; CHECK32-LABEL: context_in_func:
; CHECK32: movl    %esp, %eax

define void @write_frame_context(i8* swiftasync %context, i8* %new_context) "frame-pointer"="non-leaf" {
  %ptr = call i8** @llvm.swift.async.context.addr()
  store i8* %new_context, i8** %ptr
  ret void
}

; CHECK64-LABEL: write_frame_context:
; CHECK64: movq    %rbp, [[TMP:%.*]]
; CHECK64: subq    $8, [[TMP]]
; CHECK64: movq    %rcx, ([[TMP]])

define void @simple_fp_elim(i8* swiftasync %context) "frame-pointer"="non-leaf" {
  ret void
}

; CHECK64-LABEL: simple_fp_elim:
; CHECK64-NOT: btsq
