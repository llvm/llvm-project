; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; | case1 | alloca + align < probe_size
define i32 @foo1(i64 %i) local_unnamed_addr #0 {
; CHECK-LABEL: foo1:
; CHECK:        # %bb.0:
; CHECK-NEXT:	pushq	%rbp
; CHECK-NEXT:	.cfi_def_cfa_offset 16
; CHECK-NEXT:	.cfi_offset %rbp, -16
; CHECK-NEXT:	movq	%rsp, %rbp
; CHECK-NEXT:	.cfi_def_cfa_register %rbp
; CHECK-NEXT:   andq    $-64, %rsp
; CHECK-NEXT:   subq    $832, %rsp                      # imm = 0x340
; CHECK-NEXT:   movl    $1, (%rsp,%rdi,4)
; CHECK-NEXT:   movl    (%rsp), %eax
; CHECK-NEXT:   movq    %rbp, %rsp
; CHECK-NEXT:   popq    %rbp
; CHECK-NEXT:	.cfi_def_cfa %rsp, 8
; CHECK-NEXT:	retq

  %a = alloca i32, i32 200, align 64
  %b = getelementptr inbounds i32, i32* %a, i64 %i
  store volatile i32 1, i32* %b
  %c = load volatile i32, i32* %a
  ret i32 %c
}

; | case2 | alloca > probe_size, align > probe_size
define i32 @foo2(i64 %i) local_unnamed_addr #0 {
; CHECK-LABEL: foo2:
; CHECK:        # %bb.0:
; CHECK-NEXT:	pushq	%rbp
; CHECK-NEXT:	.cfi_def_cfa_offset 16
; CHECK-NEXT:	.cfi_offset %rbp, -16
; CHECK-NEXT:	movq	%rsp, %rbp
; CHECK-NEXT:	.cfi_def_cfa_register %rbp
; CHECK-NEXT:   andq    $-2048, %rsp                    # imm = 0xF800
; CHECK-NEXT:   subq    $2048, %rsp                     # imm = 0x800
; CHECK-NEXT:   movq    $0, (%rsp)
; CHECK-NEXT:   subq    $4096, %rsp                     # imm = 0x1000
; CHECK-NEXT:   movq    $0, (%rsp)
; CHECK-NEXT:   subq    $2048, %rsp                     # imm = 0x800
; CHECK-NEXT:   movl    $1, (%rsp,%rdi,4)
; CHECK-NEXT:   movl    (%rsp), %eax
; CHECK-NEXT:   movq    %rbp, %rsp
; CHECK-NEXT:   popq    %rbp
; CHECK-NEXT:   .cfi_def_cfa %rsp, 8
; CHECK-NEXT:   retq

  %a = alloca i32, i32 2000, align 2048
  %b = getelementptr inbounds i32, i32* %a, i64 %i
  store volatile i32 1, i32* %b
  %c = load volatile i32, i32* %a
  ret i32 %c
}

; | case3 | alloca < probe_size, align < probe_size, alloca + align > probe_size
define i32 @foo3(i64 %i) local_unnamed_addr #0 {
; CHECK-LABEL: foo3:
; CHECK:        # %bb.0:
; CHECK-NEXT:   pushq   %rbp
; CHECK-NEXT:   .cfi_def_cfa_offset 16
; CHECK-NEXT:   .cfi_offset %rbp, -16
; CHECK-NEXT:   movq    %rsp, %rbp
; CHECK-NEXT:   .cfi_def_cfa_register %rbp
; CHECK-NEXT:   andq    $-1024, %rsp                    # imm = 0xFC00
; CHECK-NEXT:   subq    $3072, %rsp                     # imm = 0xC00
; CHECK-NEXT:   movq    $0, (%rsp)
; CHECK-NEXT:   subq    $1024, %rsp                     # imm = 0x400
; CHECK-NEXT:   movl    $1, (%rsp,%rdi,4)
; CHECK-NEXT:   movl    (%rsp), %eax
; CHECK-NEXT:   movq    %rbp, %rsp
; CHECK-NEXT:   popq    %rbp
; CHECK-NEXT:   .cfi_def_cfa %rsp, 8
; CHECK-NEXT:   retq


  %a = alloca i32, i32 1000, align 1024
  %b = getelementptr inbounds i32, i32* %a, i64 %i
  store volatile i32 1, i32* %b
  %c = load volatile i32, i32* %a
  ret i32 %c
}

; | case4 | alloca + probe_size < probe_size, followed by dynamic alloca
define i32 @foo4(i64 %i) local_unnamed_addr #0 {
; CHECK-LABEL: foo4:
; CHECK:        # %bb.0:
; CHECK-NEXT:	pushq	%rbp
; CHECK-NEXT:	.cfi_def_cfa_offset 16
; CHECK-NEXT:	.cfi_offset %rbp, -16
; CHECK-NEXT:	movq	%rsp, %rbp
; CHECK-NEXT:	.cfi_def_cfa_register %rbp
; CHECK-NEXT:	pushq	%rbx
; CHECK-NEXT:	andq	$-64, %rsp
; CHECK-NEXT:	subq	$896, %rsp                      # imm = 0x380
; CHECK-NEXT:	movq	%rsp, %rbx
; CHECK-NEXT:	.cfi_offset %rbx, -24
; CHECK-NEXT:	movl	$1, (%rbx,%rdi,4)
; CHECK-NEXT:	movl	(%rbx), %ecx
; CHECK-NEXT:	movq	%rsp, %rax
; CHECK-NEXT:	leaq	15(,%rcx,4), %rcx
; CHECK-NEXT:	andq	$-16, %rcx
; CHECK-NEXT:	subq	%rcx, %rax
; CHECK-NEXT:	cmpq	%rsp, %rax
; CHECK-NEXT:	jle	.LBB3_3
; CHECK-NEXT:.LBB3_2:                                # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:	movq	$0, (%rsp)
; CHECK-NEXT:	subq	$4096, %rsp                     # imm = 0x1000
; CHECK-NEXT:	cmpq	%rsp, %rax
; CHECK-NEXT:	jg	.LBB3_2
; CHECK-NEXT:.LBB3_3:
; CHECK-NEXT:	andq	$-64, %rax
; CHECK-NEXT:	movq	%rax, %rsp
; CHECK-NEXT:	movl	(%rax), %eax
; CHECK-NEXT:	leaq	-8(%rbp), %rsp
; CHECK-NEXT:	popq	%rbx
; CHECK-NEXT:	popq	%rbp
; CHECK-NEXT:	.cfi_def_cfa %rsp, 8
; CHECK-NEXT:	retq

  %a = alloca i32, i32 200, align 64
  %b = getelementptr inbounds i32, i32* %a, i64 %i
  store volatile i32 1, i32* %b
  %c = load volatile i32, i32* %a
  %d = alloca i32, i32 %c, align 64
  %e = load volatile i32, i32* %d
  ret i32 %e
}

attributes #0 =  {"probe-stack"="inline-asm"}

