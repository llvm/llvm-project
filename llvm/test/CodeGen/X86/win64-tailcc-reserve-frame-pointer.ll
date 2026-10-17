; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s | FileCheck %s

; With a tail-call reserve, frame-pointer-relative and stack-pointer-relative
; addressing must agree. In the first two functions the old return address is
; loaded from the entry RSP, the new one is stored 48 bytes below it, and the
; last store overwrites the old return-address slot. The third function has a
; reserve (112) large enough that the frame-pointer offset is computed from the
; locals alone (32), not from locals plus reserve (clipped at 128).

declare tailcc void @g(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)

define tailcc void @with_fp(i64 %a, i64 %b) "frame-pointer"="all" {
; CHECK-LABEL: with_fp:
; CHECK:         subq $48, %rsp
; CHECK-NEXT:    .seh_stackalloc 48
; CHECK-NEXT:    pushq %rbp
; CHECK-NEXT:    .seh_pushreg %rbp
; CHECK-NEXT:    subq $32, %rsp
; CHECK-NEXT:    .seh_stackalloc 32
; CHECK-NEXT:    leaq 32(%rsp), %rbp
; CHECK-NEXT:    .seh_setframe %rbp, 32
; CHECK-NEXT:    .seh_endprologue
; CHECK:         movq 56(%rbp), %rax
; CHECK-NEXT:    movq %rax, 8(%rbp)
; CHECK:         movq %rdx, 56(%rbp)
  %x = alloca [2 x i64], align 16
  %y = alloca i64, align 8
  store volatile i64 %a, ptr %x, align 8
  store volatile i64 %b, ptr %y, align 8
  musttail call tailcc void @g(i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b)
  ret void
}

define tailcc void @without_fp(i64 %a, i64 %b) "frame-pointer"="none" {
; CHECK-LABEL: without_fp:
; CHECK:         subq $48, %rsp
; CHECK-NEXT:    .seh_stackalloc 48
; CHECK-NEXT:    subq $40, %rsp
; CHECK-NEXT:    .seh_stackalloc 40
; CHECK-NEXT:    .seh_endprologue
; CHECK:         movq 88(%rsp), %rax
; CHECK-NEXT:    movq %rax, 40(%rsp)
; CHECK:         movq %rdx, 88(%rsp)
  %x = alloca [2 x i64], align 16
  %y = alloca i64, align 8
  store volatile i64 %a, ptr %x, align 8
  store volatile i64 %b, ptr %y, align 8
  musttail call tailcc void @g(i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b)
  ret void
}

declare tailcc void @g18(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)

define tailcc void @with_fp_large_reserve(i64 %a, i64 %b) "frame-pointer"="all" {
; CHECK-LABEL: with_fp_large_reserve:
; CHECK:         subq $112, %rsp
; CHECK-NEXT:    .seh_stackalloc 112
; CHECK-NEXT:    pushq %rbp
; CHECK-NEXT:    .seh_pushreg %rbp
; CHECK-NEXT:    subq $32, %rsp
; CHECK-NEXT:    .seh_stackalloc 32
; CHECK-NEXT:    leaq 32(%rsp), %rbp
; CHECK-NEXT:    .seh_setframe %rbp, 32
; CHECK-NEXT:    .seh_endprologue
; CHECK-NEXT:    movq %rcx, -32(%rbp)
; CHECK-NEXT:    movq %rdx, -8(%rbp)
; CHECK-NEXT:    movq 120(%rbp), %rax
; CHECK-NEXT:    movq %rax, 8(%rbp)
; CHECK:         movq %rdx, 120(%rbp)
  %x = alloca [2 x i64], align 16
  %y = alloca i64, align 8
  store volatile i64 %a, ptr %x, align 8
  store volatile i64 %b, ptr %y, align 8
  musttail call tailcc void @g18(i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b)
  ret void
}
