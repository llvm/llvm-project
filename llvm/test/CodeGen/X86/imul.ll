; RUN: llc < %s -mtriple=x86_64-pc-linux-gnu | FileCheck %s --check-prefix=X64
; RUN: llc < %s -mtriple=x86_64-pc-linux-gnux32 | FileCheck %s --check-prefix=X64
; RUN: llc < %s -mtriple=i686-pc-linux | FileCheck %s --check-prefix=X86

define i32 @mul4_32(i32 %A) {
; X64-LABEL: mul4_32:
; X64: leal
; X86-LABEL: mul4_32:
; X86: shll
    %mul = mul i32 %A, 4
    ret i32 %mul
}

define i64 @mul4_64(i64 %A) {
; X64-LABEL: mul4_64:
; X64: leaq
; X86-LABEL: mul4_64:
; X86: shldl
; X86: shll
    %mul = mul i64 %A, 4
    ret i64 %mul
}

define i32 @mul4096_32(i32 %A) {
; X64-LABEL: mul4096_32:
; X64: shll
; X86-LABEL: mul4096_32:
; X86: shll
    %mul = mul i32 %A, 4096
    ret i32 %mul
}

define i64 @mul4096_64(i64 %A) {
; X64-LABEL: mul4096_64:
; X64: shlq
; X86-LABEL: mul4096_64:
; X86: shldl
; X86: shll
    %mul = mul i64 %A, 4096
    ret i64 %mul
}

define i32 @mulmin4096_32(i32 %A) {
; X64-LABEL: mulmin4096_32:
; X64: shll
; X64-NEXT: negl
; X86-LABEL: mulmin4096_32:
; X86: shll
; X86-NEXT: negl
    %mul = mul i32 %A, -4096
    ret i32 %mul
}

define i64 @mulmin4096_64(i64 %A) {
; X64-LABEL: mulmin4096_64:
; X64: shlq
; X64-NEXT: negq
; X86-LABEL: mulmin4096_64:
; X86: shldl
; X86-NEXT: shll
; X86-NEXT: xorl
; X86-NEXT: negl
; X86-NEXT: sbbl
    %mul = mul i64 %A, -4096
    ret i64 %mul
}

define i32 @mul3_32(i32 %A) {
; X64-LABEL: mul3_32:
; X64: leal
; X86-LABEL: mul3_32:
; But why?!
; X86: imull
    %mul = mul i32 %A, 3
    ret i32 %mul
}

define i64 @mul3_64(i64 %A) {
; X64-LABEL: mul3_64:
; X64: leaq
; X86-LABEL: mul3_64:
; X86: mull
; X86-NEXT: imull
    %mul = mul i64 %A, 3
    ret i64 %mul
}

define i32 @mul40_32(i32 %A) {
; X64-LABEL: mul40_32:
; X64: shll
; X64-NEXT: leal
; X86-LABEL: mul40_32:
; X86: shll
; X86-NEXT: leal
    %mul = mul i32 %A, 40
    ret i32 %mul
}

define i64 @mul40_64(i64 %A) {
; X64-LABEL: mul40_64:
; X64: shlq
; X64-NEXT: leaq
; X86-LABEL: mul40_64:
; X86: leal
; X86-NEXT: movl
; X86-NEXT: mull
; X86-NEXT: leal
    %mul = mul i64 %A, 40
    ret i64 %mul
}

define i32 @mul4_32_minsize(i32 %A) minsize {
; X64-LABEL: mul4_32_minsize:
; X64: leal
; X86-LABEL: mul4_32_minsize:
; X86: shll
    %mul = mul i32 %A, 4
    ret i32 %mul
}

define i32 @mul40_32_minsize(i32 %A) minsize {
; X64-LABEL: mul40_32_minsize:
; X64: imull
; X86-LABEL: mul40_32_minsize:
; X86: imull
    %mul = mul i32 %A, 40
    ret i32 %mul
}

define i32 @mul33_32(i32 %A) {
; X64-LABEL: mul33_32:
; X64: shll
; X64-NEXT: leal
; X86-LABEL: mul33_32:
; X86: shll
; X86-NEXT: addl
    %mul = mul i32 %A, 33
    ret i32 %mul
}

define i32 @mul31_32(i32 %A) {
; X64-LABEL: mul31_32:
; X64: shll
; X64-NEXT: subl
; X86-LABEL: mul31_32:
; X86: shll
; X86-NEXT: subl
    %mul = mul i32 %A, 31
    ret i32 %mul
}

define i32 @mul0_32(i32 %A) {
; X64-LABEL: mul0_32:
; X64: xorl	%eax, %eax
    %mul = mul i32 %A, 0
    ret i32 %mul
}

define i32 @mul4294967295_32(i32 %A) {
; X64-LABEL: mul4294967295_32:
; X64: negl	%edi
; X64-NEXT:	movl	%edi, %eax
    %mul = mul i32 %A, 4294967295
    ret i32 %mul
}

define i64 @mul18446744073709551615_64(i64 %A) {
; X64-LABEL: mul18446744073709551615_64:
; X64: negq	%rdi
; X64-NEXT:	movq	%rdi, %rax
    %mul = mul i64 %A, 18446744073709551615
    ret i64 %mul
}
