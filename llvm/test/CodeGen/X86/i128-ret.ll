; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s --check-prefix=SYSV
; RUN: llc < %s -mtriple=x86_64-win32 | FileCheck %s --check-prefix=WIN64
; SYSV: movq (%rdi), %rax
; SYSV: movq 8(%rdi), %rdx
; WIN64: movq %rcx, %rax
; WIN64: movaps (%rdx), %xmm0
; WIN64: movaps %xmm0, (%rcx)

define i128 @test(ptr%P) {
        %A = load i128, ptr %P
        ret i128 %A
}

