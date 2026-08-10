; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-win32 | FileCheck %s
; CHECK: movq ([[A0:%rdi|%rcx]]), %rax
; CHECK: movq 8([[A0]]), %rdx

define i128 @test(ptr%P) {
        %A = load i128, ptr %P
        ret i128 %A
}

