; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-win32 | FileCheck %s
; CHECK: movq ([[A0:%rdi|%rcx]]), %rax
; CHECK: movq 8([[A0]]), %rdx

define i128 @test(i128 *%P) {
        %A = load i128* %P
        ret i128 %A
}

