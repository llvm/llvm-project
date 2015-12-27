; RUN: llc -mcpu=generic -mtriple=x86_64-mingw32 < %s | FileCheck %s
; CHECK: pushq   %rbp
; CHECK: subq    $32, %rsp
; CHECK: leaq    32(%rsp), %rbp
; CHECK: movaps  %xmm8, -16(%rbp)
; CHECK: movaps  %xmm7, -32(%rbp)

define i32 @a() nounwind {
entry:
        tail call void asm sideeffect "", "~{xmm7},~{xmm8},~{dirflag},~{fpsr},~{flags}"() nounwind
        ret i32 undef
}
