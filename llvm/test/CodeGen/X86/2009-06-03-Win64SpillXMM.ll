; RUN: llc -mcpu=generic -mtriple=x86_64-mingw32 < %s | FileCheck %s
; CHECK: subq    $40, %rsp
; CHECK: movaps  %xmm8, 16(%rsp)
; CHECK: movaps  %xmm7, (%rsp)

define i32 @a() nounwind {
entry:
        tail call void asm sideeffect "", "~{xmm7},~{xmm8},~{dirflag},~{fpsr},~{flags}"() nounwind
        ret i32 undef
}
