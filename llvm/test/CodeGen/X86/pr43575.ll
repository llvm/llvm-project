; RUN: llc < %s -mtriple=x86_64-apple-macosx10.14.0 -O0 | FileCheck %s

define void @exit(i32 %status)
; CHECK-LABEL: exit:
; CHECK:       ## %bb.0:
; CHECK:    ## InlineAsm Start
; CHECK:    movq $60, %rax
; CHECK:    syscall
; CHECK:    ## InlineAsm End
; CHECK:    retq
{
    call void asm sideeffect inteldialect "mov rax, 60; syscall", ""()
    ret void
}
