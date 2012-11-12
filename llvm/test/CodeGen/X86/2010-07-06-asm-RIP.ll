; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s
; PR 4752

@n = global i32 0                                 ; <i32*> [#uses=2]

define void @f(i32*) nounwind ssp {
  ret void
}

define void @g() nounwind ssp {
entry:
; CHECK: _g:
; CHECK: push $_f$_f
; CHECK: call _f(%rip)
  call void asm sideeffect "push\09$1$1\0A\09call\09${1:a}\0A\09pop\09%edx", "imr,i,~{dirflag},~{fpsr},~{flags},~{memory},~{cc},~{edi},~{esi},~{edx},~{ecx},~{ebx},~{eax}"(i32* @n, void (i32*)* @f) nounwind
  br label %return

return:                                           ; preds = %entry
  ret void
}

