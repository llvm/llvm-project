; RUN: llc < %s -mtriple=x86_64-apple-darwin -no-integrated-as | FileCheck %s
; PR 4752

@n = global i32 0                                 ; <ptr> [#uses=2]

define void @f(ptr) nounwind ssp {
  ret void
}

define void @g() nounwind ssp {
entry:
; CHECK: _g:
; CHECK: push $_f$_f
; CHECK: call _f(%rip)
  call void asm sideeffect "push\09$1$1\0A\09call\09${1:a}\0A\09pop\09%edx", "imr,i,~{dirflag},~{fpsr},~{flags},~{memory},~{cc},~{edi},~{esi},~{edx},~{ecx},~{ebx},~{eax}"(ptr @n, ptr @f) nounwind
  br label %return

return:                                           ; preds = %entry
  ret void
}

