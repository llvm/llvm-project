;; The stack is marked executable only when the "executable-stack" module flag
;; requests it, not when trampolines are present.

; RUN: rm -rf %t && split-file %s %t
; RUN: llc < %t/exec.ll -mtriple=x86_64-linux | FileCheck %s --check-prefix=EXEC
; RUN: llc < %t/exec.ll -mtriple=amd64-solaris | FileCheck %s --check-prefix=NONE
; RUN: llc < %t/zero.ll -mtriple=x86_64-linux | FileCheck %s --check-prefix=NOEXEC
; RUN: llc < %t/trampoline.ll -mtriple=x86_64-linux | FileCheck %s --check-prefix=NOEXEC

; EXEC:   .section	".note.GNU-stack","x",@progbits
; NOEXEC: .section	".note.GNU-stack","",@progbits
; NONE-NOT: .note.GNU-stack

;--- exec.ll
!llvm.module.flags = !{!0}
!0 = !{i32 7, !"executable-stack", i32 1}

;--- zero.ll
!llvm.module.flags = !{!0}
!0 = !{i32 7, !"executable-stack", i32 0}

;--- trampoline.ll
declare void @nested(ptr nest, i32)

define ptr @f(ptr %tramp, ptr %nest) {
  call void @llvm.init.trampoline(ptr %tramp, ptr @nested, ptr %nest)
  %fp = call ptr @llvm.adjust.trampoline(ptr %tramp)
  ret ptr %fp
}
