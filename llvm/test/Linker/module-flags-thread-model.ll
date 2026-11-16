; RUN: split-file %s %t

;; A module without the flag links cleanly with one that sets it, and
;; the explicit value is preserved.

; RUN: llvm-link %t/none.ll %t/single.ll -S -o - | FileCheck %s --check-prefix=SINGLE
; RUN: llvm-link %t/single.ll %t/none.ll -S -o - | FileCheck %s --check-prefix=SINGLE

;; Two modules that agree link cleanly.
; RUN: llvm-link %t/single.ll %t/single.ll -S -o - | FileCheck %s --check-prefix=SINGLE

;; Two modules that disagree are rejected by the 'error' merge behavior.
; RUN: not llvm-link %t/single.ll %t/posix.ll -S -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFLICT

; SINGLE: !{i32 1, !"thread-model", !"single"}
; CONFLICT: linking module flags 'thread-model': IDs have conflicting values

;--- none.ll
define void @f() {
  ret void
}

;--- single.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"thread-model", !"single"}

;--- posix.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"thread-model", !"posix"}
