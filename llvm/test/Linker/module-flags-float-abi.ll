; RUN: split-file %s %t

;; A module without the flag (target default) links cleanly with one that sets
;; it, and the explicit value is preserved.
; RUN: llvm-link %t/none.ll %t/hard.ll -S -o - | FileCheck %s --check-prefix=HARD
; RUN: llvm-link %t/hard.ll %t/none.ll -S -o - | FileCheck %s --check-prefix=HARD

;; Two modules that agree link cleanly.
; RUN: llvm-link %t/hard.ll %t/hard.ll -S -o - | FileCheck %s --check-prefix=HARD

;; Two modules that disagree are rejected by the 'error' merge behavior.
; RUN: not llvm-link %t/hard.ll %t/soft.ll -S -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFLICT

; HARD: !{i32 1, !"float-abi", !"hard"}
; CONFLICT: linking module flags 'float-abi': IDs have conflicting values

;--- none.ll
define void @f() {
  ret void
}

;--- hard.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"float-abi", !"hard"}

;--- soft.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"float-abi", !"soft"}
