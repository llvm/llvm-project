; RUN: split-file %s %t

;; A module without the flag (target default) links cleanly with one that sets
;; it, and the explicit value is preserved.
; RUN: llvm-link %t/none.ll %t/sjlj.ll -S -o - | FileCheck %s --check-prefix=SJLJ
; RUN: llvm-link %t/sjlj.ll %t/none.ll -S -o - | FileCheck %s --check-prefix=SJLJ

;; Two modules that agree link cleanly.
; RUN: llvm-link %t/sjlj.ll %t/sjlj.ll -S -o - | FileCheck %s --check-prefix=SJLJ

;; Two modules that disagree are rejected by the 'error' merge behavior.
; RUN: not llvm-link %t/sjlj.ll %t/dwarf.ll -S -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFLICT

; SJLJ: !{i32 1, !"exception-model", !"sjlj"}
; CONFLICT: linking module flags 'exception-model': IDs have conflicting values

;--- none.ll
define void @f() {
  ret void
}

;--- sjlj.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"exception-model", !"sjlj"}

;--- dwarf.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"exception-model", !"dwarf"}
