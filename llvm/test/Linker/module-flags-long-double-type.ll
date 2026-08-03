; RUN: split-file %s %t

;; A module with an explicit "long-double-type" flag links cleanly with a module
;; that has no flag, and the explicit value is preserved (both link orders).
; RUN: llvm-link %t/fp128.ll %t/none.ll -S -o - | FileCheck %s --check-prefix=FP128
; RUN: llvm-link %t/none.ll %t/fp128.ll -S -o - | FileCheck %s --check-prefix=FP128

;; Two modules that agree link cleanly.
; RUN: llvm-link %t/fp128.ll %t/fp128.ll -S -o - | FileCheck %s --check-prefix=FP128

;; Two modules that disagree are rejected by the 'error' merge behavior.
; RUN: not llvm-link %t/fp128.ll %t/ppc_fp128.ll -S -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFLICT

; FP128: !{i32 1, !"long-double-type", !"fp128"}
; CONFLICT: linking module flags 'long-double-type': IDs have conflicting values

;--- none.ll
target triple = "powerpc64le-unknown-linux-gnu"
define void @g() {
  ret void
}

;--- fp128.ll
target triple = "powerpc64le-unknown-linux-gnu"
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"long-double-type", !"fp128"}

;--- ppc_fp128.ll
target triple = "powerpc64le-unknown-linux-gnu"
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"long-double-type", !"ppc_fp128"}
