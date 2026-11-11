; RUN: split-file %s %t

;; A module without the flag links cleanly with one that sets it, and
;; the explicit value is preserved.

; RUN: llvm-link %t/none.ll %t/lldb.ll -S -o - | FileCheck --check-prefix=LLDB %s
; RUN: llvm-link %t/lldb.ll %t/none.ll -S -o - | FileCheck --check-prefix=LLDB %s

;; Two modules that agree link cleanly, with no warning.
; RUN: llvm-link %t/lldb.ll %t/lldb.ll -S -o - 2>&1 | FileCheck --check-prefix=LLDB --implicit-check-not=warning %s

;; Two modules that disagree link successfully and warn while keeping
;; the first module's value.

; RUN: llvm-link %t/lldb.ll %t/sce.ll -S -o - 2>&1 | FileCheck --check-prefix=CONFLICT-LLDB %s
; RUN: llvm-link %t/sce.ll %t/lldb.ll -S -o - 2>&1 | FileCheck --check-prefix=CONFLICT-SCE %s

; LLDB: !{i32 2, !"debugger-tuning", !"lldb"}

; CONFLICT-LLDB: linking module flags 'debugger-tuning': IDs have conflicting values
; CONFLICT-LLDB: !{i32 2, !"debugger-tuning", !"lldb"}

; CONFLICT-SCE: linking module flags 'debugger-tuning': IDs have conflicting values
; CONFLICT-SCE: !{i32 2, !"debugger-tuning", !"sce"}

;--- none.ll
define void @f() {
  ret void
}

;--- lldb.ll
!llvm.module.flags = !{!0}
!0 = !{i32 2, !"debugger-tuning", !"lldb"}

;--- sce.ll
!llvm.module.flags = !{!0}
!0 = !{i32 2, !"debugger-tuning", !"sce"}
