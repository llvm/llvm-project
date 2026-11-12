; RUN: split-file %s %t
; RUN: llvm-link %t/base.ll %t/mode.ll 2>&1 | FileCheck --check-prefix=MODE %s
; RUN: llvm-link %t/base.ll %t/mechanism.ll 2>&1 | FileCheck --check-prefix=MECHANISM %s
; RUN: llvm-link %t/base.ll %t/same.ll

; MODE: warning: linking module flags 'cfguard': IDs have conflicting values
; MECHANISM: warning: linking module flags 'cfguard-mechanism': IDs have conflicting values

;--- base.ll
define void @foo() {
  ret void
}
!llvm.module.flags = !{!0,!1}
!0 = !{i32 2, !"cfguard", i32 1}
!1 = !{i32 2, !"cfguard-mechanism", i32 1}

;--- mode.ll
define void @bar() {
  ret void
}
!llvm.module.flags = !{!0,!1}
!0 = !{i32 2, !"cfguard", i32 2}
!1 = !{i32 2, !"cfguard-mechanism", i32 1}

;--- mechanism.ll
define void @bar() {
  ret void
}
!llvm.module.flags = !{!0,!1}
!0 = !{i32 2, !"cfguard", i32 1}
!1 = !{i32 2, !"cfguard-mechanism", i32 2}

;--- same.ll
define void @bar() {
  ret void
}
!llvm.module.flags = !{!0,!1}
!0 = !{i32 2, !"cfguard", i32 1}
!1 = !{i32 2, !"cfguard-mechanism", i32 1}
