; RUN: split-file %s %t
; RUN: llvm-link %t/main.ll %t/match.ll
; RUN: not llvm-link %t/main.ll %t/mismatch.ll 2>&1 | \
; RUN:   FileCheck --check-prefix=CHECK-MISMATCH %s

; CHECK-MISMATCH: error: linking module flags 'warn-stack-size': IDs have conflicting values

;--- main.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"warn-stack-size", i32 80}
;--- match.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"warn-stack-size", i32 80}
;--- mismatch.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"warn-stack-size", i32 81}
