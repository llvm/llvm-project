; RUN: split-file %s %t
; RUN: llvm-link %t/main.ll %t/8.ll
; RUN: not llvm-link %t/main.ll %t/16.ll 2>&1 | FileCheck --check-prefix=CHECK-16 %s

;--- main.ll
; NONE: error: linking module flags 'override-stack-alignment': IDs have conflicting values
; CHECK-16: error: linking module flags 'override-stack-alignment': IDs have conflicting values
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"override-stack-alignment", i32 8}
;--- 8.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"override-stack-alignment", i32 8}
;--- 16.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"override-stack-alignment", i32 16}
