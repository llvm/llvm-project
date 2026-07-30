; RUN: split-file %s %t
; RUN: llvm-as < %t/soft.ll | llvm-dis | FileCheck %s --check-prefix=SOFT
; RUN: llvm-as < %t/hard.ll | llvm-dis | FileCheck %s --check-prefix=HARD

;--- soft.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"float-abi", !"soft"}
; SOFT: !0 = !{i32 1, !"float-abi", !"soft"}

;--- hard.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"float-abi", !"hard"}
; HARD: !0 = !{i32 1, !"float-abi", !"hard"}
