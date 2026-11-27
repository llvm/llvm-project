; RUN: split-file %s %t
; RUN: llvm-as < %t/aapcs.ll | llvm-dis | FileCheck %s --check-prefix=AAPCS
; RUN: llvm-as < %t/lp64d.ll | llvm-dis | FileCheck %s --check-prefix=LP64D

;--- aapcs.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"target-abi", !"aapcs"}
; AAPCS: !0 = !{i32 1, !"target-abi", !"aapcs"}

;--- lp64d.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"target-abi", !"lp64d"}
; LP64D: !0 = !{i32 1, !"target-abi", !"lp64d"}
