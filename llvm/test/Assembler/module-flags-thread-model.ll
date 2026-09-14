; RUN: split-file %s %t
; RUN: llvm-as < %t/posix.ll | llvm-dis | FileCheck %s --check-prefix=POSIX
; RUN: llvm-as < %t/single.ll | llvm-dis | FileCheck %s --check-prefix=SINGLE

;--- posix.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"thread-model", !"posix"}
; POSIX: !0 = !{i32 1, !"thread-model", !"posix"}

;--- single.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"thread-model", !"single"}
; SINGLE: !0 = !{i32 1, !"thread-model", !"single"}
