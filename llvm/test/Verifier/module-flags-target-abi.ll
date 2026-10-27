; RUN: split-file %s %t
; RUN: not llvm-as < %t/not-string.ll -disable-output 2>&1 | FileCheck %s --check-prefix=NOTSTRING
; RUN: not llvm-as < %t/empty-string.ll -disable-output 2>&1 | FileCheck %s --check-prefix=EMPTY
; RUN: not llvm-as < %t/too-few.ll -disable-output 2>&1 | FileCheck %s --check-prefix=TOOFEW
; RUN: not llvm-as < %t/too-many.ll -disable-output 2>&1 | FileCheck %s --check-prefix=TOOMANY

;--- not-string.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"target-abi", i32 1}
; NOTSTRING: target-abi metadata requires a non-empty string argument

;--- empty-string.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"target-abi", !""}
; EMPTY: target-abi metadata requires a non-empty string argument

;--- too-few.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"target-abi"}
; TOOFEW: incorrect number of operands in module flag

;--- too-many.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"target-abi", !"aapcs", !"extra"}
; TOOMANY: incorrect number of operands in module flag
