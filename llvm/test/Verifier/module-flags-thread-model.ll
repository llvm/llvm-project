; RUN: split-file %s %t
; RUN: not llvm-as < %t/not-string.ll -disable-output 2>&1 | FileCheck %s --check-prefix=NOTSTRING
; RUN: not llvm-as < %t/bad-value.ll -disable-output 2>&1 | FileCheck %s --check-prefix=BADVALUE
; RUN: not llvm-as < %t/empty-value.ll -disable-output 2>&1 | FileCheck %s --check-prefix=EMPTY
; RUN: not llvm-as < %t/too-few.ll -disable-output 2>&1 | FileCheck %s --check-prefix=TOOFEW
; RUN: not llvm-as < %t/too-many.ll -disable-output 2>&1 | FileCheck %s --check-prefix=TOOMANY
; RUN: not llvm-as < %t/bad-behavior.ll -disable-output 2>&1 | FileCheck %s --check-prefix=BEHAVIOR

;--- not-string.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"thread-model", i32 1}
; NOTSTRING: thread-model metadata requires a string argument

;--- bad-value.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"thread-model", !"multi"}
; BADVALUE: invalid thread-model metadata value

;--- empty-value.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"thread-model", !""}
; EMPTY: invalid thread-model metadata value

;--- too-few.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"thread-model"}
; TOOFEW: incorrect number of operands in module flag

;--- too-many.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"thread-model", !"single", !"extra"}
; TOOMANY: incorrect number of operands in module flag

;--- bad-behavior.ll
!llvm.module.flags = !{!0}
!0 = !{i32 2, !"thread-model", !"single"}
; BEHAVIOR: thread-model module flag must use 'error' merge behavior
