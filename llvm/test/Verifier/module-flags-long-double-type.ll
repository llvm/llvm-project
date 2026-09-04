; RUN: split-file %s %t
; RUN: not llvm-as < %t/not-string.ll -disable-output 2>&1 | FileCheck %s --check-prefix=NOTSTRING
; RUN: not llvm-as < %t/bad-value.ll -disable-output 2>&1 | FileCheck %s --check-prefix=BADVALUE
; RUN: not llvm-as < %t/empty-value.ll -disable-output 2>&1 | FileCheck %s --check-prefix=EMPTY
; RUN: not llvm-as < %t/too-few.ll -disable-output 2>&1 | FileCheck %s --check-prefix=TOOFEW
; RUN: not llvm-as < %t/too-many.ll -disable-output 2>&1 | FileCheck %s --check-prefix=TOOMANY
; RUN: not llvm-as < %t/bad-behavior.ll -disable-output 2>&1 | FileCheck %s --check-prefix=BEHAVIOR

;--- not-string.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"long-double-type", i32 1}
; NOTSTRING: long-double-type metadata requires a string argument

;--- bad-value.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"long-double-type", !"bogus"}
; BADVALUE: invalid long-double-type metadata value

;--- empty-value.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"long-double-type", !""}
; EMPTY: invalid long-double-type metadata value

;--- too-few.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"long-double-type"}
; TOOFEW: incorrect number of operands in module flag

;--- too-many.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"long-double-type", !"fp128", !"extra"}
; TOOMANY: incorrect number of operands in module flag

;--- bad-behavior.ll
!llvm.module.flags = !{!0}
!0 = !{i32 2, !"long-double-type", !"fp128"}
; BEHAVIOR: long-double-type module flag must use 'error' merge behavior
