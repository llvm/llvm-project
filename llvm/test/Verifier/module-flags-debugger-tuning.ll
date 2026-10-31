; RUN: split-file %s %t
; RUN: not llvm-as < %t/not-string.ll -disable-output 2>&1 | FileCheck -check-prefix=NOTSTRING %s
; RUN: not llvm-as < %t/empty-string.ll -disable-output 2>&1 | FileCheck -check-prefix=EMPTY %s
; RUN: not llvm-as < %t/too-few.ll -disable-output 2>&1 | FileCheck -check-prefix=TOOFEW %s
; RUN: not llvm-as < %t/too-many.ll -disable-output 2>&1 | FileCheck -check-prefix=TOOMANY %s
; RUN: llvm-as < %t/unrecognized.ll -disable-output

;--- not-string.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"debugger-tuning", i32 1}
; NOTSTRING: debugger-tuning metadata requires a non-empty string argument

;--- empty-string.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"debugger-tuning", !""}
; EMPTY: debugger-tuning metadata requires a non-empty string argument

;--- too-few.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"debugger-tuning"}
; TOOFEW: incorrect number of operands in module flag

;--- too-many.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"debugger-tuning", !"lldb", !"extra"}
; TOOMANY: incorrect number of operands in module flag

;; An unrecognized name is not an error.
;--- unrecognized.ll
!llvm.module.flags = !{!0}
!0 = !{i32 2, !"debugger-tuning", !"wombat"}
