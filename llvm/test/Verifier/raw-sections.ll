; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: llvm.raw.sections entry must have four operands
; CHECK: llvm.raw.sections entry operand 0 must be a string (section name)

!llvm.raw.sections = !{!0, !1, !2}
!0 = !{!"__clangast", i32 8, i32 1, !"data"}
!1 = !{!"__clangast", i32 8, !"data"}
!2 = !{i32 0, i32 8, i32 1, !"data"}
