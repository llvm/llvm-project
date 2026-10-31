; RUN: not llc -mtriple=powerpc64-ibm-aix-xcoff %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: error: llvm.raw.sections entry '__clangast' is not supported on this target
; CHECK: error: llvm.raw.sections entry '__other' is not supported on this target

!0 = !{!"__clangast", i32 8, i32 0, !"\de\ad\be\ef"}
!1 = !{!"__other", i32 1, i32 0, !"\00"}
!llvm.raw.sections = !{!0, !1}
