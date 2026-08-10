; RUN: llvm-link %s %S/Inputs/unique-fwd-decl-b.ll -S -o - | FileCheck %s

; Test that the arguments of !a and !b get uniqued.
; CHECK: !a = !{!0}
; CHECK: !b = !{!0}

!a = !{!0}
!0 = !{!1}
!1 = !{}
