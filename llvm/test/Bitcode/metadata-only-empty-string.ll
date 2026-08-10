; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; CHECK: !named = !{!0}
!named = !{!0}

; CHECK: !0 = !{!""}
!0 = !{!""}
