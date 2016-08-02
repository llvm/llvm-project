; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s
!named = !{!0}

; CHECK: [[@LINE+1]]:14: error: use of undefined metadata '!1'
!0 = !{!0, !1}
