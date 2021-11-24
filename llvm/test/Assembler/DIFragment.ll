; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: !named = !{!0, !1}
!named = !{!0, !1}

; CHECK: !0 = distinct !DIFragment()
!0 = distinct !DIFragment()

; CHECK: !1 = distinct !DIFragment()
!1 = distinct !DIFragment()
