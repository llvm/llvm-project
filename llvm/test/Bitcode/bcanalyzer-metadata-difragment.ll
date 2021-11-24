; RUN: llvm-as < %s | llvm-bcanalyzer -dump | FileCheck %s

!named = !{!0, !1}

; CHECK:      <METADATA_BLOCK
; CHECK-NEXT: <FRAGMENT/>
!0 = distinct !DIFragment()
; CHECK-NEXT: <FRAGMENT/>
!1 = distinct !DIFragment()
