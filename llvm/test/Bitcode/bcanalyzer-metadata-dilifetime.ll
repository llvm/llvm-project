; RUN: llvm-as < %s | llvm-bcanalyzer -dump | FileCheck %s

!named = !{!0, !1}

; CHECK:      <METADATA_BLOCK
; CHECK-NEXT: <FRAGMENT/>
!0 = distinct !DIFragment()
; CHECK-NEXT: <LIFETIME op0=0 op1=2/>
!1 = distinct !DILifetime(object: !0, location: !DIExpr())
; CHECK-NEXT: <EXPR op0=0/>
