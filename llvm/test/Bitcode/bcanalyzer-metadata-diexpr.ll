; RUN: llvm-as < %s | llvm-bcanalyzer -dump | FileCheck %s

!named = !{!0, !1}

; CHECK:      <METADATA_BLOCK
; CHECK-NEXT: <EXPR op0=0/>
!0 = !DIExpr()
; CHECK-NEXT: <EXPR op0=0 op1=15/>
!1 = !DIExpr(DIOpAdd())
