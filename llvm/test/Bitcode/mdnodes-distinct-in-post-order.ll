; RUN: llvm-as <%s | llvm-bcanalyzer -dump | FileCheck %s
; Check that distinct nodes are emitted in post-order to avoid unnecessary
; forward references.

; Nodes in this testcase are numbered to match how they are referenced in
; bitcode.  !3 is referenced as opN=3.

; The leafs should come first (in either order).
; CHECK:       <DISTINCT_NODE/>
; CHECK-NEXT:  <DISTINCT_NODE/>
!1 = distinct !{}
!2 = distinct !{}

; CHECK-NEXT:  <DISTINCT_NODE op0=1 op1=2/>
!3 = distinct !{!1, !2}

; CHECK-NEXT:  <DISTINCT_NODE op0=1 op1=3 op2=2/>
!4 = distinct !{!1, !3, !2}

; Note: named metadata nodes are not cannot reference null so their operands
; are numbered off-by-one.
; CHECK-NEXT:  <NAME
; CHECK-NEXT:  <NAMED_NODE op0=3/>
!named = !{!4}
