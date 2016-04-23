; RUN: llvm-as <%s | llvm-bcanalyzer -dump | FileCheck %s
; Check that distinct nodes break uniquing cycles, so that uniqued subgraphs
; are always in post-order.
;
; It may not be immediately obvious why this is an interesting graph.  There
; are three nodes in a cycle, and one of them (!1) is distinct.  Because the
; entry point is !2, a naive post-order traversal would give !3, !1, !2; but
; this means when !3 is parsed the reader will need a forward reference for !2.
; Forward references for uniqued node operands are expensive, whereas they're
; cheap for distinct node operands.  If the distinct node is emitted first, the
; uniqued nodes don't need any forward references at all.

; Nodes in this testcase are numbered to match how they are referenced in
; bitcode.  !3 is referenced as opN=3.

; CHECK:       <DISTINCT_NODE op0=3/>
!1 = distinct !{!3}

; CHECK-NEXT:  <NODE op0=1/>
!2 = !{!1}

; CHECK-NEXT:  <NODE op0=2/>
!3 = !{!2}

; Note: named metadata nodes are not cannot reference null so their operands
; are numbered off-by-one.
; CHECK-NEXT:  <NAME
; CHECK-NEXT:  <NAMED_NODE op0=1/>
!named = !{!2}
