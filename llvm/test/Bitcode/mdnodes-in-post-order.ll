; RUN: llvm-as <%s -bitcode-mdindex-threshold=0 | llvm-bcanalyzer -dump | FileCheck %s
; Check that nodes are emitted in post-order to minimize the need for temporary
; nodes.  The graph structure is designed to foil naive implementations of
; iteratitive post-order traersals: the leaves, !3 and !4, are reachable from
; the entry node, !6, as well as from !5.  There is one leaf on either side to
; be sure it tickles bugs whether operands are visited forward or reverse.

; Nodes in this testcase are numbered to match how they are referenced in
; bitcode.  !3 is referenced as opN=3.

; We don't care about the order of the strings (or of !3 and !4).  Let's just
; make sure the strings are first and make it clear that there are two of them.
; CHECK:       <STRINGS {{.*}} num-strings = 2 {
; CHECK-NEXT:    'leaf
; CHECK-NEXT:    'leaf
; CHECK-NEXT:  }

; Before the records we emit an offset to the index for the block
; CHECK-NEXT:   <INDEX_OFFSET

; The leafs should come first (in either order).
; CHECK-NEXT:  <NODE op0=1/>
; CHECK-NEXT:  <NODE op0=2/>
!3 = !{!"leaf3"}
!4 = !{!"leaf4"}

; CHECK-NEXT:  <NODE op0=3 op1=4/>
!5 = !{!3, !4}

; CHECK-NEXT:  <NODE op0=3 op1=5 op2=4/>
!6 = !{!3, !5, !4}

; Before the named records we emit the index containing the position of the
; previously emitted records
; CHECK-NEXT:   <INDEX {{.*}} (offset match)

; Note: named metadata nodes are not cannot reference null so their operands
; are numbered off-by-one.
; CHECK-NEXT:  <NAME
; CHECK-NEXT:  <NAMED_NODE op0=5/>
!named = !{!6}
