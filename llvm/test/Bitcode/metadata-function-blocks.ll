; RUN: llvm-as < %s | llvm-bcanalyzer -dump | FileCheck %s
; Test that metadata only used by a single function is serialized in that
; function instead of in the global pool.
;
; In order to make the bitcode records easy to follow, nodes in this testcase
; are named after the ids they are given in the bitcode.  Nodes local to a
; function have offsets of 100 or 200 (depending on the function) so that they
; remain unique within this textual IR.

; Check for strings in the global pool.
; CHECK:      <METADATA_BLOCK
; CHECK-NEXT:   <STRINGS
; CHECK-SAME:           /> num-strings = 3 {
; CHECK-NEXT:     'named'
; CHECK-NEXT:     'named and foo'
; CHECK-NEXT:     'foo and bar'
; CHECK-NEXT:   }

; Each node gets a new number.  Bottom-up traversal of nodes.
!named = !{!6}

; CHECK-NEXT:   <NODE op0=1/>
!4 = !{!"named"}

; CHECK-NEXT:   <NODE op0=2/>
!5 = !{!"named and foo"}

; CHECK-NEXT:   <NODE op0=1 op1=4 op2=5/>
!6 = !{!"named", !4, !5}

; CHECK-NEXT:   <NODE op0=3/>
!7 = !{!"foo and bar"}

; CHECK-NOT:    <NODE
; CHECK:      </METADATA_BLOCK

; Look at metadata local to @foo, starting with strings.
; CHECK:      <FUNCTION_BLOCK
; CHECK:        <METADATA_BLOCK
; CHECK-NEXT:     <STRINGS
; CHECK-SAME:             /> num-strings = 1 {
; CHECK-NEXT:       'foo'
; CHECK-NEXT:     }

; Function-local nodes start at 9 (strings at 8).
; CHECK-NEXT:     <NODE op0=8/>
!109 = !{!"foo"}

; CHECK-NEXT:     <NODE op0=8 op1=3 op2=9 op3=7 op4=5/>
!110 = !{!"foo", !"foo and bar", !109, !7, !5}

; CHECK-NEXT:   </METADATA_BLOCK
define void @foo() !foo !110 {
  unreachable
}

; Look at metadata local to @bar, starting with strings.
; CHECK:    <FUNCTION_BLOCK
; CHECK:      <METADATA_BLOCK
; CHECK-NEXT:   <STRINGS
; CHECK-SAME:           /> num-strings = 1 {
; CHECK-NEXT:     'bar'
; CHECK-NEXT:   }

; Function-local nodes start at 9 (strings at 8).
; CHECK-NEXT:   <NODE op0=8/>
!209 = !{!"bar"}

; CHECK-NEXT:   <NODE op0=8 op1=3 op2=9 op3=7/>
!210 = !{!"bar", !"foo and bar", !209, !7}

; CHECK-NEXT: </METADATA_BLOCK
define void @bar() {
  unreachable, !bar !210
}
