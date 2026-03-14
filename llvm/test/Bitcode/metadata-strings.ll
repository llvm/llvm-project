; RUN: llvm-as < %s | llvm-bcanalyzer -dump | FileCheck %s

!named = !{!0}

; CHECK:      <METADATA_BLOCK
; CHECK-NEXT: <STRINGS
; CHECK-SAME: /> num-strings = 3 {
; CHECK-NEXT:   'a'
; CHECK-NEXT:   'b'
; CHECK-NEXT:   'c'
; CHECK-NEXT: }
!0 = !{!"a", !"b", !"c"}
