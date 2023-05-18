; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; PR21882: confirm we don't crash when high bits are set in a character in a
; metadata string.

; CHECK: !name = !{!0}
!name = !{!0}
; CHECK: !0 = !{!"\80"}
!0 = !{!"\80"}
