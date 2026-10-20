; RUN: llvm-link %s %S/Inputs/unique-fwd-decl-order.ll -S -o - | FileCheck %s
; RUN: llvm-link %S/Inputs/unique-fwd-decl-order.ll %s -S -o - | FileCheck %s

; This test exercises MDNode hashing.  For the nodes to be correctly uniqued,
; the hash of a to-be-created MDNode has to match the hash of an
; operand-just-changed MDNode (with the same operands).
;
; Note that these two assembly files number the nodes identically, even though
; the nodes are in a different order.  This is for the reader's convenience.

; CHECK: !named = !{!0}
!named = !{!0}

; CHECK: !0 = !{!1}
!0 = !{!1}

; CHECK: !1 = !{}
!1 = !{}

; CHECK-NOT: !2
