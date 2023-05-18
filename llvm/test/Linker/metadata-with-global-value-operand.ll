; RUN: llvm-link -S -o - %s %S/Inputs/metadata-with-global-value-operand.ll | FileCheck %s
; This test confirms that the !{null} from the second module doesn't get mapped
; onto the abandoned !{ptr @var} node from this module.

; CHECK: @var = global
@var = global i1 false

; CHECK: !named.vars = !{!0}
; CHECK: !named.null = !{!1}
!named.vars = !{!0}

; CHECK: !0 = !{ptr @var}
; CHECK: !1 = !{null}
!0 = !{ptr @var}
