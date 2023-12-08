; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

;; Debug info:
; CHECK:     OpName %[[#FOO:]] "foo"

;; Types:
; CHECK:     %[[#VOID:]] = OpTypeVoid
; CHECK:     %[[#FN:]] = OpTypeFunction %[[#VOID]]

;; Functions:
; CHECK:     %[[#FOO]] = OpFunction %[[#VOID]] None %[[#FN]]
; CHECK-NOT: OpFunctionParameter
;; NOTE: In 2.4, it isn't explicitly written that a function always has a least
;;       one block. In fact, 2.4.11 seems to imply that there are at least two
;;       blocks in functions with a body, but that doesn't make much sense.
;;       However, in order to distinguish between function declaration and
;;       definition, a function needs at least one block, hence why this test
;;       expects one OpLabel + OpReturn.
; CHECK:     OpLabel
; CHECK:     OpReturn
; CHECK-NOT: OpLabel
; CHECK:     OpFunctionEnd
define void @foo() {
  ret void
}
