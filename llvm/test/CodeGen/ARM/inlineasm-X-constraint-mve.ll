; RUN: llc -mtriple=thumbv8.1m.main-none-eabi -mattr=+mve,+fp-armv8d16sp < %s -o - | FileCheck %s
;
; Regression for ARMTargetLowering::LowerXConstraint: 64-bit integer vectors
; with MVE (no NEON) must lower the "X" constraint to "w" so the operand is
; allocated in DPR, not GPR.

define void @vector_x_constraint_64(ptr %p) nounwind {
entry:
  %v = load <8 x i8>, ptr %p, align 8
  ; Operand is a 64-bit vector; backend should not treat this as plain "r".
  tail call void asm sideeffect "/* $0 */", "X"(<8 x i8> %v)
  ret void
}

; CHECK-LABEL: vector_x_constraint_64:
; CHECK: vldr d{{[0-9]+}}, [r0]
; CHECK: @APP
; CHECK-NEXT: @ d{{[0-9]+}}
; CHECK-NEXT: @NO_APP
