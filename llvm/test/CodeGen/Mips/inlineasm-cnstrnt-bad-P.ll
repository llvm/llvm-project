;
; This is a negative test. The constant value given for the constraint (P).
; A constant in the range of 1 to 655535 inclusive.
; Our example uses the positive value 655536.
;
; RUN: not llc -march=mipsel < %s  2> %t
; RUN: FileCheck --check-prefix=CHECK-ERRORS < %t %s

define i32 @main() nounwind {
entry:

;CHECK-ERRORS:	error: invalid operand for inline asm constraint 'P'

  tail call i32 asm sideeffect "addi $0,$1,$2", "=r,r,P"(i32 undef, i32 655536) nounwind
  ret i32 0
}
