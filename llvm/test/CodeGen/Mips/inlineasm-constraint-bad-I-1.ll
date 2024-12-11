;
;This is a negative test. The constant value given for the constraint
;is greater than 16 bits.
;
; RUN: not llc -march=mipsel < %s  2> %t
; RUN: FileCheck --check-prefix=CHECK-ERRORS < %t %s

define i32 @main() nounwind {
entry:

;CHECK-ERRORS:	error: invalid operand for inline asm constraint 'I'
  tail call i32 asm sideeffect "addiu $0,$1,$2", "=r,r,I"(i32 7, i32 1048576) nounwind
  ret i32 0
}

