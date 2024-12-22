;
;This is a negative test. The constant value given for the constraint (J)
;is non-zero (3).
;
; RUN: not llc -march=mipsel < %s  2> %t
; RUN: FileCheck --check-prefix=CHECK-ERRORS < %t %s

define i32 @main() nounwind {
entry:

;CHECK-ERRORS:	error: invalid operand for inline asm constraint 'J'

  tail call i32 asm "addiu $0,$1,$2", "=r,r,J"(i32 1024, i32 3) nounwind
  ret i32 0
}

