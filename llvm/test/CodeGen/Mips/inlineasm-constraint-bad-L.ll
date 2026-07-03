;
;This is a negative test. The constant value given for the constraint (L)
;is non-zero in the lower 16 bits (0x00100003).
;
; RUN: not llc -mtriple=mipsel < %s  2> %t
; RUN: FileCheck --check-prefix=CHECK-ERRORS < %t %s

define i32 @main() nounwind {
entry:

;CHECK-ERRORS:	error: invalid operand for inline asm constraint 'L'

  tail call i32 asm "addiu $0,$1,$2", "=r,r,L"(i32 7, i32 1048579) nounwind
  ret i32 0
}

