 
;This is a negative test. The constant value given for the constraint (N).
;immediate in the range of -65535 to -1 (inclusive).
;Our example uses the positive value 3.
;
; RUN: not llc -mtriple=mipsel < %s  2> %t
; RUN: FileCheck --check-prefix=CHECK-ERRORS < %t %s

define i32 @main() nounwind {
entry:

;CHECK-ERRORS:	error: invalid operand for inline asm constraint 'N'

  tail call i32 asm sideeffect "addiu $0,$1,$2", "=r,r,N"(i32 7, i32 3) nounwind
  ret i32 0
}

