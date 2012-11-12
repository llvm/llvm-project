;
;This is a negative test. The constant value given for the constraint (O).
;signed 15 bit immediate (+- 16383).
;Our example uses the positive value 16384.
;
; RUN: not llc -march=mipsel < %s  2> %t
; RUN: FileCheck --check-prefix=CHECK-ERRORS < %t %s

define i32 @main() nounwind {
entry:

;CHECK-ERRORS:	error: invalid operand for inline asm constraint 'O'

  tail call i32 asm sideeffect "addi $0,$1,$2", "=r,r,O"(i32 undef, i32 16384) nounwind
  ret i32 0
}
