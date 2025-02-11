;
;This is a negative test. The constant value given for the constraint (K)
;is greater than 16 bits (0x00100000).
;
; RUN: not llc -mtriple=mipsel < %s  2> %t
; RUN: FileCheck --check-prefix=CHECK-ERRORS < %t %s

define i32 @main() nounwind {
entry:

;CHECK-ERRORS:	error: invalid operand for inline asm constraint 'K'

  tail call i32 asm "addu $0,$1,$2", "=r,r,K"(i32 1024, i32 1048576) nounwind
  ret i32 0
}

