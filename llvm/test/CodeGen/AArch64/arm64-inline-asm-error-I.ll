; RUN: not llc -mtriple=arm64-eabi < %s  2> %t
; RUN: FileCheck --check-prefix=CHECK-ERRORS < %t %s

; Check for at least one invalid constant.
; CHECK-ERRORS:	error: invalid operand for inline asm constraint 'I'

define i32 @constraint_I(i32 %i, i32 %j) nounwind ssp {
entry:
  %0 = tail call i32 asm sideeffect "add $0, $1, $2", "=r,r,I"(i32 %i, i32 4097) nounwind
  ret i32 %0
}
