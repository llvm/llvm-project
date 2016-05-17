; RUN: not llc -march=arm64 < %s  2> %t
; RUN: FileCheck --check-prefix=CHECK-ERRORS < %t %s

; Check for at least one invalid constant.
; CHECK-ERRORS:	error: invalid operand for inline asm constraint 'J'

define i32 @constraint_J(i32 %i, i32 %j) nounwind ssp {
entry:
  %0 = tail call i32 asm sideeffect "sub $0, $1, $2", "=r,r,J"(i32 %i, i32 2) nounwind
  ret i32 %0
}
