; RUN: not llc -march=arm64 < %s  2> %t
; RUN: FileCheck --check-prefix=CHECK-ERRORS < %t %s

; Check for at least one invalid constant.
; CHECK-ERRORS:	error: invalid operand for inline asm constraint 'M'

define i32 @constraint_M(i32 %i, i32 %j) nounwind {
entry:
  %0 = tail call i32 asm sideeffect "movk $0, $1", "=r,M"(i32 305418240) nounwind
  ret i32 %0
}
