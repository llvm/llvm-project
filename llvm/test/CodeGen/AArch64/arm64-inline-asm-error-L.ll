; RUN: not llc -mtriple=arm64-eabi < %s  2> %t
; RUN: FileCheck --check-prefix=CHECK-ERRORS < %t %s

; Check for at least one invalid constant.
; CHECK-ERRORS:	error: value out of range for constraint 'L'

define i32 @constraint_L(i32 %i, i32 %j) nounwind {
entry:
  %0 = tail call i32 asm sideeffect "eor $0, $1, $2", "=r,r,L"(i32 %i, i64 -1) nounwind
  ret i32 %0
}
