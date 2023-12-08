; Negative test for the 'm' operand code. This operand code is applicable
; for an immediate whic is exact power of 2.

; RUN: not llc -march=mips < %s 2>&1 | FileCheck %s

define i32 @foo() nounwind {
entry:
; CHECK: error: invalid operand in inline asm: 'addiu $0, $1, ${2:y}'
  tail call i32 asm sideeffect "addiu $0, $1, ${2:y}", "=r,r,I"(i32 7, i32 3) ;
  ret i32 0
}
