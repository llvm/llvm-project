; Negative test. The constraint 'l' represents the register 'lo'.
; Check error message in case of invalid usage.
;
; RUN: not llc -march=mips -filetype=obj < %s 2>&1 | FileCheck %s

define void @constraint_l() nounwind {
entry:

; CHECK: error: invalid operand for instruction

  tail call i16 asm sideeffect "addiu $0,$1,$2", "=l,r,r,~{$1}"(i16 0, i16 0)
  ret void
}
