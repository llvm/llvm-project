; RUN: llc < %s -march=ppc32 -verify-machineinstrs

; Machine code verifier will call isRegTiedToDefOperand() on /all/ register use
; operands.  We must make sure that the operand flag is found correctly.

; This test case is actually not specific to PowerPC, but the (imm, reg) format
; of PowerPC "m" operands trigger this bug.

define void @memory_asm_operand(i32 %a) {
  ; "m" operand will be represented as:
  ; INLINEASM <es:fake $0>, 10, %R2, 20, -4, %R1
  ; It is difficult to find the flag operand (20) when starting from %R1
  call i32 asm "lbzx $0, $1", "=r,m" (i32 %a)
  ret void
}

