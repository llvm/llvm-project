; RUN: not llc -mtriple=riscv32 < %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv64 < %s 2>&1 | FileCheck %s

define void @constraint_I() {
; CHECK: error: invalid operand for inline asm constraint 'I'
  tail call void asm sideeffect "addi a0, a0, $0", "I"(i32 2048)
; CHECK: error: invalid operand for inline asm constraint 'I'
  tail call void asm sideeffect "addi a0, a0, $0", "I"(i32 -2049)
  ret void
}

define void @constraint_J() {
; CHECK: error: invalid operand for inline asm constraint 'J'
  tail call void asm sideeffect "addi a0, a0, $0", "J"(i32 1)
  ret void
}

define void @constraint_K() {
; CHECK: error: invalid operand for inline asm constraint 'K'
  tail call void asm sideeffect "csrwi mstatus, $0", "K"(i32 32)
; CHECK: error: invalid operand for inline asm constraint 'K'
  tail call void asm sideeffect "csrwi mstatus, $0", "K"(i32 -1)
  ret void
}
