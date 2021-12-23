; RUN: not llc -mtriple=riscv32 < %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv64 < %s 2>&1 | FileCheck %s

define void @constraint_I() {
; CHECK: error: value out of range for constraint 'I'
  tail call void asm sideeffect "addi a0, a0, $0", "I"(i32 2048)
; CHECK: error: value out of range for constraint 'I'
  tail call void asm sideeffect "addi a0, a0, $0", "I"(i32 -2049)
  ret void
}

define void @constraint_J() {
; CHECK: error: value out of range for constraint 'J'
  tail call void asm sideeffect "addi a0, a0, $0", "J"(i32 1)
  ret void
}

define void @constraint_K() {
; CHECK: error: value out of range for constraint 'K'
  tail call void asm sideeffect "csrwi mstatus, $0", "K"(i32 32)
; CHECK: error: value out of range for constraint 'K'
  tail call void asm sideeffect "csrwi mstatus, $0", "K"(i32 -1)
  ret void
}

define void @constraint_f() nounwind {
; CHECK: error: couldn't allocate input reg for constraint 'f'
  tail call void asm "fadd.s fa0, fa0, $0", "f"(float 0.0)
; CHECK: error: couldn't allocate input reg for constraint 'f'
  tail call void asm "fadd.d fa0, fa0, $0", "f"(double 0.0)
  ret void
}

define void @constraint_r_fixed_vec() nounwind {
; CHECK: error: couldn't allocate input reg for constraint 'r'
  tail call void asm "add a0, a0, $0", "r"(<4 x i32> zeroinitializer)
  ret void
}

define void @constraint_r_scalable_vec() nounwind {
; CHECK: error: couldn't allocate input reg for constraint 'r'
  tail call void asm "add a0, a0, $0", "r"(<vscale x 4 x i32> zeroinitializer)
  ret void
}
