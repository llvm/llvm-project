; RUN: not llc --mtriple=loongarch32 < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,LA32
; RUN: not llc --mtriple=loongarch64 < %s 2>&1 | FileCheck %s

define void @constraint_l() {
; CHECK: error: value out of range for constraint 'l'
  tail call void asm sideeffect "lu12i.w $$a0, $0", "l"(i32 32768)
; CHECK: error: value out of range for constraint 'l'
  tail call void asm sideeffect "lu12i.w $$a0, $0", "l"(i32 -32769)
  ret void
}

define void @constraint_I() {
; CHECK: error: value out of range for constraint 'I'
  tail call void asm sideeffect "addi.w $$a0, $$a0, $0", "I"(i32 2048)
; CHECK: error: value out of range for constraint 'I'
  tail call void asm sideeffect "addi.w $$a0, $$a0, $0", "I"(i32 -2049)
  ret void
}

define void @constraint_J() {
; CHECK: error: value out of range for constraint 'J'
  tail call void asm sideeffect "addi.w $$a0, $$a0, $$0", "J"(i32 1)
  ret void
}

define void @constraint_K() {
; CHECK: error: value out of range for constraint 'K'
  tail call void asm sideeffect "andi.w $$a0, $$a0, $0", "K"(i32 4096)
; CHECK: error: value out of range for constraint 'K'
  tail call void asm sideeffect "andi.w $$a0, $$a0, $0", "K"(i32 -1)
  ret void
}

define void @constraint_f() nounwind {
; LA32: error: couldn't allocate input reg for constraint 'f'
  tail call void asm "fadd.s $$fa0, $$fa0, $0", "f"(float 0.0)
; LA32: error: couldn't allocate input reg for constraint 'f'
  tail call void asm "fadd.s $$fa0, $$fa0, $0", "f"(double 0.0)
  ret void
}

define void @constraint_r_vec() nounwind {
; CHECK: error: couldn't allocate input reg for constraint 'r'
  tail call void asm "add.w $$a0, $$a0, $0", "r"(<4 x i32> zeroinitializer)
  ret void
}
