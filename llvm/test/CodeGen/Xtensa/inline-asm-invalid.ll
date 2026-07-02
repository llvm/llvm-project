; RUN: not llc -mtriple=xtensa -mcpu=generic < %s 2>&1 | FileCheck %s

; CHECK: error: could not allocate input reg for constraint 'f'
define void @constraint_f() nounwind {
  tail call void asm "add.s f0, f1, $0", "f"(float 0.0)
  ret void
}
