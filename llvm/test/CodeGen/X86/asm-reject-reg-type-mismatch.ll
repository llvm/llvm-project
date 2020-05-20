; RUN: not llc -o /dev/null %s 2>&1 | FileCheck %s
target triple = "x86_64--"

; CHECK: error: couldn't allocate output register for constraint '{ax}'
define i128 @blup() {
  %v = tail call i128 asm "", "={ax},0"(i128 0)
  ret i128 %v
}

; CHECK: error: couldn't allocate input reg for constraint 'r'
define void @fp80(x86_fp80) {
  tail call void asm sideeffect "", "r"(x86_fp80 %0)
  ret void
}

; CHECK: error: couldn't allocate input reg for constraint 'f'
define void @f_constraint_i128(i128* %0) {
  %2 = load i128, i128* %0, align 16
  tail call void asm sideeffect "", "f"(i128 %2)
  ret void
}
