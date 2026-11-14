; RUN: not llc -o /dev/null %s 2>&1 | FileCheck %s
target triple = "x86_64--"

; CHECK: error: could not allocate output register for constraint '{ax}'
define i128 @blup() {
  %v = tail call i128 asm "", "={ax},0"(i128 0)
  ret i128 %v
}

; CHECK: error: could not allocate input reg for constraint 'r'
define void @fp80(x86_fp80) {
  tail call void asm sideeffect "", "r"(x86_fp80 %0)
  ret void
}

; CHECK: error: could not allocate input reg for constraint 'f'
define void @f_constraint_i128(ptr %0) {
  %2 = load i128, ptr %0, align 16
  tail call void asm sideeffect "", "f"(i128 %2)
  ret void
}

; CHECK: error: could not allocate output register for constraint 'r'
define void @r_constraint_v4i128(ptr %0) {
  %2 = alloca <4 x i128>, align 64
  %3 = tail call <4 x i128> asm sideeffect "", "=r"()
  store <4 x i128> %3, ptr %0, align 64
  ret void
}

; There is no integer type of the same size as these FP types to bitcast to.
; CHECK: error: could not allocate output register for constraint '{cr0}'
define x86_fp80 @cr0_fp80(x86_fp80 %0) {
  %2 = tail call x86_fp80 asm "", "={cr0},0"(x86_fp80 %0)
  ret x86_fp80 %2
}

; CHECK: error: could not allocate input reg for constraint '{cr0}'
define void @cr0_v3f32(<3 x float> %0) {
  tail call void asm sideeffect "", "{cr0}"(<3 x float> %0)
  ret void
}
