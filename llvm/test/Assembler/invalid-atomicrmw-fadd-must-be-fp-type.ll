; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: error: atomicrmw fadd operand must be a floating point type
define void @f(i32* %ptr) {
  atomicrmw fadd i32* %ptr, i32 2 seq_cst
  ret void
}
