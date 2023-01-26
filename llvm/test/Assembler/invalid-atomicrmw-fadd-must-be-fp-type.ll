; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: error: atomicrmw fadd operand must be a floating point type
define void @f(ptr %ptr) {
  atomicrmw fadd ptr %ptr, i32 2 seq_cst
  ret void
}
