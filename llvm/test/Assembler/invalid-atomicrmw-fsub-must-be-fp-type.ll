; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: error: atomicrmw fsub operand must be a floating point type
define void @f(ptr %ptr) {
  atomicrmw fsub ptr %ptr, i32 2 seq_cst
  ret void
}
