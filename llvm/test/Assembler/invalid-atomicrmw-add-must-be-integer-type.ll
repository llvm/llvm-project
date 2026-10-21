; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: error: atomicrmw add operand must be an integer or fixed vector of integer type
define void @f(ptr %ptr) {
  atomicrmw add ptr %ptr, float 1.0 seq_cst
  ret void
}
