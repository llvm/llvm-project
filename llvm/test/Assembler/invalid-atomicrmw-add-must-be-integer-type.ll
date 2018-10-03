; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: error: atomicrmw add operand must be an integer
define void @f(float* %ptr) {
  atomicrmw add float* %ptr, float 1.0 seq_cst
  ret void
}
