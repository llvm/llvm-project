; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: error: atomicrmw xchg operand must be an integer or floating point type
define void @f(i32** %ptr) {
  atomicrmw xchg i32** %ptr, i32* null seq_cst
  ret void
}
