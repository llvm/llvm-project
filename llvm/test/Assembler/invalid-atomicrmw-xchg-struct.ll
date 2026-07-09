; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: error: atomicrmw xchg operand must be an integer type, a floating-point type, a pointer type, or a fixed vector of any of these types
define void @f(ptr %ptr) {
  atomicrmw xchg ptr %ptr, { i32 } zeroinitializer seq_cst
  ret void
}
