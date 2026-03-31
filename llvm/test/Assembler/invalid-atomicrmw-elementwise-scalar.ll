; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

define i32 @bad_scalar(ptr %p, i32 %v) {
  %old = atomicrmw elementwise add ptr %p, i32 %v monotonic
  ret i32 %old
}

; CHECK: atomicrmw elementwise operand must be a fixed vector type
