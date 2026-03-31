; RUN: not llvm-as -disable-output %s -o /dev/null 2>&1 | FileCheck %s

define <4 x i32> @bad_fadd(ptr %p, <4 x i32> %v) {
  %old = atomicrmw elementwise fadd ptr %p, <4 x i32> %v monotonic
  ret <4 x i32> %old
}

; CHECK: atomicrmw fadd operand must be a floating point type
