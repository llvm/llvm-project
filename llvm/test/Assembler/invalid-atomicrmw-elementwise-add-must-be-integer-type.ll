; RUN: not llvm-as -disable-output %s | FileCheck %s

define <4 x float> @bad_add(ptr %p, <4 x float> %v) {
  %old = atomicrmw elementwise add ptr %p, <4 x float> %v monotonic
  ret <4 x float> %old
}

; CHECK: atomicrmw add operand must be an integer
