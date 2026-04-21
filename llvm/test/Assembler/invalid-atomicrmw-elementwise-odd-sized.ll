; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

define <5 x i32> @bad_odd_sized_vector(ptr %p, <5 x i32> %v) {
  %old = atomicrmw elementwise add ptr %p, <5 x i32> %v monotonic, align 4
  ret <5 x i32> %old
}

; CHECK: atomicrmw operand must have a power-of-two byte size
