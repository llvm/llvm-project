; RUN: not opt -passes=verify < %s 2>&1 | FileCheck %s

; CHECK: atomic store operand must have integer, pointer, or floating point type!
; CHECK: atomic load operand must have integer, pointer, or floating point type!

define void @foo(ptr %P, <1 x i64> %v) {
  store atomic <1 x i64> %v, ptr %P unordered, align 8
  ret void
}

define <1 x i64> @bar(ptr %P) {
  %v = load atomic <1 x i64>, ptr %P unordered, align 8
  ret <1 x i64> %v
}
