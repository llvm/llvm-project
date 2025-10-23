; RUN: not opt -passes=verify < %s 2>&1 | FileCheck %s
; CHECK: atomic store operand must have integer, pointer, floating point, or vector type!
; CHECK: atomic load operand must have integer, pointer, floating point, or vector type!

%ty = type { i32 };

define void @foo(ptr %P, %ty %v) {
  store atomic %ty %v, ptr %P unordered, align 8
  ret void
}

define %ty @bar(ptr %P) {
  %v = load atomic %ty, ptr %P unordered, align 8
  ret %ty %v
}
