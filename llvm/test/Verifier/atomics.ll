; RUN: not opt -passes=verify < %s 2>&1 | FileCheck %s

; CHECK: atomic store operand must have integer, pointer, or floating point type!
; CHECK: atomic load operand must have integer, pointer, or floating point type!

define void @foo(ptr %P, x86_mmx %v) {
  store atomic x86_mmx %v, ptr %P unordered, align 8
  ret void
}

define x86_mmx @bar(ptr %P) {
  %v = load atomic x86_mmx, ptr %P unordered, align 8
  ret x86_mmx %v
}
