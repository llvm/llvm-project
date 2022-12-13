; RUN: not llvm-as < %s 2>&1 | FileCheck %s

define void @f(ptr %a, i32 %b, i32 %c) {
; CHECK: invalid cmpxchg success ordering
  %x = cmpxchg ptr %a, i32 %b, i32 %c unordered monotonic
  ret void
}
