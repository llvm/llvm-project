; RUN: not llvm-as < %s 2>&1 | FileCheck %s

define void @f(i32* %a, i32 %b, i32 %c) {
; CHECK: invalid cmpxchg failure ordering
  %x = cmpxchg i32* %a, i32 %b, i32 %c acq_rel unordered
  ret void
}
