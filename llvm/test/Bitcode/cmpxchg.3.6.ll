; RUN: llvm-dis < %s.bc | FileCheck %s

define void @f2(i32* %x, i32 %y.orig, i32 %z) {
entry:
  br label %a
b:
  cmpxchg i32* %x, i32 %y, i32 %z acquire acquire
; CHECK: cmpxchg ptr %x, i32 %y, i32 %z acquire acquire
  ret void
a:
  %y = add i32 %y.orig, 1
  br label %a
}
