; REQUIRES: asserts
; RUN: opt -passes=hotcoldsplit -debug-only=hotcoldsplit -S < %s -o /dev/null 2>&1 | FileCheck %s

declare void @sink() cold

define void @foo(i32 %arg, i1 %arg2) {
entry:
  br i1 %arg2, label %cold1, label %exit

cold1:
  ; CHECK: Applying bonus for: 4 non-returning terminators
  call void @sink()
  br i1 %arg2, label %cold2, label %cold3

cold2:
  br label %cold4

cold3:
  br label %cold4

cold4:
  unreachable

exit:
  ret void
}
