; REQUIRES: asserts
; RUN: opt -hotcoldsplit -debug-only=hotcoldsplit -S < %s -o /dev/null 2>&1 | FileCheck %s

declare void @sink() cold

@g = global i32 0

define i32 @foo(i32 %arg) {
entry:
  br i1 undef, label %cold, label %exit

cold:
  ; CHECK: Applying penalty for: 1 output
  ; CHECK: Applying penalty for: 1 non-region successors
  %local = load i32, i32* @g
  call void @sink()
  br label %exit

exit:
  %p = phi i32 [ %local, %cold ], [ 0, %entry ]
  ret i32 %p
}
