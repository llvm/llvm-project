; REQUIRES: asserts
; RUN: opt -passes=hotcoldsplit -debug-only=hotcoldsplit -hotcoldsplit-threshold=2 -S < %s -o /dev/null 2>&1 | FileCheck %s

declare void @sink() cold

@g = global i32 0

define i32 @foo(i32 %arg, i1 %arg2) {
entry:
  br i1 %arg2, label %cold, label %exit

cold:
  ; CHECK: Applying penalty for splitting: 2
  ; CHECK-NEXT: Applying penalty for: 1 params
  ; CHECK-NEXT: Applying penalty for: 1 outputs/split phis
  ; CHECK-NEXT: penalty = 7
  %local = load i32, ptr @g
  call void @sink()
  br label %exit

exit:
  %p = phi i32 [ %local, %cold ], [ 0, %entry ]
  ret i32 %p
}
