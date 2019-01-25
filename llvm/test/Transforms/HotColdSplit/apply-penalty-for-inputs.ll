; REQUIRES: asserts
; RUN: opt -hotcoldsplit -debug-only=hotcoldsplit -S < %s -o /dev/null 2>&1 | FileCheck %s

declare void @sink(i32*, i32, i32) cold

@g = global i32 0

define void @foo(i32 %arg) {
  %local = load i32, i32* @g
  br i1 undef, label %cold, label %exit

cold:
  ; CHECK: Applying penalty for: 2 inputs
  call void @sink(i32* @g, i32 %arg, i32 %local)
  ret void

exit:
  ret void
}
