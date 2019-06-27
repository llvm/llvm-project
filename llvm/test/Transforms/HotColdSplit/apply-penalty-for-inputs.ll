; REQUIRES: asserts
; RUN: opt -hotcoldsplit -debug-only=hotcoldsplit -hotcoldsplit-threshold=2 -hotcoldsplit-max-params=2 -S < %s -o /dev/null 2>&1 | FileCheck %s

declare void @sink(i32*, i32, i32) cold

@g = global i32 0

define void @foo(i32 %arg) {
  %local = load i32, i32* @g
  br i1 undef, label %cold, label %exit

cold:
  ; CHECK: Applying penalty for splitting: 2
  ; CHECK-NEXT: Applying penalty for: 2 params
  ; CHECK-NEXT: Applying penalty for: 0 outputs/split phis
  ; CHECK-NEXT: penalty = 6
  call void @sink(i32* @g, i32 %arg, i32 %local)
  ret void

exit:
  ret void
}

define void @bar(i32* %p1, i32 %p2, i32 %p3) {
  br i1 undef, label %cold, label %exit

cold:
  ; CHECK: Applying penalty for splitting: 2
  ; CHECK-NEXT: 3 inputs and 0 outputs exceeds parameter limit (2)
  ; CHECK-NEXT: penalty = 2147483647
  call void @sink(i32* %p1, i32 %p2, i32 %p3)
  ret void

exit:
  ret void
}
