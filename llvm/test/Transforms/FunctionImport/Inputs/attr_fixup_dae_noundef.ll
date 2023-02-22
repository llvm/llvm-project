; This file contains the post-"deadargelim" IR.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @outer(i32 noundef %arg) {
  ; The parameter was originally `noundef %arg`, changed to `poison` by "deadargelim".
  call void @inner(i32 poison)
  ret void
}

; %arg was originally `noundef`, removed by "deadargelim".
define void @inner(i32 %arg) #0 {
  ret void
}

attributes #0 = { noinline }
