target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

define i32 @foo(i32) local_unnamed_addr #0 {
  ret i32 10
}

attributes #0 = { noinline }
