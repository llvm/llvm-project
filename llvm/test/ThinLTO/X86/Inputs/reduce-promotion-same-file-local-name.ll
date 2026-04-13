target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

source_filename = "foo.c"

define dso_local i32 @m2(i32 noundef %0) local_unnamed_addr {
  %2 = tail call fastcc i32 @foo(i32 noundef %0)
  %3 = shl nsw i32 %0, 1
  %4 = tail call fastcc i32 @foo(i32 noundef %3)
  %5 = add nsw i32 %4, %2
  ret i32 %5
}

define internal fastcc range(i32 -2147483647, -2147483648) i32 @foo(i32 noundef %0) unnamed_addr #1 {
  %2 = add nsw i32 %0, 5
  %3 = sdiv i32 %2, %0
  ret i32 %3
}

attributes #1 = { noinline }
