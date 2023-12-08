target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind ssp uwtable
define i32 @test() local_unnamed_addr {
  %1 = tail call i32 (...) @foo()
  %2 = tail call i32 (...) @bar()
  %3 = add nsw i32 %2, %1
  ret i32 %3
}

declare i32 @foo(...) local_unnamed_addr

declare i32 @bar(...) local_unnamed_addr
