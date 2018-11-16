target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind ssp uwtable
define i32 @test() local_unnamed_addr {
  %1 = tail call i32 (...) @foo()
  ret i32 %1
}

declare i32 @foo(...) local_unnamed_addr
