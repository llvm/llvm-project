target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@gFoo = internal unnamed_addr global i32 1, align 4

; Function Attrs: norecurse nounwind readonly ssp uwtable
define i32 @foo() local_unnamed_addr {
  %1 = load i32, ptr @gFoo, align 4
  ret i32 %1
}

; Function Attrs: nounwind ssp uwtable
define void @bar() local_unnamed_addr {
  %1 = tail call i32 @rand()
  store i32 %1, ptr @gFoo, align 4
  ret void
}

declare i32 @rand() local_unnamed_addr
