target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = local_unnamed_addr global i32 10, align 4
@B = local_unnamed_addr constant i32 20, align 4

; Function Attrs: norecurse nounwind readonly uwtable
define i32 @foo() local_unnamed_addr #0 {
  %1 = load i32, ptr @B, align 4
  %2 = load i32, ptr @A, align 4
  %3 = add nsw i32 %2, %1
  ret i32 %3
}

; Function Attrs: norecurse nounwind readnone uwtable
define i32 @bar() local_unnamed_addr {
  ret i32 42
}

attributes #0 = { noinline }
