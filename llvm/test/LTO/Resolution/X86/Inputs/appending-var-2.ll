target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"foo" = type { i8 }

@llvm.used = appending global [1 x ptr] [ptr @bar], section "llvm.metadata"

; Function Attrs: norecurse nounwind readnone uwtable
define dso_local i32 @bar(ptr nocapture readnone %this) align 2 !type !0 {
entry:
  ret i32 0
}

!0 = !{i64 0, !"typeid1"}
