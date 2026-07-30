target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%class.A = type { [16 x i8] }

define void @bar(ptr) {
  ret void
}

!llvm.module.flags = !{!1}

!1 = !{i32 1, !"EnableSplitLTOUnit", i32 0}
