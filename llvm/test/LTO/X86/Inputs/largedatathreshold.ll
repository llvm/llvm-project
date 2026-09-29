target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @bar() {
  ret void
}
!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"Code Model", i32 3}
!1 = !{i32 1, !"Large Data Threshold", i32 101}
