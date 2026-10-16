target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

@large_data = hidden global [105 x i8] zeroinitializer
@small_data = hidden global [50 x i8] zeroinitializer

define void @bar() {
  ret void
}
!llvm.module.flags = !{!0, !1}

!0 = !{i32 9, !"Code Model", i32 3}
!1 = !{i32 9, !"Large Data Threshold", i32 101}
