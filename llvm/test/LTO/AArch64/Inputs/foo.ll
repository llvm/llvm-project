target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

define dso_local i32 @foo() #0 {
entry:
  ret i32 42
}

attributes #0 = { noinline nounwind optnone uwtable }

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 8, !"branch-target-enforcement", i32 1}
!1 = !{i32 8, !"sign-return-address", i32 1}
!2 = !{i32 8, !"sign-return-address-all", i32 1}
!3 = !{i32 8, !"sign-return-address-with-bkey", i32 1}
