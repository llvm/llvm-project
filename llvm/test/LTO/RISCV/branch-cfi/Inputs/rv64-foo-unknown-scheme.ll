target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone
define dso_local signext i32 @foo() #0 {
entry:
  ret i32 0
}

attributes #0 = { noinline nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit" }

!llvm.module.flags = !{!0, !1, !2, !4, !5, !6}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"target-abi", !"lp64"}
!2 = !{i32 6, !"riscv-isa", !3}
!3 = !{!"rv64i2p1"}
!4 = !{i32 8, !"cf-protection-branch", i32 1}
!5 = !{i32 1, !"cf-branch-label-scheme", !"unknown-scheme"}
!6 = !{i32 8, !"SmallDataLimit", i32 0}
