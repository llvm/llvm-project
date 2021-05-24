; ModuleID = 'test2.cc'
source_filename = "test2.cc"
target datalayout = "e-m:w-p:64:64-i32:32-i64:64-i128:128-n32:64-S128"
target triple = "arm64-pc-windows-msvc19.11.0"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @"?f@@YAXXZ"() #0 {
entry:
  ret void
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "min-legal-vector-width"="0" "frame-pointer"="none" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+neon" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 2}
!1 = !{!"clang version 9.0.0 "}
