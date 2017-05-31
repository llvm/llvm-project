; ModuleID = 'thinlto-function-summary-callgraph2.ll'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define void @func() #0 !prof !2 {
entry:
    ret void
}

!2 = !{!"function_entry_count", i64 1}
