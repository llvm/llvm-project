; RUN: opt -S -lowertypetests -lowertypetests-summary-action=export -lowertypetests-read-summary=%S/Inputs/use-typeid1-typeid2.yaml -lowertypetests-write-summary=%t < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare !type !2 extern_weak void @h(i8)

!cfi.functions = !{!0, !1}

!0 = !{!"h", i8 2, !2}
!1 = !{!"h", i8 0, !2}
!2 = !{i64 0, !"typeid1"}

; CHECK-DAG: @h = alias void (i8), bitcast
