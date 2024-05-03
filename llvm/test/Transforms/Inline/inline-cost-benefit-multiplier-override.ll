; RUN: opt < %s -passes='require<profile-summary>,cgscc(inline)' -pass-remarks=inline -pass-remarks-missed=inline -inline-savings-multiplier=4 -inline-savings-profitable-multiplier=5 -S 2>&1| FileCheck %s

; Test that inline cost benefit multipler could be configured from command line.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; @inlined_caleee is inlined by cost-benefit anlysis.
; @not_inlined_callee is not inlined, decided by cost-benefit analysis
; CHECK: remark: <unknown>:0:0: 'inlined_callee' inlined into 'caller' with (cost=always): benefit over cost
; CHECK: remark: <unknown>:0:0: 'not_inlined_callee' not inlined into 'caller' because it should never be inlined (cost=never): cost over benefit

define i32 @inlined_callee(i32 %c) !prof !17 {
entry:
  %mul = mul nsw i32 %c, %c
  ret i32 %mul
}

define i32 @not_inlined_callee(i32 %c) !prof !18 {
entry:
  %add = add nsw i32 %c, 2
  ret i32 %add
}

define i32 @caller(i32 %a, i32 %c)  !prof !15 {
entry:
  %rem = srem i32 %a, 3
  %cmp = icmp eq i32 %rem, 0
  br i1 %cmp, label %if.then, label %if.end, !prof !16

if.then:
; CHECK-LABEL: if.then:
; CHECK-NOT: call i32 @inlined_callee
  %call = tail call i32 @inlined_callee(i32 %c) "inline-cycle-savings-for-test"="26" "inline-runtime-cost-for-test"="1"
  br label %return

if.end:
; CHECK-LABEL: if.end:
; CHECK: call i32 @not_inlined_callee
  %call1 = tail call i32 @not_inlined_callee(i32 %c) "inline-cycle-savings-for-test"="19" "inline-runtime-cost-for-test"="1"
  br label %return

return:
  %retval.0 = phi i32 [ %call, %if.then ], [ %call1, %if.end ]
  ret i32 %retval.0
}

!llvm.module.flags = !{!1}

!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10}
!3 = !{!"ProfileFormat", !"InstrProf"}
!4 = !{!"TotalCount", i64 10000}
!5 = !{!"MaxCount", i64 1000}
!6 = !{!"MaxInternalCount", i64 1}
!7 = !{!"MaxFunctionCount", i64 1000}
!8 = !{!"NumCounts", i64 3}
!9 = !{!"NumFunctions", i64 3}
!10 = !{!"DetailedSummary", !11}
!11 = !{!12, !13, !14}
!12 = !{i32 10000, i64 100, i32 1}
!13 = !{i32 990000, i64 100, i32 1}
!14 = !{i32 999999, i64 1, i32 2}
!15 = !{!"function_entry_count", i64 500}
!16 = !{!"branch_weights", i32 1, i32 2}
!17 = !{!"function_entry_count", i64 200}
!18 = !{!"function_entry_count", i64 400}
