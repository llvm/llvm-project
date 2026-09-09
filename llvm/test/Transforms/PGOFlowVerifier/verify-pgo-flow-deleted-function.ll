; RUN: opt < %s -verify-pgo-flow \
; RUN:     -passes='function(instcombine),globaldce,function(instcombine)' \
; RUN:     -disable-output 2>&1 | FileCheck %s
;
; InstCombine caches unused @gone. GlobalDCE deletes it. CallbackVH must
; drop function-keyed maps so the later InstCombine walk does not use a
; dangling Function* (that typically crashes as UAF). @bad_callee must
; still be diagnosed afterward.

; CHECK: PGOFlowVerify[EntryCountMismatch] bad_callee: entry=1 vs caller-sum=10
; CHECK-NOT: PGOFlowVerify{{.*}}gone

define internal void @gone() !prof !0 {
entry:
  %a = add i32 0, 0
  ret void
}

define internal i32 @bad_callee(i32 %x) !prof !0 {
entry:
  %y = add i32 %x, 0
  ret i32 %y
}

define i32 @bad_caller(i32 %x) !prof !1 {
entry:
  %r = call i32 @bad_callee(i32 %x), !prof !2
  ret i32 %r
}

!0 = !{!"function_entry_count", i64 1}
!1 = !{!"function_entry_count", i64 1}
!2 = !{!"branch_weights", i32 10}

!llvm.module.flags = !{!10}
!10 = !{i32 1, !"ProfileSummary", !11}
!11 = !{!12, !13, !14, !15, !16, !17, !18, !19}
!12 = !{!"ProfileFormat", !"InstrProf"}
!13 = !{!"TotalCount", i64 12}
!14 = !{!"MaxCount", i64 10}
!15 = !{!"MaxInternalCount", i64 10}
!16 = !{!"MaxFunctionCount", i64 10}
!17 = !{!"NumCounts", i64 3}
!18 = !{!"NumFunctions", i64 3}
!19 = !{!"DetailedSummary", !20}
!20 = !{!21}
!21 = !{i32 10000, i64 10, i32 1}
