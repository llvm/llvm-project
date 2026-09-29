; RUN: opt < %s -passes=pgo-memop-opt -pgo-memop-count-threshold=100 -pgo-memop-percent-threshold=10 -S | FileCheck %s

define void @foo(ptr %dst, ptr %src, ptr %dst2, ptr %src2, i64 %n) !prof !27 {
entry:
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 %n, i1 false), !prof !28
  ret void
}

; CHECK:  switch i64 %n, label %[[DEFAULT_LABEL:.*]] [
; CHECK:    i64 0, label %[[CASE_0_LABEL:.*]]
; CHECK:    i64 1, label %[[CASE_1_LABEL:.*]]
; CHECK:    i64 2, label %[[CASE_2_LABEL:.*]]
; CHECK:  ], !prof [[SWITCH_BW:![0-9]+]]
; CHECK: [[CASE_0_LABEL]]:
; CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 0, i1 false)
; CHECK:   br label %[[MERGE_LABEL:.*]]
; CHECK: [[CASE_1_LABEL]]:
; CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 1, i1 false)
; CHECK:   br label %[[MERGE_LABEL:.*]]
; CHECK: [[CASE_2_LABEL]]:
; CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 2, i1 false)
; CHECK:   br label %[[MERGE_LABEL:.*]]
; CHECK: [[DEFAULT_LABEL]]:
; CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 %n, i1 false), !prof [[NEWVP:![0-9]+]]
; CHECK:   br label %[[MERGE_LABEL]]
; CHECK: [[MERGE_LABEL]]:
; CHECK:   ret void

; It should skip range values 9, 17, 33, 65, 129 and promote (up to) three values, 0,
; 1, 2 (not 3), and preserve all unpromoted values in the new VP metadata.
; CHECK: [[SWITCH_BW]] = !{!"branch_weights", i32 524, i32 101, i32 101, i32 101}
; CHECK: [[NEWVP]] = !{!"VP", i32 1, i64 524, i64 9, i64 104, i64 17, i64 103, i64 33, i64 103, i64 65, i64 102, i64 129, i64 102, i64 3, i64 101}

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1)

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 579}
!4 = !{!"MaxCount", i64 556}
!5 = !{!"MaxInternalCount", i64 20}
!6 = !{!"MaxFunctionCount", i64 556}
!7 = !{!"NumCounts", i64 6}
!8 = !{!"NumFunctions", i64 3}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11, !12, !13, !14, !15, !16, !16, !17, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26}
!11 = !{i32 10000, i64 556, i32 1}
!12 = !{i32 100000, i64 556, i32 1}
!13 = !{i32 200000, i64 556, i32 1}
!14 = !{i32 300000, i64 556, i32 1}
!15 = !{i32 400000, i64 556, i32 1}
!16 = !{i32 500000, i64 556, i32 1}
!17 = !{i32 600000, i64 556, i32 1}
!18 = !{i32 700000, i64 556, i32 1}
!19 = !{i32 800000, i64 556, i32 1}
!20 = !{i32 900000, i64 556, i32 1}
!21 = !{i32 950000, i64 556, i32 1}
!22 = !{i32 990000, i64 20, i32 2}
!23 = !{i32 999000, i64 1, i32 5}
!24 = !{i32 999900, i64 1, i32 5}
!25 = !{i32 999990, i64 1, i32 5}
!26 = !{i32 999999, i64 1, i32 5}
!27 = !{!"function_entry_count", i64 827}
!28 = !{!"VP", i32 1, i64 827, i64 9, i64 104, i64 17, i64 103, i64 33, i64 103, i64 65, i64 102, i64 129, i64 102, i64 0, i64 101, i64 1, i64 101, i64 2, i64 101, i64 3, i64 101}
