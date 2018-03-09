; RUN: opt < %s -pgo-memop-opt -pgo-memop-count-threshold=90 -pgo-memop-percent-threshold=15 -S | FileCheck %s --check-prefix=MEMOP_OPT
; RUN: opt < %s -passes=pgo-memop-opt -pgo-memop-count-threshold=90 -pgo-memop-percent-threshold=15 -S | FileCheck %s --check-prefix=MEMOP_OPT
; RUN: opt < %s -pgo-memop-opt -pgo-memop-count-threshold=90 -pgo-memop-percent-threshold=15 -pass-remarks-with-hotness -pass-remarks-output=%t.opt.yaml -S | FileCheck %s --check-prefix=MEMOP_OPT
; RUN: FileCheck %s -input-file=%t.opt.yaml --check-prefix=YAML
; RUN: opt < %s -passes=pgo-memop-opt -pgo-memop-count-threshold=90 -pgo-memop-percent-threshold=15 -pass-remarks-with-hotness -pass-remarks-output=%t.opt.yaml -S | FileCheck %s --check-prefix=MEMOP_OPT
; RUN: FileCheck %s -input-file=%t.opt.yaml --check-prefix=YAML


target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i8* %dst, i8* %src, i8* %dst2, i8* %src2, i32* %a, i32 %n) !prof !27 {
entry:
  br label %for.cond

for.cond:
  %i.0 = phi i32 [ 0, %entry ], [ %inc5, %for.inc4 ]
  %cmp = icmp slt i32 %i.0, %n
  br i1 %cmp, label %for.body, label %for.end6, !prof !28

for.body:
  br label %for.cond1

for.cond1:
  %j.0 = phi i32 [ 0, %for.body ], [ %inc, %for.inc ]
  %idx.ext = sext i32 %i.0 to i64
  %add.ptr = getelementptr inbounds i32, i32* %a, i64 %idx.ext
  %0 = load i32, i32* %add.ptr, align 4
  %cmp2 = icmp slt i32 %j.0, %0
  br i1 %cmp2, label %for.body3, label %for.end, !prof !29

for.body3:
  %add = add nsw i32 %i.0, 1
  %conv = sext i32 %add to i64
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 %conv, i32 1, i1 false), !prof !30
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst2, i8* %src2, i64 %conv, i32 1, i1 false), !prof !31
  br label %for.inc

; MEMOP_OPT:  switch i64 %conv, label %[[DEFAULT_LABEL:.*]] [
; MEMOP_OPT:    i64 1, label %[[CASE_1_LABEL:.*]]
; MEMOP_OPT:  ], !prof [[SWITCH_BW:![0-9]+]] 
; MEMOP_OPT: [[CASE_1_LABEL]]:
; MEMOP_OPT:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 1, i32 1, i1 false)
; MEMOP_OPT:   br label %[[MERGE_LABEL:.*]]
; MEMOP_OPT: [[DEFAULT_LABEL]]:
; MEMOP_OPT:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 %conv, i32 1, i1 false), !prof [[NEWVP:![0-9]+]]
; MEMOP_OPT:   br label %[[MERGE_LABEL]]
; MEMOP_OPT: [[MERGE_LABEL]]:
; MEMOP_OPT:  switch i64 %conv, label %[[DEFAULT_LABEL2:.*]] [
; MEMOP_OPT:    i64 1, label %[[CASE_1_LABEL2:.*]]
; MEMOP_OPT:  ], !prof [[SWITCH_BW:![0-9]+]] 
; MEMOP_OPT: [[CASE_1_LABEL2]]:
; MEMOP_OPT:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst2, i8* %src2, i64 1, i32 1, i1 false)
; MEMOP_OPT:   br label %[[MERGE_LABEL2:.*]]
; MEMOP_OPT: [[DEFAULT_LABEL2]]:
; MEMOP_OPT:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst2, i8* %src2, i64 %conv, i32 1, i1 false), !prof [[NEWVP]]
; MEMOP_OPT:   br label %[[MERGE_LABEL2]]
; MEMOP_OPT: [[MERGE_LABEL2]]:
; MEMOP_OPT:   br label %for.inc
; MEMOP_OPT: [[SWITCH_BW]] = !{!"branch_weights", i32 457, i32 99}
; Should be 457 total left (original total count 556, minus 99 from specialized
; value 1, which is removed from VP array. Also, we only end up with 5 total
; values, since the default max number of promotions is 5 and therefore
; the rest of the values are ignored when extracting the VP metadata.
; MEMOP_OPT: [[NEWVP]] = !{!"VP", i32 1, i64 457, i64 2, i64 88, i64 3, i64 77, i64 9, i64 72, i64 4, i64 66}

for.inc:
  %inc = add nsw i32 %j.0, 1
  br label %for.cond1

for.end:
  br label %for.inc4

for.inc4:
  %inc5 = add nsw i32 %i.0, 1
  br label %for.cond

for.end6:
  ret void
}

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
!27 = !{!"function_entry_count", i64 1}
!28 = !{!"branch_weights", i32 20, i32 1}
!29 = !{!"branch_weights", i32 556, i32 20}
!30 = !{!"VP", i32 1, i64 556, i64 1, i64 99, i64 2, i64 88, i64 3, i64 77, i64 9, i64 72, i64 4, i64 66, i64 5, i64 55, i64 6, i64 44, i64 7, i64 33, i64 8, i64 22}
!31 = !{!"VP", i32 1, i64 556, i64 1, i64 99, i64 2, i64 88, i64 3, i64 77, i64 9, i64 72, i64 4, i64 66, i64 5, i64 55, i64 6, i64 44, i64 7, i64 33, i64 8, i64 22}

declare void @llvm.lifetime.start(i64, i8* nocapture)

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32, i1)

declare void @llvm.lifetime.end(i64, i8* nocapture)

; YAML:      --- !Passed
; YAML-NEXT: Pass:            pgo-memop-opt
; YAML-NEXT: Name:            memopt-opt
; YAML-NEXT: Function:        foo
; YAML-NEXT: Hotness:         0
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'optimized '
; YAML-NEXT:   - Intrinsic:       memcpy
; YAML-NEXT:   - String:          ' with count '
; YAML-NEXT:   - Count:           '99'
; YAML-NEXT:   - String:          ' out of '
; YAML-NEXT:   - Total:           '556'
; YAML-NEXT:   - String:          ' for '
; YAML-NEXT:   - Versions:        '1'
; YAML-NEXT:   - String:          ' versions'
; YAML-NEXT: ...
; YAML-NEXT: --- !Passed
; YAML-NEXT: Pass:            pgo-memop-opt
; YAML-NEXT: Name:            memopt-opt
; YAML-NEXT: Function:        foo
; YAML-NEXT: Hotness:         0
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'optimized '
; YAML-NEXT:   - Intrinsic:       memcpy
; YAML-NEXT:   - String:          ' with count '
; YAML-NEXT:   - Count:           '99'
; YAML-NEXT:   - String:          ' out of '
; YAML-NEXT:   - Total:           '556'
; YAML-NEXT:   - String:          ' for '
; YAML-NEXT:   - Versions:        '1'
; YAML-NEXT:   - String:          ' versions'
; YAML-NEXT: ...
