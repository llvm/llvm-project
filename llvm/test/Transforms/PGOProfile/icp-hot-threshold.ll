; RUN: opt < %s -passes=pgo-icall-prom -profile-summary-cutoff-hot-icp=200000 -pass-remarks=pgo-icall-prom 2>&1 | FileCheck %s --check-prefix=PASS-REMARK
; RUN: opt < %s -passes=pgo-icall-prom -profile-summary-cutoff-hot-icp=100000 -pass-remarks=pgo-icall-prom 2>&1 | FileCheck %s --check-prefix=FAIL-REMARK

; PASS-REMARK: remark: <unknown>:0:0: Promote indirect call to func4 with count 5 out of 14
; PASS-REMARK: remark: <unknown>:0:0: Promote indirect call to func2 with count 4 out of 9
; PASS-REMARK: remark: <unknown>:0:0: Promote indirect call to func3 with count 3 out of 5

; FAIL-REMARK-NOT: remark: <unknown>:0:0: Promote indirect call to func4
; FAIL-REMARK-NOT: remark: <unknown>:0:0: Promote indirect call to func2
; FAIL-REMARK-NOT: remark: <unknown>:0:0: Promote indirect call to func3

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@foo = common global ptr null, align 8

define i32 @func1() {
entry:
  ret i32 0
}

define i32 @func2() {
entry:
  ret i32 1
}

define i32 @func3() {
entry:
  ret i32 2
}

define i32 @func4() {
entry:
  ret i32 3
}

define i32 @bar() {
entry:
  %tmp = load ptr, ptr @foo, align 8
  %call = call i32 %tmp(), !prof !34
  ret i32 %call
}


!llvm.module.flags = !{!0, !1, !2, !3, !4, !5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{i32 1, !"ProfileSummary", !6}
!6 = !{!7, !8, !9, !10, !11, !12, !13, !14, !15, !16}
!7 = !{!"ProfileFormat", !"InstrProf"}
!8 = !{!"TotalCount", i64 3}
!9 = !{!"MaxCount", i64 1}
!10 = !{!"MaxInternalCount", i64 1}
!11 = !{!"MaxFunctionCount", i64 1}
!12 = !{!"NumCounts", i64 7}
!13 = !{!"NumFunctions", i64 4}
!14 = !{!"IsPartialProfile", i64 0}
!15 = !{!"PartialProfileRatio", double 0.000000e+00}
!16 = !{!"DetailedSummary", !17}
!17 = !{!18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33}
!18 = !{i32 10000, i64 16, i32 1}
!19 = !{i32 100000, i64 15, i32 2}
!20 = !{i32 200000, i64 14, i32 3}
!21 = !{i32 300000, i64 13, i32 4}
!22 = !{i32 400000, i64 12, i32 5}
!23 = !{i32 500000, i64 11, i32 6}
!24 = !{i32 600000, i64 10, i32 7}
!25 = !{i32 700000, i64 9, i32 8}
!26 = !{i32 800000, i64 8, i32 9}
!27 = !{i32 900000, i64 7, i32 10}
!28 = !{i32 950000, i64 6, i32 11}
!29 = !{i32 990000, i64 5, i32 12}
!30 = !{i32 999000, i64 4, i32 13}
!31 = !{i32 999900, i64 3, i32 14}
!32 = !{i32 999990, i64 2, i32 15}
!33 = !{i32 999999, i64 1, i32 16}
!34 = !{!"VP", i32 0, i64 14, i64 7651369219802541373, i64 5, i64 -4377547752858689819, i64 4, i64 -6929281286627296573, i64 3, i64 -2545542355363006406, i64 2}
