; RUN: opt < %s -passes='require<profile-summary>,cgscc(inline)' -inline-threshold=100 -S | FileCheck %s
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; When 'callee' is inlined into caller1 and caller2, the indirect call and vtable
; value profiles of the inlined copy should be scaled based on callers' profiles.
; The indirect call and vtable value profiles in 'callee' should be updated.
define i32 @callee(ptr %0, i32 %1) !prof !19 {
; CHECK-LABEL: define i32 @callee(
; CHECK-SAME: ptr [[TMP0:%.*]], i32 [[TMP1:%.*]]) !prof [[PROF0:![0-9]+]] {
; CHECK-NEXT:    [[TMP3:%.*]] = load ptr, ptr [[TMP0]], align 8, !prof [[PROF1:![0-9]+]]
; CHECK-NEXT:    [[TMP4:%.*]] = getelementptr inbounds i8, ptr [[TMP3]], i64 8
; CHECK-NEXT:    [[TMP5:%.*]] = load ptr, ptr [[TMP4]], align 8
; CHECK-NEXT:    [[TMP6:%.*]] = tail call i32 [[TMP5]](ptr [[TMP0]], i32 [[TMP1]]), !prof [[PROF2:![0-9]+]]
; CHECK-NEXT:    ret i32 [[TMP6]]
;
  %3 = load ptr, ptr %0, !prof !15
  %5 = getelementptr inbounds i8, ptr %3, i64 8
  %6 = load ptr, ptr %5
  %7 = tail call i32 %6(ptr %0, i32 %1), !prof !16
  ret i32 %7
}

define i32 @caller1(i32 %0) !prof !17 {
; CHECK-LABEL: define i32 @caller1(
; CHECK-SAME: i32 [[TMP0:%.*]]) !prof [[PROF3:![0-9]+]] {
; CHECK-NEXT:    [[TMP2:%.*]] = tail call ptr @_Z10createTypei(i32 [[TMP0]])
; CHECK-NEXT:    [[TMP3:%.*]] = load ptr, ptr [[TMP2]], align 8, !prof [[PROF4:![0-9]+]]
; CHECK-NEXT:    [[TMP4:%.*]] = getelementptr inbounds i8, ptr [[TMP3]], i64 8
; CHECK-NEXT:    [[TMP5:%.*]] = load ptr, ptr [[TMP4]], align 8
; CHECK-NEXT:    [[TMP6:%.*]] = tail call i32 [[TMP5]](ptr [[TMP2]], i32 [[TMP0]]), !prof [[PROF5:![0-9]+]]
; CHECK-NEXT:    ret i32 [[TMP6]]
;
  %2 = tail call ptr @_Z10createTypei(i32 %0)
  %3 = tail call i32 @callee(ptr %2, i32 %0)
  ret i32 %3
}

define i32 @caller2(i32 %0) !prof !18  {
; CHECK-LABEL: define i32 @caller2(
; CHECK-SAME: i32 [[TMP0:%.*]]) !prof [[PROF6:![0-9]+]] {
; CHECK-NEXT:    [[TMP2:%.*]] = tail call ptr @_Z10createTypei(i32 [[TMP0]])
; CHECK-NEXT:    [[TMP3:%.*]] = load ptr, ptr [[TMP2]], align 8, !prof [[PROF7:![0-9]+]]
; CHECK-NEXT:    [[TMP4:%.*]] = getelementptr inbounds i8, ptr [[TMP3]], i64 8
; CHECK-NEXT:    [[TMP5:%.*]] = load ptr, ptr [[TMP4]], align 8
; CHECK-NEXT:    [[TMP6:%.*]] = tail call i32 [[TMP5]](ptr [[TMP2]], i32 [[TMP0]]), !prof [[PROF8:![0-9]+]]
; CHECK-NEXT:    ret i32 [[TMP6]]
;
  %2 = tail call ptr @_Z10createTypei(i32 %0)
  %3 = tail call i32 @callee(ptr %2, i32 %0)
  ret i32 %3
}

declare ptr @_Z10createTypei(i32)

!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10}
!3 = !{!"ProfileFormat", !"InstrProf"}
!4 = !{!"TotalCount", i64 10000}
!5 = !{!"MaxCount", i64 10}
!6 = !{!"MaxInternalCount", i64 1}
!7 = !{!"MaxFunctionCount", i64 1000}
!8 = !{!"NumCounts", i64 3}
!9 = !{!"NumFunctions", i64 3}
!10 = !{!"DetailedSummary", !11}
!11 = !{!12, !13, !14}
!12 = !{i32 10000, i64 100, i32 1}
!13 = !{i32 999000, i64 100, i32 1}
!14 = !{i32 999999, i64 1, i32 2}
!15 = !{!"VP", i32 2, i64 1600, i64 321, i64 1000, i64 789, i64 600}
!16 = !{!"VP", i32 0, i64 1600, i64 123, i64 1000, i64 456, i64 600}
!17 = !{!"function_entry_count", i64 1000}
!18 = !{!"function_entry_count", i64 600}
!19 = !{!"function_entry_count", i64 1700}
;.
; CHECK: [[PROF0]] = !{!"function_entry_count", i64 100}
; CHECK: [[PROF1]] = !{!"VP", i32 2, i64 94, i64 321, i64 58, i64 789, i64 35}
; CHECK: [[PROF2]] = !{!"VP", i32 0, i64 94, i64 123, i64 58, i64 456, i64 35}
; CHECK: [[PROF3]] = !{!"function_entry_count", i64 1000}
; CHECK: [[PROF4]] = !{!"VP", i32 2, i64 941, i64 321, i64 588, i64 789, i64 352}
; CHECK: [[PROF5]] = !{!"VP", i32 0, i64 941, i64 123, i64 588, i64 456, i64 352}
; CHECK: [[PROF6]] = !{!"function_entry_count", i64 600}
; CHECK: [[PROF7]] = !{!"VP", i32 2, i64 564, i64 321, i64 352, i64 789, i64 211}
; CHECK: [[PROF8]] = !{!"VP", i32 0, i64 564, i64 123, i64 352, i64 456, i64 211}
;.
