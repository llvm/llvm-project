;; -stats requires asserts
; requires: asserts

; RUN: llc -stop-after=block-placement %s -o - | llc --run-pass=static-data-splitter -stats -x mir -o - 2>&1 | FileCheck %s --check-prefix=STAT

; `func_with_hot_jumptable` contains a hot jump table and `func_with_cold_jumptable` contains a cold one. 
; `func_without_entry_count` simulates the functions without profile information (e.g., not instrumented or not profiled),
; it's jump table hotness is unknown and regarded as hot conservatively.
;
; Tests stat messages are expected.
; TODO: Update test to verify section suffixes when target-lowering and assembler changes are implemented.
;
; STAT-DAG: 1 static-data-splitter - Number of cold jump tables seen
; STAT-DAG: 1 static-data-splitter - Number of hot jump tables seen
; STAT-DAG: 1 static-data-splitter - Number of jump tables with unknown hotness

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str.2 = private constant [7 x i8] c"case 3\00"
@.str.3 = private constant [7 x i8] c"case 4\00"
@.str.4 = private constant [7 x i8] c"case 5\00"
@str.9 = private constant [7 x i8] c"case 2\00"
@str.10 = private constant [7 x i8] c"case 1\00"
@str.11 = private constant [8 x i8] c"default\00"

define i32 @func_with_hot_jumptable(i32 %num) !prof !13 {
entry:
  switch i32 %num, label %sw.default [
    i32 1, label %sw.bb
    i32 2, label %sw.bb1
    i32 3, label %sw.bb3
    i32 4, label %sw.bb5
    i32 5, label %sw.bb7
  ], !prof !14

sw.bb:                                            ; preds = %entry
  %puts11 = tail call i32 @puts(ptr @str.10)
  br label %sw.epilog

sw.bb1:                                           ; preds = %entry
  %puts = tail call i32 @puts(ptr @str.9)
  br label %sw.epilog

sw.bb3:                                           ; preds = %entry
  %call4 = tail call i32 (ptr, ...) @printf(ptr @.str.2)
  br label %sw.bb5

sw.bb5:                                           ; preds = %entry, %sw.bb3
  %call6 = tail call i32 (ptr, ...) @printf(ptr @.str.3)
  br label %sw.bb7

sw.bb7:                                           ; preds = %entry, %sw.bb5
  %call8 = tail call i32 (ptr, ...) @printf(ptr @.str.4)
  br label %sw.epilog

sw.default:                                       ; preds = %entry
  %puts12 = tail call i32 @puts(ptr @str.11)
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.default, %sw.bb7, %sw.bb1, %sw.bb
  %div = sdiv i32 %num, 3
  ret i32 %div
}

define void @func_with_cold_jumptable(i32 %num) !prof !15 {
entry:
  switch i32 %num, label %sw.default [
    i32 1, label %sw.bb
    i32 2, label %sw.bb1
    i32 3, label %sw.bb3
    i32 4, label %sw.bb5
    i32 5, label %sw.bb7
  ], !prof !16

sw.bb:                                            ; preds = %entry
  %puts10 = tail call i32 @puts(ptr @str.10)
  br label %sw.epilog

sw.bb1:                                           ; preds = %entry
  %puts = tail call i32 @puts(ptr @str.9)
  br label %sw.epilog

sw.bb3:                                           ; preds = %entry
  %call4 = tail call i32 (ptr, ...) @printf(ptr @.str.2)
  br label %sw.bb5

sw.bb5:                                           ; preds = %entry, %sw.bb3
  %call6 = tail call i32 (ptr, ...) @printf(ptr @.str.3)
  br label %sw.bb7

sw.bb7:                                           ; preds = %entry, %sw.bb5
  %call8 = tail call i32 (ptr, ...) @printf(ptr @.str.4)
  br label %sw.epilog

sw.default:                                       ; preds = %entry
  %puts11 = tail call i32 @puts(ptr @str.11)
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.default, %sw.bb7, %sw.bb1, %sw.bb
  ret void
}

define void @func_without_entry_count(i32 %num) {
entry:
  switch i32 %num, label %sw.default [
    i32 1, label %sw.bb
    i32 2, label %sw.bb1
    i32 3, label %sw.bb3
    i32 4, label %sw.bb5
    i32 5, label %sw.bb7
  ]

sw.bb:                                            ; preds = %entry
  %puts10 = tail call i32 @puts(ptr @str.10)
  br label %sw.epilog

sw.bb1:                                           ; preds = %entry
  %puts = tail call i32 @puts(ptr @str.9)
  br label %sw.epilog

sw.bb3:                                           ; preds = %entry
  %call4 = tail call i32 (ptr, ...) @printf(ptr @.str.2)
  br label %sw.bb5

sw.bb5:                                           ; preds = %entry, %sw.bb3
  %call6 = tail call i32 (ptr, ...) @printf(ptr @.str.3)
  br label %sw.bb7

sw.bb7:                                           ; preds = %entry, %sw.bb5
  %call8 = tail call i32 (ptr, ...) @printf(ptr @.str.4)
  br label %sw.epilog

sw.default:                                       ; preds = %entry
  %puts11 = tail call i32 @puts(ptr @str.11)
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.default, %sw.bb7, %sw.bb1, %sw.bb
  ret void
}

declare i32 @puts(ptr)
declare i32 @printf(ptr, ...)

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 230002}
!4 = !{!"MaxCount", i64 100000}
!5 = !{!"MaxInternalCount", i64 50000}
!6 = !{!"MaxFunctionCount", i64 100000}
!7 = !{!"NumCounts", i64 14}
!8 = !{!"NumFunctions", i64 3}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11, !12}
!11 = !{i32 990000, i64 10000, i32 7}
!12 = !{i32 999999, i64 1, i32 9}
!13 = !{!"function_entry_count", i64 100000}
!14 = !{!"branch_weights", i32 50000, i32 10000, i32 10000, i32 10000, i32 10000, i32 10000}
!15 = !{!"function_entry_count", i64 1}
!16 = !{!"branch_weights", i32 1, i32 0, i32 0, i32 0, i32 0, i32 0}
