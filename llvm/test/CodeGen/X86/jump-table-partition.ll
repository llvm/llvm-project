; -stats requires asserts
; requires: asserts

; COM: Investigate test failure with fuchsia environment and re-enable the test.
; UNSUPPORTED: target={{.*}}-fuchsia

; Stop after 'finalize-isel' for simpler MIR, and lower the minimum number of
; jump table entries so 'switch' needs fewer cases to generate a jump table.
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -stop-after=finalize-isel -min-jump-table-entries=2 %s -o %t.mir
; RUN: llc -mtriple=x86_64-unknown-linux-gnu --run-pass=static-data-splitter -stats -x mir %t.mir -o - 2>&1 | FileCheck %s --check-prefix=STAT

; Tests stat messages are expected.
; COM: Update test to verify section suffixes when target-lowering and assembler changes are implemented.
; COM: Also run static-data-splitter pass with -static-data-default-hotness=cold and check data section suffix.
 
; STAT-DAG: 2 static-data-splitter - Number of cold jump tables seen
; STAT-DAG: 2 static-data-splitter - Number of hot jump tables seen
; STAT-DAG: 1 static-data-splitter - Number of jump tables with unknown hotness

; In function @foo, the 2 switch instructions to jt0.* and jt1.* get lowered to hot jump tables,
; and the 2 switch instructions to jt2.* and jt3.* get lowered to cold jump tables.

; @func_without_profile doesn't have profiles. It's jump table hotness is unknown.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@str.9 = private constant [7 x i8] c".str.9\00"
@str.10 = private constant [8 x i8] c".str.10\00"
@str.11 = private constant [8 x i8] c".str.11\00"

@case2 = private constant [7 x i8] c"case 2\00"
@case1 = private constant [7 x i8] c"case 1\00"
@default = private constant [8 x i8] c"default\00"
@jt3 = private constant [4 x i8] c"jt3\00"

define i32 @foo(i32 %num) !prof !13 {
entry:
  %mod3 = sdiv i32 %num, 3
  switch i32 %mod3, label %jt0.default [
    i32 1, label %jt0.bb1
    i32 2, label %jt0.bb2
  ], !prof !14

jt0.bb1:
  call i32 @puts(ptr @case1)
  br label %jt0.epilog

jt0.bb2:
  call i32 @puts(ptr @case2)
  br label %jt0.epilog

jt0.default:
  call i32 @puts(ptr @default)
  br label %jt0.epilog

jt0.epilog:
  %zero = icmp eq i32 %num, 0
  br i1 %zero, label %cold, label %hot, !prof !15

cold:
 %c2 = call i32 @transform(i32 %num)
  switch i32 %c2, label %jt2.default [
    i32 1, label %jt2.bb1
    i32 2, label %jt2.bb2
  ], !prof !14

jt2.bb1:
  call i32 @puts(ptr @case1)
  br label %jt1.epilog

jt2.bb2:
  call i32 @puts(ptr @case2)
  br label %jt1.epilog

jt2.default:
  call i32 @puts(ptr @default)
  br label %jt2.epilog

jt2.epilog:
  %c2cmp = icmp ne i32 %c2, 0
  br i1 %c2cmp, label %return, label %jt3.prologue, !prof !16

hot:
  %c1 = call i32 @compute(i32 %num)
  switch i32 %c1, label %jt1.default [
    i32 1, label %jt1.bb1
    i32 2, label %jt1.bb2
  ], !prof !14

jt1.bb1:
  call i32 @puts(ptr @case1)
  br label %jt1.epilog

jt1.bb2:
  call i32 @puts(ptr @case2)
  br label %jt1.epilog

jt1.default:
  call i32 @puts(ptr @default)
  br label %jt1.epilog

jt1.epilog:
  br label %return

jt3.prologue:
  %c3 = call i32 @cleanup(i32 %num)
  switch i32 %c3, label %jt3.default [
    i32 1, label %jt3.bb1
    i32 2, label %jt3.bb2
  ], !prof !14

jt3.bb1:
  call i32 @puts(ptr @case1)
  br label %jt3.epilog

jt3.bb2:
  call i32 @puts(ptr @case2)
  br label %jt3.epilog

jt3.default:
  call i32 @puts(ptr @default)
  br label %jt3.epilog

jt3.epilog:
  call i32 @puts(ptr @jt3)
  br label %return

return:
  ret i32 %mod3
}

define void @func_without_profile(i32 %num) {
entry:
  switch i32 %num, label %sw.default [
    i32 1, label %sw.bb
    i32 2, label %sw.bb1
  ]

sw.bb:
  call i32 @puts(ptr @str.10)
  br label %sw.epilog

sw.bb1: 
  call i32 @puts(ptr @str.9)
  br label %sw.epilog

sw.default:
  call i32 @puts(ptr @str.11)
  br label %sw.epilog

sw.epilog:                                       
  ret void
}

declare i32 @puts(ptr)
declare i32 @printf(ptr, ...)
declare i32 @compute(i32)
declare i32 @transform(i32)
declare i32 @cleanup(i32)

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
!14 = !{!"branch_weights", i32 60000, i32 20000, i32 20000}
!15 = !{!"branch_weights", i32 1, i32 99999}
!16 = !{!"branch_weights", i32 99998, i32 1}
