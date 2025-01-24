; -stats requires asserts
; REQUIRES: asserts

; Stop after 'finalize-isel' for simpler MIR, and lower the minimum number of
; jump table entries so 'switch' needs fewer cases to generate a jump table.
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -stop-after=finalize-isel -min-jump-table-entries=2 %s -o %t.mir
; RUN: llc -mtriple=x86_64-unknown-linux-gnu --run-pass=static-data-splitter -stats -x mir %t.mir -o - 2>&1 | FileCheck %s --check-prefix=STAT

 ; Tests stat messages are expected.
; STAT: 2 static-data-splitter - Number of cold jump tables seen
; STAT: 2 static-data-splitter - Number of hot jump tables seen
; STAT: 1 static-data-splitter - Number of jump tables with unknown hotness

; When 'partition-static-data-sections' is enabled, static data splitter pass will
; categorize jump tables and assembly printer will place hot jump tables in the
; `.rodata.hot`-prefixed section, and cold ones in the `.rodata.unlikely`-prefixed section.
; Section names will optionally have `.<func>` if -function-sections is enabled.
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-split-machine-functions -partition-static-data-sections=true -function-sections=true -min-jump-table-entries=2 -unique-section-names=false  %s -o - 2>&1 | FileCheck %s --check-prefixes=LINEAR,JT
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-split-machine-functions -partition-static-data-sections=true -function-sections=true -min-jump-table-entries=2  %s -o - 2>&1 | FileCheck %s --check-prefixes=FUNC,JT,DEFAULTHOT
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-split-machine-functions -partition-static-data-sections=true -function-sections=false -min-jump-table-entries=2 %s -o - 2>&1 | FileCheck %s --check-prefixes=FUNCLESS,JT --implicit-check-not=unique

; Tests that `-static-data-default-hotness` can override hotness for data with
; unknown hotness.
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-split-machine-functions -partition-static-data-sections=true -min-jump-table-entries=2 -static-data-default-hotness=cold -function-sections=true %s -o - 2>&1 | FileCheck %s --check-prefixes=FUNC,JT,DEFAULTCOLD

; LINEAR:    .section .rodata.hot,"a",@progbits,unique,2
; FUNC:     .section .rodata.hot.foo,"a",@progbits
; FUNCLESS: .section .rodata.hot,"a",@progbits
; JT: .LJTI0_0:
; JT: .LJTI0_2:
; LINEAR:    	.section	.rodata.unlikely,"a",@progbits,unique,3
; FUNC:       .section .rodata.unlikely.foo,"a",@progbits
; FUNCLESS:   .section .rodata.unlikely,"a",@progbits
; JT: .LJTI0_1:
; JT: .LJTI0_3:
; DEFAULTHOT: .section .rodata.hot.func_without_entry_count,"a",@progbits
; DEFAULTHOT: .LJTI1_0:
; FUNCLESS: .section .rodata.hot,"a",@progbits
; FUNCLESS: .LJTI1_0:

; DEFAULTCOLD: .section .rodata.unlikely.func_without_entry_count,"a",@progbits
; DEFAULTCOLD: .LJTI1_0:

; @foo has four jump tables, jt0, jt1, jt2 and jt3 in the input basic block
; order; jt0 and jt2 are hot, and jt1 and jt3 are cold.
;
; @func_with_hot_jt is a function with one entry count, and a hot loop using a
; jump table.

; @func_without_entry_count simulates the functions without profile information
; (e.g., not instrumented or not profiled), it's jump table hotness is unknown
; and regarded as hot conservatively.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@str.9 = private constant [7 x i8] c".str.9\00"
@str.10 = private constant [8 x i8] c".str.10\00"
@str.11 = private constant [8 x i8] c".str.11\00"

@case2 = private constant [7 x i8] c"case 2\00"
@case1 = private constant [7 x i8] c"case 1\00"
@default = private constant [8 x i8] c"default\00"
@jt3 = private constant [3 x i8] c"jt\00"

; jt0 and jt2 are hot. jt1 and jt3 are cold.
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
  br i1 %zero, label %hot, label %cold, !prof !17

hot:
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
  br i1 %c2cmp, label %return, label %jt3.prologue, !prof !18

cold:
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

define void @func_without_entry_count(i32 %num) {
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
!15 = !{!"function_entry_count", i64 1}
!16 = !{!"branch_weights", i32 1, i32 0, i32 0, i32 0, i32 0, i32 0}
!17 = !{!"branch_weights", i32 99999, i32 1}
!18 = !{!"branch_weights", i32 99998, i32 1}
!19 = !{!"branch_weights", i32 97000, i32 1000, i32 1000, i32 1000}
