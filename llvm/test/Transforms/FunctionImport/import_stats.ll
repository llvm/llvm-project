; Test to check thin link importing stats

; -stats requires asserts
; REQUIRES: asserts

; REQUIRES: x86-registered-target

; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/import_stats.ll -o %t2.bc

; Test thin link stats with both new and old LTO
; RUN: llvm-lto -thinlto-action=run -stats %t.bc %t2.bc \
; RUN:		2>&1 | FileCheck %s --check-prefix=THINLINKSTATS
; RUN: llvm-lto2 run -stats -o %t3 %t.bc %t2.bc \
; RUN:          -r %t.bc,hot_function,plx \
; RUN:          -r %t.bc,hot, \
; RUN:          -r %t.bc,critical, \
; RUN:          -r %t.bc,none, \
; RUN:          -r %t2.bc,hot,plx \
; RUN:          -r %t2.bc,critical,plx \
; RUN:          -r %t2.bc,none,plx \
; RUN:          -r %t2.bc,globalvar,plx \
; RUN:          2>&1 | FileCheck %s --check-prefix=THINLINKSTATS

; THINLINKSTATS-DAG: 1 function-import   - Number of global variables thin link decided to import
; THINLINKSTATS-DAG: 1 function-import  - Number of critical functions thin link decided to import
; THINLINKSTATS-DAG: 3 function-import  - Number of functions thin link decided to import
; THINLINKSTATS-DAG: 1 function-import  - Number of hot functions thin link decided to import

; ModuleID = 'import_stats.ll'
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; This function has a high profile count, so entry block is hot.
define void @hot_function(i1 %a) !prof !20 {
entry:
  call void @hot()
  call void @critical()
  br i1 %a, label %None1, label %None2, !prof !42
None1:          ; half goes here
  call void @none()
  br label %exit
None2:          ; half goes here
  br label %exit
exit:
  ret void
}

declare void @hot()
declare void @none()
declare void @critical()

!42 = !{!"branch_weights", i32 1, i32 1}

!llvm.module.flags = !{!1}
!20 = !{!"function_entry_count", i64 100, i64 696010031887058302}

!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10}
!3 = !{!"ProfileFormat", !"InstrProf"}
!4 = !{!"TotalCount", i64 300}
!5 = !{!"MaxCount", i64 100}
!6 = !{!"MaxInternalCount", i64 100}
!7 = !{!"MaxFunctionCount", i64 100}
!8 = !{!"NumCounts", i64 4}
!9 = !{!"NumFunctions", i64 1}
!10 = !{!"DetailedSummary", !11}
!11 = !{!12, !13, !14}
!12 = !{i32 10000, i64 100, i32 1}
!13 = !{i32 999000, i64 100, i32 1}
!14 = !{i32 999999, i64 1, i32 4}
