; RUN: llvm-profdata merge %S/Inputs/fix_entry_count_sampled.proftext -o %t.profdata
; RUN: opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S | FileCheck %s --check-prefix=USE

; Instrumentation PGO sampling makes corrupt looking counters possible.  This
; tests one extreme case:
; Test loading zero profile counts for all instrumented blocks while the entry
; block is not instrumented.  Additionally include a non-zero profile count for
; a select instruction, which prevents short circuiting the PGO application.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @test_no_entry_block_counter(i32 %n) {
; USE: define i32 @test_no_entry_block_counter(i32 %n)
; USE-SAME: !prof ![[ENTRY_COUNT:[0-9]*]]
entry:
  %cmp = icmp slt i32 42, %n
  br i1 %cmp, label %tail1, label %tail2
tail1:
  %ret = select i1 true, i32 %n, i32 42
; USE:  %ret = select i1 true, i32 %n, i32 42
; USE-SAME: !prof ![[BW_FOR_SELECT:[0-9]+]]
  ret i32 %ret
tail2:
  ret i32 42
}
; USE: ![[ENTRY_COUNT]] = !{!"function_entry_count", i64 1}
; USE: ![[BW_FOR_SELECT]] = !{!"branch_weights", i32 1, i32 0}
