; RUN: rm -rf %t && split-file %s %t

; RUN: llvm-profdata merge %t/main.proftext -o %t/main.profdata
; RUN: opt < %t/main.ll -passes=pgo-instr-use -pgo-test-profile-file=%t/main.profdata -S | FileCheck %s

;--- main.ll

; Instrumentation PGO sampling makes corrupt looking counters possible.  This
; tests one extreme case:
; Test loading zero profile counts for all instrumented blocks while the entry
; block is not instrumented.  Additionally include a non-zero profile count for
; a select instruction, which prevents short circuiting the PGO application.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @test_no_entry_block_counter(i32 %n) {
; CHECK: define i32 @test_no_entry_block_counter(i32 %n)
; CHECK-SAME: !prof ![[ENTRY_COUNT:[0-9]*]]
entry:
  %cmp = icmp slt i32 42, %n
  br i1 %cmp, label %tail1, label %tail2
tail1:
  %ret = select i1 true, i32 %n, i32 42
; CHECK:  %ret = select i1 true, i32 %n, i32 42
; CHECK-SAME: !prof ![[BW_FOR_SELECT:[0-9]+]]
  ret i32 %ret
tail2:
  ret i32 42
}
; CHECK: ![[ENTRY_COUNT]] = !{!"function_entry_count", i64 1}
; CHECK: ![[BW_FOR_SELECT]] = !{!"branch_weights", i32 1, i32 0}

;--- main.proftext
:ir
test_no_entry_block_counter
431494656217155589
3
0
0
1

