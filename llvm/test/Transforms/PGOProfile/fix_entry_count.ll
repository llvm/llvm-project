; RUN: llvm-profdata merge %S/Inputs/fix_entry_count.proftext -o %t.profdata
; RUN: opt < %s -pgo-instr-use -pgo-instrument-entry=true -pgo-test-profile-file=%t.profdata -S | FileCheck %s --check-prefix=USE
; RUN: opt < %s -passes=pgo-instr-use -pgo-instrument-entry=true -pgo-test-profile-file=%t.profdata -S | FileCheck %s --check-prefix=USE

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @test_simple_for(i32 %n) {
; USE: define i32 @test_simple_for(i32 %n)
; USE-SAME: !prof ![[ENTRY_COUNT:[0-9]*]]
entry:
  br label %for.cond

for.cond:
  %i = phi i32 [ 0, %entry ], [ %inc1, %for.inc ]
  %sum = phi i32 [ 1, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i, %n
  br i1 %cmp, label %for.body, label %for.end
; USE: br i1 %cmp, label %for.body, label %for.end
; USE-SAME: !prof ![[BW_FOR_COND:[0-9]+]]

for.body:
  %inc = add nsw i32 %sum, 1
  br label %for.inc

for.inc:
  %inc1 = add nsw i32 %i, 1
  br label %for.cond

for.end:
  ret i32 %sum
}
; USE: ![[ENTRY_COUNT]] = !{!"function_entry_count", i64 1}
; USE: ![[BW_FOR_COND]] = !{!"branch_weights", i32 96, i32 1}
