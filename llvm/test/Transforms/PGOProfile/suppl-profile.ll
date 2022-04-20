; Supplement instr profile suppl-profile.proftext with sample profile
; sample-profile.proftext.
; RUN: llvm-profdata merge -instr -suppl-min-size-threshold=0 \
; RUN:   -supplement-instr-with-sample=%p/Inputs/sample-profile.proftext \
; RUN:   %S/Inputs/suppl-profile.proftext -o %t.profdata
; RUN: opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Check test_simple_for has a non-zero entry count and doesn't have any other
; prof metadata.
; CHECK: @test_simple_for(i32 %n) {{.*}} !prof ![[ENTRY_COUNT:[0-9]+]]
; CHECK-NOT: !prof !
; CHECK: ![[ENTRY_COUNT]] = !{!"function_entry_count", i64 540}
define i32 @test_simple_for(i32 %n) {
entry:
  br label %for.cond

for.cond:
  %i = phi i32 [ 0, %entry ], [ %inc1, %for.inc ]
  %sum = phi i32 [ 1, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i, %n
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %inc = add nsw i32 %sum, 1
  br label %for.inc

for.inc:
  %inc1 = add nsw i32 %i, 1
  br label %for.cond

for.end:
  ret i32 %sum
}
