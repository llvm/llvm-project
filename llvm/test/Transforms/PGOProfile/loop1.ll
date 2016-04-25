; RUN: opt < %s -pgo-instr-gen -S | FileCheck %s --check-prefix=GEN
; RUN: llvm-profdata merge %S/Inputs/loop1.proftext -o %t.profdata
; RUN: opt < %s -pgo-instr-use -pgo-test-profile-file=%t.profdata -S | FileCheck %s --check-prefix=USE
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; GEN: $__llvm_profile_raw_version = comdat any
; GEN: @__llvm_profile_raw_version = constant i64 72057594037927939, comdat
; GEN: @__profn_test_simple_for = private constant [15 x i8] c"test_simple_for"

define i32 @test_simple_for(i32 %n) {
entry:
; GEN: entry:
; GEN-NOT: call void @llvm.instrprof.increment
  br label %for.cond

for.cond:
; GEN: for.cond:
; GEN-NOT: call void @llvm.instrprof.increment
  %i = phi i32 [ 0, %entry ], [ %inc1, %for.inc ]
  %sum = phi i32 [ 1, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i, %n
  br i1 %cmp, label %for.body, label %for.end
; USE: br i1 %cmp, label %for.body, label %for.end
; USE-SAME: !prof ![[BW_FOR_COND:[0-9]+]]
; USE: ![[BW_FOR_COND]] = !{!"branch_weights", i32 96, i32 4}

for.body:
; GEN: for.body:
; GEN-NOT: call void @llvm.instrprof.increment
  %inc = add nsw i32 %sum, 1
  br label %for.inc

for.inc:
; GEN: for.inc:
; GEN: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([15 x i8], [15 x i8]* @__profn_test_simple_for, i32 0, i32 0), i64 34137660316, i32 2, i32 0)
  %inc1 = add nsw i32 %i, 1
  br label %for.cond

for.end:
; GEN: for.end:
; GEN: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([15 x i8], [15 x i8]* @__profn_test_simple_for, i32 0, i32 0), i64 34137660316, i32 2, i32 1)
  ret i32 %sum
}
