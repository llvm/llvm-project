; RUN: llvm-profdata merge %S/Inputs/large_count_remarks.proftext -o %t.profdata
; RUN: opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -pass-remarks=pgo-instrumentation -pgo-emit-branch-prob -S 2>&1| FileCheck %s --check-prefix=ANALYSIS

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @test(i32 %i) {
entry:
  %cmp = icmp sgt i32 %i, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %add = add nsw i32 %i, 2
  br label %if.end

if.end:
  %retv = phi i32 [ %add, %if.then ], [ %i, %entry ]
  ret i32 %retv
}

; ANALYSIS:remark: <unknown>:0:0: sgt_i32_Zero {{.*}}50.00% (total count : 40000000000)
