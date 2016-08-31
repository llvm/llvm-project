; Test -sanitizer-coverage-trace-geps=1
; RUN: opt < %s -sancov -sanitizer-coverage-level=1 -sanitizer-coverage-trace-geps=1  -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @gep_1(i32* nocapture %a, i32 %i)  {
entry:
  %idxprom = sext i32 %i to i64
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %idxprom
  store i32 0, i32* %arrayidx, align 4
  ret void
}

; CHECK-LABEL: define void @gep_1(i32* nocapture %a, i32 %i)
; CHECK:   call void @__sanitizer_cov_trace_gep(i64 %idxprom)
; CHECK: ret void


define void @gep_2([1000 x i32]* nocapture %a, i32 %i, i32 %j) {
entry:
  %idxprom = sext i32 %j to i64
  %idxprom1 = sext i32 %i to i64
  %arrayidx2 = getelementptr inbounds [1000 x i32], [1000 x i32]* %a, i64 %idxprom1, i64 %idxprom
  store i32 0, i32* %arrayidx2, align 4
  ret void
}

; CHECK-LABEL: define void @gep_2([1000 x i32]* nocapture %a, i32 %i, i32 %j) {
; CHECK: call void @__sanitizer_cov_trace_gep(i64 %idxprom1)
; CHECK: call void @__sanitizer_cov_trace_gep(i64 %idxprom)
; CHECK: ret void
