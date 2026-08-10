; Test -sanitizer-coverage-trace-geps=1
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=1 -sanitizer-coverage-trace-geps=1  -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @gep_1(ptr captures(none) %a, i32 %i)  {
entry:
  %idxprom = sext i32 %i to i64
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %idxprom
  store i32 0, ptr %arrayidx, align 4
  ret void
}

; CHECK-LABEL: define void @gep_1(ptr captures(none) %a, i32 %i)
; CHECK:   call void @__sanitizer_cov_trace_gep(i64 %idxprom)
; CHECK: ret void


define void @gep_2(ptr captures(none) %a, i32 %i, i32 %j) {
entry:
  %idxprom = sext i32 %j to i64
  %idxprom1 = sext i32 %i to i64
  %arrayidx2 = getelementptr inbounds [1000 x i32], ptr %a, i64 %idxprom1, i64 %idxprom
  store i32 0, ptr %arrayidx2, align 4
  ret void
}

; CHECK-LABEL: define void @gep_2(ptr captures(none) %a, i32 %i, i32 %j)
; CHECK: call void @__sanitizer_cov_trace_gep(i64 %idxprom1)
; CHECK: call void @__sanitizer_cov_trace_gep(i64 %idxprom)
; CHECK: ret void

; Just make sure we don't insturment this one and don't crash
define void @gep_3(<2 x ptr> %a, i32 %i, i32 %j) {
entry:
  %0 = getelementptr i8, <2 x ptr> %a, <2 x i64> <i64 8, i64 8>
  ret void
}

