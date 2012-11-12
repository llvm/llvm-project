; RUN: llc < %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin11.4.2"

%v8_uniform_Stats.0.2.4.10 = type { i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i64, i64, i64, i32, i32, i32, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32, i64, [7 x i32], [7 x i64] }

@globalStats = external global %v8_uniform_Stats.0.2.4.10

define void @MergeStats() nounwind {
allocas:
  %r.i.i720 = atomicrmw max i64* getelementptr inbounds (%v8_uniform_Stats.0.2.4.10* @globalStats, i64 0, i32 30), i64 0 seq_cst
  ret void
}
