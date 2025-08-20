; REQUIRES: asserts
; RUN: opt < %s -passes=loop-vectorize -force-target-instruction-cost=1 -debug-only=loop-vectorize -S -disable-output 2>&1 | FileCheck %s
target triple = "aarch64-linux-gnu"

define i32 @invalid_legacy_cost(i64 %N) #0 {
; CHECK: LV: Checking a loop in 'invalid_legacy_cost
; CHECK: LV: Found an estimated cost of Invalid for VF vscale x 2 For instruction: %0 = alloca i8, i64 0, align 16
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %0 = alloca i8, i64 0, align 16
  %arrayidx = getelementptr ptr, ptr null, i64 %iv
  store ptr %0, ptr %arrayidx, align 8
  %iv.next = add i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv, %N
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:
  ret i32 0
}

attributes #0 = { "target-features"="+neon,+sve" vscale_range(1,16) }
