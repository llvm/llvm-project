; REQUIRES: asserts
; RUN: opt -loop-vectorize -S -mattr=avx512f --debug-only=loop-vectorize < %s 2>&1 | FileCheck %s 

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = global [10240 x i32] zeroinitializer, align 16
@B = global [10240 x i32] zeroinitializer, align 16

; Function Attrs: nounwind uwtable
define void @load_i32_interleave4() {
;CHECK-LABEL: load_i32_interleave4
;CHECK: Found an estimated cost of 1 for VF 1 For instruction:   %0 = load
;CHECK: Found an estimated cost of 5 for VF 2 For instruction:   %0 = load
;CHECK: Found an estimated cost of 5 for VF 4 For instruction:   %0 = load
;CHECK: Found an estimated cost of 8 for VF 8 For instruction:   %0 = load
;CHECK: Found an estimated cost of 22 for VF 16 For instruction:   %0 = load
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds [10240 x i32], [10240 x i32]* @A, i64 0, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 16
  %1 = or i64 %indvars.iv, 1
  %arrayidx2 = getelementptr inbounds [10240 x i32], [10240 x i32]* @A, i64 0, i64 %1
  %2 = load i32, i32* %arrayidx2, align 4
  %add3 = add nsw i32 %2, %0
  %3 = or i64 %indvars.iv, 2
  %arrayidx6 = getelementptr inbounds [10240 x i32], [10240 x i32]* @A, i64 0, i64 %3
  %4 = load i32, i32* %arrayidx6, align 8
  %add7 = add nsw i32 %add3, %4
  %5 = or i64 %indvars.iv, 3
  %arrayidx10 = getelementptr inbounds [10240 x i32], [10240 x i32]* @A, i64 0, i64 %5
  %6 = load i32, i32* %arrayidx10, align 4
  %add11 = add nsw i32 %add7, %6
  %arrayidx13 = getelementptr inbounds [10240 x i32], [10240 x i32]* @B, i64 0, i64 %indvars.iv
  store i32 %add11, i32* %arrayidx13, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 4
  %cmp = icmp slt i64 %indvars.iv.next, 1024
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

define void @load_i32_interleave5() {
;CHECK-LABEL: load_i32_interleave5
;CHECK: Found an estimated cost of 1 for VF 1 For instruction:   %0 = load
;CHECK: Found an estimated cost of 6 for VF 2 For instruction:   %0 = load
;CHECK: Found an estimated cost of 9 for VF 4 For instruction:   %0 = load
;CHECK: Found an estimated cost of 18 for VF 8 For instruction:   %0 = load
;CHECK: Found an estimated cost of 35 for VF 16 For instruction:   %0 = load
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds [10240 x i32], [10240 x i32]* @A, i64 0, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %1 = add nuw nsw i64 %indvars.iv, 1
  %arrayidx2 = getelementptr inbounds [10240 x i32], [10240 x i32]* @A, i64 0, i64 %1
  %2 = load i32, i32* %arrayidx2, align 4
  %add3 = add nsw i32 %2, %0
  %3 = add nuw nsw i64 %indvars.iv, 2
  %arrayidx6 = getelementptr inbounds [10240 x i32], [10240 x i32]* @A, i64 0, i64 %3
  %4 = load i32, i32* %arrayidx6, align 4
  %add7 = add nsw i32 %add3, %4
  %5 = add nuw nsw i64 %indvars.iv, 3
  %arrayidx10 = getelementptr inbounds [10240 x i32], [10240 x i32]* @A, i64 0, i64 %5
  %6 = load i32, i32* %arrayidx10, align 4
  %add11 = add nsw i32 %add7, %6
  %7 = add nuw nsw i64 %indvars.iv, 4
  %arrayidx14 = getelementptr inbounds [10240 x i32], [10240 x i32]* @A, i64 0, i64 %7
  %8 = load i32, i32* %arrayidx14, align 4
  %add15 = add nsw i32 %add11, %8
  %arrayidx17 = getelementptr inbounds [10240 x i32], [10240 x i32]* @B, i64 0, i64 %indvars.iv
  store i32 %add15, i32* %arrayidx17, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 5
  %cmp = icmp slt i64 %indvars.iv.next, 1024
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}
