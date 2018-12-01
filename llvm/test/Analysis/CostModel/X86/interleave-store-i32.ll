; REQUIRES: asserts
; RUN: opt -loop-vectorize -S -mattr=avx512f --debug-only=loop-vectorize < %s 2>&1| FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = global [10240 x i32] zeroinitializer, align 16
@B = global [10240 x i32] zeroinitializer, align 16

; Function Attrs: nounwind uwtable
define void @store_i32_interleave4() {
;CHECK-LABEL: store_i32_interleave4
;CHECK: Found an estimated cost of 1 for VF 1 For instruction:   store i32 %add16
;CHECK: Found an estimated cost of 5 for VF 2 For instruction:   store i32 %add16
;CHECK: Found an estimated cost of 5 for VF 4 For instruction:   store i32 %add16
;CHECK: Found an estimated cost of 11 for VF 8 For instruction:   store i32 %add16
;CHECK: Found an estimated cost of 22 for VF 16 For instruction:   store i32 %add16
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds [10240 x i32], [10240 x i32]* @A, i64 0, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 16
  %arrayidx2 = getelementptr inbounds [10240 x i32], [10240 x i32]* @B, i64 0, i64 %indvars.iv
  store i32 %0, i32* %arrayidx2, align 16
  %add = add nsw i32 %0, 1
  %1 = or i64 %indvars.iv, 1
  %arrayidx7 = getelementptr inbounds [10240 x i32], [10240 x i32]* @B, i64 0, i64 %1
  store i32 %add, i32* %arrayidx7, align 4
  %add10 = add nsw i32 %0, 2
  %2 = or i64 %indvars.iv, 2
  %arrayidx13 = getelementptr inbounds [10240 x i32], [10240 x i32]* @B, i64 0, i64 %2
  store i32 %add10, i32* %arrayidx13, align 8
  %add16 = add nsw i32 %0, 3
  %3 = or i64 %indvars.iv, 3
  %arrayidx19 = getelementptr inbounds [10240 x i32], [10240 x i32]* @B, i64 0, i64 %3
  store i32 %add16, i32* %arrayidx19, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 4
  %cmp = icmp slt i64 %indvars.iv.next, 1024
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

define void @store_i32_interleave5() {
;CHECK-LABEL: store_i32_interleave5
;CHECK: Found an estimated cost of 1 for VF 1 For instruction:   store i32 %add22
;CHECK: Found an estimated cost of 7 for VF 2 For instruction:   store i32 %add22
;CHECK: Found an estimated cost of 14 for VF 4 For instruction:   store i32 %add22
;CHECK: Found an estimated cost of 21 for VF 8 For instruction:   store i32 %add22
;CHECK: Found an estimated cost of 35 for VF 16 For instruction:   store i32 %add22
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds [10240 x i32], [10240 x i32]* @A, i64 0, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds [10240 x i32], [10240 x i32]* @B, i64 0, i64 %indvars.iv
  store i32 %0, i32* %arrayidx2, align 4
  %add = add nsw i32 %0, 1
  %1 = add nuw nsw i64 %indvars.iv, 1
  %arrayidx7 = getelementptr inbounds [10240 x i32], [10240 x i32]* @B, i64 0, i64 %1
  store i32 %add, i32* %arrayidx7, align 4
  %add10 = add nsw i32 %0, 2
  %2 = add nuw nsw i64 %indvars.iv, 2
  %arrayidx13 = getelementptr inbounds [10240 x i32], [10240 x i32]* @B, i64 0, i64 %2
  store i32 %add10, i32* %arrayidx13, align 4
  %add16 = add nsw i32 %0, 3
  %3 = add nuw nsw i64 %indvars.iv, 3
  %arrayidx19 = getelementptr inbounds [10240 x i32], [10240 x i32]* @B, i64 0, i64 %3
  store i32 %add16, i32* %arrayidx19, align 4
  %add22 = add nsw i32 %0, 4
  %4 = add nuw nsw i64 %indvars.iv, 4
  %arrayidx25 = getelementptr inbounds [10240 x i32], [10240 x i32]* @B, i64 0, i64 %4
  store i32 %add22, i32* %arrayidx25, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 5
  %cmp = icmp slt i64 %indvars.iv.next, 1024
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}
