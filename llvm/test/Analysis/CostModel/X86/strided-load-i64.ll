; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -S -mattr=avx512f --debug-only=loop-vectorize < %s 2>&1| FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = global [10240 x i64] zeroinitializer, align 16
@B = global [10240 x i64] zeroinitializer, align 16

; Function Attrs: nounwind uwtable
define void @load_i64_stride2() {
;CHECK-LABEL: load_i64_stride2
;CHECK: Found an estimated cost of 1 for VF 1 For instruction:   %1 = load
;CHECK: Found an estimated cost of 1 for VF 2 For instruction:   %1 = load
;CHECK: Found an estimated cost of 1 for VF 4 For instruction:   %1 = load
;CHECK: Found an estimated cost of 2 for VF 8 For instruction:   %1 = load
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = shl nsw i64 %indvars.iv, 1
  %arrayidx = getelementptr inbounds [10240 x i64], [10240 x i64]* @A, i64 0, i64 %0
  %1 = load i64, i64* %arrayidx, align 16
  %arrayidx2 = getelementptr inbounds [10240 x i64], [10240 x i64]* @B, i64 0, i64 %indvars.iv
  store i64 %1, i64* %arrayidx2, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

define void @load_i64_stride3() {
;CHECK-LABEL: load_i64_stride3
;CHECK: Found an estimated cost of 1 for VF 1 For instruction:   %1 = load
;CHECK: Found an estimated cost of 1 for VF 2 For instruction:   %1 = load
;CHECK: Found an estimated cost of 2 for VF 4 For instruction:   %1 = load
;CHECK: Found an estimated cost of 3 for VF 8 For instruction:   %1 = load
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = mul nsw i64 %indvars.iv, 3
  %arrayidx = getelementptr inbounds [10240 x i64], [10240 x i64]* @A, i64 0, i64 %0
  %1 = load i64, i64* %arrayidx, align 16
  %arrayidx2 = getelementptr inbounds [10240 x i64], [10240 x i64]* @B, i64 0, i64 %indvars.iv
  store i64 %1, i64* %arrayidx2, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

define void @load_i64_stride4() {
;CHECK-LABEL: load_i64_stride4
;CHECK: Found an estimated cost of 1 for VF 1 For instruction:   %1 = load
;CHECK: Found an estimated cost of 1 for VF 2 For instruction:   %1 = load
;CHECK: Found an estimated cost of 2 for VF 4 For instruction:   %1 = load
;CHECK: Found an estimated cost of 5 for VF 8 For instruction:   %1 = load
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = mul nsw i64 %indvars.iv, 4
  %arrayidx = getelementptr inbounds [10240 x i64], [10240 x i64]* @A, i64 0, i64 %0
  %1 = load i64, i64* %arrayidx, align 16
  %arrayidx2 = getelementptr inbounds [10240 x i64], [10240 x i64]* @B, i64 0, i64 %indvars.iv
  store i64 %1, i64* %arrayidx2, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
