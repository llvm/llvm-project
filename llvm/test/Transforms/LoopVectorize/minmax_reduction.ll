; RUN: opt -S -passes=loop-vectorize,dce -force-vector-width=2 -force-vector-interleave=1  < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

@A = common global [1024 x i32] zeroinitializer, align 16
@fA = common global [1024 x float] zeroinitializer, align 16
@dA = common global [1024 x double] zeroinitializer, align 16

; Signed tests.

; Turn this into a max reduction. Make sure we use a splat to initialize the
; vector for the reduction.
; CHECK-LABEL: @max_red(
; CHECK: %[[VAR:.*]] = insertelement <2 x i32> poison, i32 %max, i64 0
; CHECK: {{.*}} = shufflevector <2 x i32> %[[VAR]], <2 x i32> poison, <2 x i32> zeroinitializer
; CHECK: icmp sgt <2 x i32>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: call i32 @llvm.vector.reduce.smax.v2i32

define i32 @max_red(i32 %max) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi i32 [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x i32], ptr @A, i64 0, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp3 = icmp sgt i32 %0, %max.red.08
  %max.red.0 = select i1 %cmp3, i32 %0, i32 %max.red.08
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 %max.red.0
}

; Turn this into a max reduction. The select has its inputs reversed therefore
; this is a max reduction.
; CHECK-LABEL: @max_red_inverse_select(
; CHECK: icmp slt <2 x i32>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: call i32 @llvm.vector.reduce.smax.v2i32

define i32 @max_red_inverse_select(i32 %max) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi i32 [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x i32], ptr @A, i64 0, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp3 = icmp slt i32 %max.red.08, %0
  %max.red.0 = select i1 %cmp3, i32 %0, i32 %max.red.08
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 %max.red.0
}

; Turn this into a min reduction.
; CHECK-LABEL: @min_red(
; CHECK: icmp slt <2 x i32>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: call i32 @llvm.vector.reduce.smin.v2i32

define i32 @min_red(i32 %max) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi i32 [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x i32], ptr @A, i64 0, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp3 = icmp slt i32 %0, %max.red.08
  %max.red.0 = select i1 %cmp3, i32 %0, i32 %max.red.08
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 %max.red.0
}

; Turn this into a min reduction. The select has its inputs reversed therefore
; this is a min reduction.
; CHECK-LABEL: @min_red_inverse_select(
; CHECK: icmp sgt <2 x i32>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: call i32 @llvm.vector.reduce.smin.v2i32

define i32 @min_red_inverse_select(i32 %max) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi i32 [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x i32], ptr @A, i64 0, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp3 = icmp sgt i32 %max.red.08, %0
  %max.red.0 = select i1 %cmp3, i32 %0, i32 %max.red.08
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 %max.red.0
}

; Unsigned tests.

; Turn this into a max reduction.
; CHECK-LABEL: @umax_red(
; CHECK: icmp ugt <2 x i32>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: call i32 @llvm.vector.reduce.umax.v2i32

define i32 @umax_red(i32 %max) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi i32 [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x i32], ptr @A, i64 0, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp3 = icmp ugt i32 %0, %max.red.08
  %max.red.0 = select i1 %cmp3, i32 %0, i32 %max.red.08
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 %max.red.0
}

; Turn this into a max reduction. The select has its inputs reversed therefore
; this is a max reduction.
; CHECK-LABEL: @umax_red_inverse_select(
; CHECK: icmp ult <2 x i32>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: call i32 @llvm.vector.reduce.umax.v2i32

define i32 @umax_red_inverse_select(i32 %max) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi i32 [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x i32], ptr @A, i64 0, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp3 = icmp ult i32 %max.red.08, %0
  %max.red.0 = select i1 %cmp3, i32 %0, i32 %max.red.08
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 %max.red.0
}

; Turn this into a min reduction.
; CHECK-LABEL: @umin_red(
; CHECK: icmp ult <2 x i32>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: call i32 @llvm.vector.reduce.umin.v2i32

define i32 @umin_red(i32 %max) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi i32 [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x i32], ptr @A, i64 0, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp3 = icmp ult i32 %0, %max.red.08
  %max.red.0 = select i1 %cmp3, i32 %0, i32 %max.red.08
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 %max.red.0
}

; Turn this into a min reduction. The select has its inputs reversed therefore
; this is a min reduction.
; CHECK-LABEL: @umin_red_inverse_select(
; CHECK: icmp ugt <2 x i32>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: call i32 @llvm.vector.reduce.umin.v2i32

define i32 @umin_red_inverse_select(i32 %max) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi i32 [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x i32], ptr @A, i64 0, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp3 = icmp ugt i32 %max.red.08, %0
  %max.red.0 = select i1 %cmp3, i32 %0, i32 %max.red.08
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 %max.red.0
}

; SGE -> SLT
; Turn this into a min reduction (select inputs are reversed).
; CHECK-LABEL: @sge_min_red(
; CHECK: icmp sge <2 x i32>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: call i32 @llvm.vector.reduce.smin.v2i32

define i32 @sge_min_red(i32 %max) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi i32 [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x i32], ptr @A, i64 0, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp3 = icmp sge i32 %0, %max.red.08
  %max.red.0 = select i1 %cmp3, i32 %max.red.08, i32 %0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 %max.red.0
}

; SLE -> SGT
; Turn this into a max reduction (select inputs are reversed).
; CHECK-LABEL: @sle_min_red(
; CHECK: icmp sle <2 x i32>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: call i32 @llvm.vector.reduce.smax.v2i32

define i32 @sle_min_red(i32 %max) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi i32 [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x i32], ptr @A, i64 0, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp3 = icmp sle i32 %0, %max.red.08
  %max.red.0 = select i1 %cmp3, i32 %max.red.08, i32 %0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 %max.red.0
}

; UGE -> ULT
; Turn this into a min reduction (select inputs are reversed).
; CHECK-LABEL: @uge_min_red(
; CHECK: icmp uge <2 x i32>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: call i32 @llvm.vector.reduce.umin.v2i32

define i32 @uge_min_red(i32 %max) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi i32 [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x i32], ptr @A, i64 0, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp3 = icmp uge i32 %0, %max.red.08
  %max.red.0 = select i1 %cmp3, i32 %max.red.08, i32 %0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 %max.red.0
}

; ULE -> UGT
; Turn this into a max reduction (select inputs are reversed).
; CHECK-LABEL: @ule_min_red(
; CHECK: icmp ule <2 x i32>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: call i32 @llvm.vector.reduce.umax.v2i32

define i32 @ule_min_red(i32 %max) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi i32 [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x i32], ptr @A, i64 0, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp3 = icmp ule i32 %0, %max.red.08
  %max.red.0 = select i1 %cmp3, i32 %max.red.08, i32 %0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 %max.red.0
}

; No reduction.
; CHECK-LABEL: @no_red_1(
; CHECK-NOT: icmp <2 x i32>
define i32 @no_red_1(i32 %max) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi i32 [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x i32], ptr @A, i64 0, i64 %indvars.iv
  %arrayidx1 = getelementptr inbounds [1024 x i32], ptr @A, i64 1, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %1 = load i32, ptr %arrayidx1, align 4
  %cmp3 = icmp sgt i32 %0, %1
  %max.red.0 = select i1 %cmp3, i32 %0, i32 %max.red.08
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 %max.red.0
}

; CHECK-LABEL: @no_red_2(
; CHECK-NOT: icmp <2 x i32>
define i32 @no_red_2(i32 %max) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi i32 [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x i32], ptr @A, i64 0, i64 %indvars.iv
  %arrayidx1 = getelementptr inbounds [1024 x i32], ptr @A, i64 1, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %1 = load i32, ptr %arrayidx1, align 4
  %cmp3 = icmp sgt i32 %0, %max.red.08
  %max.red.0 = select i1 %cmp3, i32 %0, i32 %1
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 %max.red.0
}

; Float tests.

; Maximum.

; Turn this into a max reduction in the presence of a no-nans-fp-math attribute.
; CHECK-LABEL: @max_red_float(
; CHECK: fcmp fast ogt <2 x float>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: call fast float @llvm.vector.reduce.fmax.v2f32

define float @max_red_float(float %max) #0 {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi float [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x float], ptr @fA, i64 0, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %cmp3 = fcmp fast ogt float %0, %max.red.08
  %max.red.0 = select i1 %cmp3, float %0, float %max.red.08
  %indvars.iv.next = add i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret float %max.red.0
}

; CHECK-LABEL: @max_red_float_ge(
; CHECK: fcmp fast oge <2 x float>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: call fast float @llvm.vector.reduce.fmax.v2f32

define float @max_red_float_ge(float %max) #0 {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi float [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x float], ptr @fA, i64 0, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %cmp3 = fcmp fast oge float %0, %max.red.08
  %max.red.0 = select i1 %cmp3, float %0, float %max.red.08
  %indvars.iv.next = add i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret float %max.red.0
}

; CHECK-LABEL: @inverted_max_red_float(
; CHECK: fcmp fast olt <2 x float>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: call fast float @llvm.vector.reduce.fmax.v2f32

define float @inverted_max_red_float(float %max) #0 {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi float [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x float], ptr @fA, i64 0, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %cmp3 = fcmp fast olt float %0, %max.red.08
  %max.red.0 = select i1 %cmp3, float %max.red.08, float %0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret float %max.red.0
}

; CHECK-LABEL: @inverted_max_red_float_le(
; CHECK: fcmp fast ole <2 x float>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: call fast float @llvm.vector.reduce.fmax.v2f32

define float @inverted_max_red_float_le(float %max) #0 {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi float [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x float], ptr @fA, i64 0, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %cmp3 = fcmp fast ole float %0, %max.red.08
  %max.red.0 = select i1 %cmp3, float %max.red.08, float %0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret float %max.red.0
}

; CHECK-LABEL: @unordered_max_red_float(
; CHECK: fcmp fast ugt <2 x float>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: call fast float @llvm.vector.reduce.fmax.v2f32

define float @unordered_max_red_float(float %max) #0 {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi float [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x float], ptr @fA, i64 0, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %cmp3 = fcmp fast ugt float %0, %max.red.08
  %max.red.0 = select i1 %cmp3, float %0, float %max.red.08
  %indvars.iv.next = add i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret float %max.red.0
}

; CHECK-LABEL: @unordered_max_red_float_ge(
; CHECK: fcmp fast uge <2 x float>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: call fast float @llvm.vector.reduce.fmax.v2f32

define float @unordered_max_red_float_ge(float %max) #0 {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi float [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x float], ptr @fA, i64 0, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %cmp3 = fcmp fast uge float %0, %max.red.08
  %max.red.0 = select i1 %cmp3, float %0, float %max.red.08
  %indvars.iv.next = add i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret float %max.red.0
}

; CHECK-LABEL: @inverted_unordered_max_red_float(
; CHECK: fcmp fast ult <2 x float>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: call fast float @llvm.vector.reduce.fmax.v2f32

define float @inverted_unordered_max_red_float(float %max) #0 {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi float [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x float], ptr @fA, i64 0, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %cmp3 = fcmp fast ult float %0, %max.red.08
  %max.red.0 = select i1 %cmp3, float %max.red.08, float %0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret float %max.red.0
}

; CHECK-LABEL: @inverted_unordered_max_red_float_le(
; CHECK: fcmp fast ule <2 x float>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: call fast float @llvm.vector.reduce.fmax.v2f32

define float @inverted_unordered_max_red_float_le(float %max) #0 {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi float [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x float], ptr @fA, i64 0, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %cmp3 = fcmp fast ule float %0, %max.red.08
  %max.red.0 = select i1 %cmp3, float %max.red.08, float %0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret float %max.red.0
}

; Minimum.

; Turn this into a min reduction in the presence of a no-nans-fp-math attribute.
; CHECK-LABEL: @min_red_float(
; CHECK: fcmp fast olt <2 x float>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: call fast float @llvm.vector.reduce.fmin.v2f32

define float @min_red_float(float %min) #0 {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %min.red.08 = phi float [ %min, %entry ], [ %min.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x float], ptr @fA, i64 0, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %cmp3 = fcmp fast olt float %0, %min.red.08
  %min.red.0 = select i1 %cmp3, float %0, float %min.red.08
  %indvars.iv.next = add i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret float %min.red.0
}

; CHECK-LABEL: @min_red_float_le(
; CHECK: fcmp fast ole <2 x float>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: call fast float @llvm.vector.reduce.fmin.v2f32

define float @min_red_float_le(float %min) #0 {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %min.red.08 = phi float [ %min, %entry ], [ %min.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x float], ptr @fA, i64 0, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %cmp3 = fcmp fast ole float %0, %min.red.08
  %min.red.0 = select i1 %cmp3, float %0, float %min.red.08
  %indvars.iv.next = add i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret float %min.red.0
}

; CHECK-LABEL: @inverted_min_red_float(
; CHECK: fcmp fast ogt <2 x float>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: call fast float @llvm.vector.reduce.fmin.v2f32

define float @inverted_min_red_float(float %min) #0 {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %min.red.08 = phi float [ %min, %entry ], [ %min.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x float], ptr @fA, i64 0, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %cmp3 = fcmp fast ogt float %0, %min.red.08
  %min.red.0 = select i1 %cmp3, float %min.red.08, float %0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret float %min.red.0
}

; CHECK-LABEL: @inverted_min_red_float_ge(
; CHECK: fcmp fast oge <2 x float>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: call fast float @llvm.vector.reduce.fmin.v2f32

define float @inverted_min_red_float_ge(float %min) #0 {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %min.red.08 = phi float [ %min, %entry ], [ %min.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x float], ptr @fA, i64 0, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %cmp3 = fcmp fast oge float %0, %min.red.08
  %min.red.0 = select i1 %cmp3, float %min.red.08, float %0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret float %min.red.0
}

; CHECK-LABEL: @unordered_min_red_float(
; CHECK: fcmp fast ult <2 x float>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: call fast float @llvm.vector.reduce.fmin.v2f32

define float @unordered_min_red_float(float %min) #0 {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %min.red.08 = phi float [ %min, %entry ], [ %min.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x float], ptr @fA, i64 0, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %cmp3 = fcmp fast ult float %0, %min.red.08
  %min.red.0 = select i1 %cmp3, float %0, float %min.red.08
  %indvars.iv.next = add i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret float %min.red.0
}

; CHECK-LABEL: @unordered_min_red_float_le(
; CHECK: fcmp fast ule <2 x float>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: call fast float @llvm.vector.reduce.fmin.v2f32

define float @unordered_min_red_float_le(float %min) #0 {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %min.red.08 = phi float [ %min, %entry ], [ %min.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x float], ptr @fA, i64 0, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %cmp3 = fcmp fast ule float %0, %min.red.08
  %min.red.0 = select i1 %cmp3, float %0, float %min.red.08
  %indvars.iv.next = add i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret float %min.red.0
}

; CHECK-LABEL: @inverted_unordered_min_red_float(
; CHECK: fcmp fast ugt <2 x float>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: call fast float @llvm.vector.reduce.fmin.v2f32

define float @inverted_unordered_min_red_float(float %min) #0 {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %min.red.08 = phi float [ %min, %entry ], [ %min.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x float], ptr @fA, i64 0, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %cmp3 = fcmp fast ugt float %0, %min.red.08
  %min.red.0 = select i1 %cmp3, float %min.red.08, float %0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret float %min.red.0
}

; CHECK-LABEL: @inverted_unordered_min_red_float_ge(
; CHECK: fcmp fast uge <2 x float>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: call fast float @llvm.vector.reduce.fmin.v2f32

define float @inverted_unordered_min_red_float_ge(float %min) #0 {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %min.red.08 = phi float [ %min, %entry ], [ %min.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x float], ptr @fA, i64 0, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %cmp3 = fcmp fast uge float %0, %min.red.08
  %min.red.0 = select i1 %cmp3, float %min.red.08, float %0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret float %min.red.0
}

; Make sure we handle doubles, too.
; CHECK-LABEL: @min_red_double(
; CHECK: fcmp fast olt <2 x double>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: call fast double @llvm.vector.reduce.fmin.v2f64

define double @min_red_double(double %min) #0 {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %min.red.08 = phi double [ %min, %entry ], [ %min.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x double], ptr @dA, i64 0, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 4
  %cmp3 = fcmp fast olt double %0, %min.red.08
  %min.red.0 = select i1 %cmp3, double %0, double %min.red.08
  %indvars.iv.next = add i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret double %min.red.0
}


; Don't this into a max reduction. The no-nans-fp-math attribute is missing
; CHECK-LABEL: @max_red_float_nans(
; CHECK-NOT: <2 x float>

define float @max_red_float_nans(float %max) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi float [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x float], ptr @fA, i64 0, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %cmp3 = fcmp fast ogt float %0, %max.red.08
  %max.red.0 = select i1 %cmp3, float %0, float %max.red.08
  %indvars.iv.next = add i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret float %max.red.0
}

; As above, with the no-signed-zeros-fp-math attribute missing
; CHECK-LABEL: @max_red_float_nsz(
; CHECK-NOT: <2 x float>

define float @max_red_float_nsz(float %max) #1 {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi float [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x float], ptr @fA, i64 0, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %cmp3 = fcmp fast ogt float %0, %max.red.08
  %max.red.0 = select i1 %cmp3, float %0, float %max.red.08
  %indvars.iv.next = add i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret float %max.red.0
}

; CHECK-LABEL: @smin_intrinsic(
; CHECK: <2 x i32> @llvm.smin.v2i32
; CHECK: i32 @llvm.vector.reduce.smin.v2i32
define i32 @smin_intrinsic(ptr nocapture readonly %x) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.012 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %s.011 = phi i32 [ 100, %entry ], [ %1, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %x, i32 %i.012
  %0 = load i32, ptr %arrayidx, align 4
  %1 = tail call i32 @llvm.smin.i32(i32 %s.011, i32 %0)
  %inc = add nuw nsw i32 %i.012, 1
  %exitcond.not = icmp eq i32 %inc, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret i32 %1
}

; CHECK-LABEL: @smax_intrinsic(
; CHECK: <2 x i32> @llvm.smax.v2i32
; CHECK: i32 @llvm.vector.reduce.smax.v2i32
define i32 @smax_intrinsic(ptr nocapture readonly %x) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.012 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %s.011 = phi i32 [ 100, %entry ], [ %1, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %x, i32 %i.012
  %0 = load i32, ptr %arrayidx, align 4
  %1 = tail call i32 @llvm.smax.i32(i32 %s.011, i32 %0)
  %inc = add nuw nsw i32 %i.012, 1
  %exitcond.not = icmp eq i32 %inc, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret i32 %1
}

; CHECK-LABEL: @umin_intrinsic(
; CHECK: <2 x i32> @llvm.umin.v2i32
; CHECK: i32 @llvm.vector.reduce.umin.v2i32
define i32 @umin_intrinsic(ptr nocapture readonly %x) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.012 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %s.011 = phi i32 [ 100, %entry ], [ %1, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %x, i32 %i.012
  %0 = load i32, ptr %arrayidx, align 4
  %1 = tail call i32 @llvm.umin.i32(i32 %s.011, i32 %0)
  %inc = add nuw nsw i32 %i.012, 1
  %exitcond.not = icmp eq i32 %inc, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret i32 %1
}

; CHECK-LABEL: @umax_intrinsic(
; CHECK: <2 x i32> @llvm.umax.v2i32
; CHECK: i32 @llvm.vector.reduce.umax.v2i32
define i32 @umax_intrinsic(ptr nocapture readonly %x) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.012 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %s.011 = phi i32 [ 100, %entry ], [ %1, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %x, i32 %i.012
  %0 = load i32, ptr %arrayidx, align 4
  %1 = tail call i32 @llvm.umax.i32(i32 %s.011, i32 %0)
  %inc = add nuw nsw i32 %i.012, 1
  %exitcond.not = icmp eq i32 %inc, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret i32 %1
}

; CHECK-LABEL: @fmin_intrinsic(
; CHECK: nnan nsz <2 x float> @llvm.minnum.v2f32
; CHECK: nnan nsz float @llvm.vector.reduce.fmin.v2f32
define float @fmin_intrinsic(ptr nocapture readonly %x) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret float %1

for.body:                                         ; preds = %entry, %for.body
  %i.012 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %s.011 = phi float [ 0.000000e+00, %entry ], [ %1, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %x, i32 %i.012
  %0 = load float, ptr %arrayidx, align 4
  %1 = tail call nnan nsz float @llvm.minnum.f32(float %s.011, float %0)
  %inc = add nuw nsw i32 %i.012, 1
  %exitcond.not = icmp eq i32 %inc, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: @fmax_intrinsic(
; CHECK: fast <2 x float> @llvm.maxnum.v2f32
; CHECK: fast float @llvm.vector.reduce.fmax.v2f32
define float @fmax_intrinsic(ptr nocapture readonly %x) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret float %1

for.body:                                         ; preds = %entry, %for.body
  %i.012 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %s.011 = phi float [ 0.000000e+00, %entry ], [ %1, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %x, i32 %i.012
  %0 = load float, ptr %arrayidx, align 4
  %1 = tail call fast float @llvm.maxnum.f32(float %s.011, float %0)
  %inc = add nuw nsw i32 %i.012, 1
  %exitcond.not = icmp eq i32 %inc, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: @fmin_intrinsic_nofast(
; CHECK-NOT: <2 x float> @llvm.minnum.v2f32
define float @fmin_intrinsic_nofast(ptr nocapture readonly %x) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret float %1

for.body:                                         ; preds = %entry, %for.body
  %i.012 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %s.011 = phi float [ 0.000000e+00, %entry ], [ %1, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %x, i32 %i.012
  %0 = load float, ptr %arrayidx, align 4
  %1 = tail call float @llvm.minnum.f32(float %s.011, float %0)
  %inc = add nuw nsw i32 %i.012, 1
  %exitcond.not = icmp eq i32 %inc, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: @fmax_intrinsic_nofast(
; CHECK-NOT: <2 x float> @llvm.maxnum.v2f32
define float @fmax_intrinsic_nofast(ptr nocapture readonly %x) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret float %1

for.body:                                         ; preds = %entry, %for.body
  %i.012 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %s.011 = phi float [ 0.000000e+00, %entry ], [ %1, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %x, i32 %i.012
  %0 = load float, ptr %arrayidx, align 4
  %1 = tail call float @llvm.maxnum.f32(float %s.011, float %0)
  %inc = add nuw nsw i32 %i.012, 1
  %exitcond.not = icmp eq i32 %inc, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: @sminmax(
; Min and max intrinsics - don't vectorize
; CHECK-NOT: <2 x i32>
define i32 @sminmax(ptr nocapture readonly %x, ptr nocapture readonly %y) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret i32 %cond9

for.body:                                         ; preds = %entry, %for.body
  %i.025 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %s.024 = phi i32 [ 0, %entry ], [ %cond9, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %x, i32 %i.025
  %0 = load i32, ptr %arrayidx, align 4
  %s.0. = tail call i32 @llvm.smin.i32(i32 %s.024, i32 %0)
  %arrayidx3 = getelementptr inbounds i32, ptr %y, i32 %i.025
  %1 = load i32, ptr %arrayidx3, align 4
  %cond9 = tail call i32 @llvm.smax.i32(i32 %s.0., i32 %1)
  %inc = add nuw nsw i32 %i.025, 1
  %exitcond.not = icmp eq i32 %inc, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: @sminmin(
; CHECK: <2 x i32> @llvm.smin.v2i32
; CHECK: <2 x i32> @llvm.smin.v2i32
; CHECK: i32 @llvm.vector.reduce.smin.v2i32
define i32 @sminmin(ptr nocapture readonly %x, ptr nocapture readonly %y) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret i32 %cond9

for.body:                                         ; preds = %entry, %for.body
  %i.025 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %s.024 = phi i32 [ 0, %entry ], [ %cond9, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %x, i32 %i.025
  %0 = load i32, ptr %arrayidx, align 4
  %s.0. = tail call i32 @llvm.smin.i32(i32 %s.024, i32 %0)
  %arrayidx3 = getelementptr inbounds i32, ptr %y, i32 %i.025
  %1 = load i32, ptr %arrayidx3, align 4
  %cond9 = tail call i32 @llvm.smin.i32(i32 %s.0., i32 %1)
  %inc = add nuw nsw i32 %i.025, 1
  %exitcond.not = icmp eq i32 %inc, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

; Make sure any check-not directives are not triggered by function declarations.
; CHECK: declare

declare i32 @llvm.smin.i32(i32, i32)
declare i32 @llvm.smax.i32(i32, i32)
declare i32 @llvm.umin.i32(i32, i32)
declare i32 @llvm.umax.i32(i32, i32)
declare float @llvm.minnum.f32(float, float)
declare float @llvm.maxnum.f32(float, float)

attributes #0 = { "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" }
attributes #1 = { "no-nans-fp-math"="true" }
