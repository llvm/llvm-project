; RUN: opt -S < %s -basicaa -loop-vectorize -simplifycfg -instsimplify -instcombine -licm -force-vector-interleave=1 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

; CHECK-LABEL: @add_a(
; CHECK: load <16 x i8>, <16 x i8>*
; CHECK: add nuw nsw <16 x i8>
; CHECK: store <16 x i8>
; Function Attrs: nounwind
define void @add_a(i8* noalias nocapture readonly %p, i8* noalias nocapture %q, i32 %len) #0 {
entry:
  %cmp8 = icmp sgt i32 %len, 0
  br i1 %cmp8, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i8, i8* %p, i64 %indvars.iv
  %0 = load i8, i8* %arrayidx
  %conv = zext i8 %0 to i32
  %add = add nuw nsw i32 %conv, 2
  %conv1 = trunc i32 %add to i8
  %arrayidx3 = getelementptr inbounds i8, i8* %q, i64 %indvars.iv
  store i8 %conv1, i8* %arrayidx3
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %len
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: @add_b(
; CHECK: load <8 x i16>, <8 x i16>*
; CHECK: add nuw nsw <8 x i16>
; CHECK: store <8 x i16>
; Function Attrs: nounwind
define void @add_b(i16* noalias nocapture readonly %p, i16* noalias nocapture %q, i32 %len) #0 {
entry:
  %cmp9 = icmp sgt i32 %len, 0
  br i1 %cmp9, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i16, i16* %p, i64 %indvars.iv
  %0 = load i16, i16* %arrayidx
  %conv8 = zext i16 %0 to i32
  %add = add nuw nsw i32 %conv8, 2
  %conv1 = trunc i32 %add to i16
  %arrayidx3 = getelementptr inbounds i16, i16* %q, i64 %indvars.iv
  store i16 %conv1, i16* %arrayidx3
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %len
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: @add_c(
; CHECK: load <8 x i8>, <8 x i8>*
; CHECK: add nuw nsw <8 x i16>
; CHECK: store <8 x i16>
; Function Attrs: nounwind
define void @add_c(i8* noalias nocapture readonly %p, i16* noalias nocapture %q, i32 %len) #0 {
entry:
  %cmp8 = icmp sgt i32 %len, 0
  br i1 %cmp8, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i8, i8* %p, i64 %indvars.iv
  %0 = load i8, i8* %arrayidx
  %conv = zext i8 %0 to i32
  %add = add nuw nsw i32 %conv, 2
  %conv1 = trunc i32 %add to i16
  %arrayidx3 = getelementptr inbounds i16, i16* %q, i64 %indvars.iv
  store i16 %conv1, i16* %arrayidx3
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %len
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: @add_d(
; CHECK: load <4 x i16>
; CHECK: add nsw <4 x i32>
; CHECK: store <4 x i32>
define void @add_d(i16* noalias nocapture readonly %p, i32* noalias nocapture %q, i32 %len) #0 {
entry:
  %cmp7 = icmp sgt i32 %len, 0
  br i1 %cmp7, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i16, i16* %p, i64 %indvars.iv
  %0 = load i16, i16* %arrayidx
  %conv = sext i16 %0 to i32
  %add = add nsw i32 %conv, 2
  %arrayidx2 = getelementptr inbounds i32, i32* %q, i64 %indvars.iv
  store i32 %add, i32* %arrayidx2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %len
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: @add_e(
; CHECK: load <16 x i8>
; CHECK: shl <16 x i8>
; CHECK: add nuw nsw <16 x i8>
; CHECK: or <16 x i8>
; CHECK: mul nuw nsw <16 x i8>
; CHECK: and <16 x i8>
; CHECK: xor <16 x i8>
; CHECK: mul nuw nsw <16 x i8>
; CHECK: store <16 x i8>
define void @add_e(i8* noalias nocapture readonly %p, i8* noalias nocapture %q, i8 %arg1, i8 %arg2, i32 %len) #0 {
entry:
  %cmp.32 = icmp sgt i32 %len, 0
  br i1 %cmp.32, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry
  %conv11 = zext i8 %arg2 to i32
  %conv13 = zext i8 %arg1 to i32
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i8, i8* %p, i64 %indvars.iv
  %0 = load i8, i8* %arrayidx
  %conv = zext i8 %0 to i32
  %add = shl i32 %conv, 4
  %conv2 = add nuw nsw i32 %add, 32
  %or = or i32 %conv, 51
  %mul = mul nuw nsw i32 %or, 60
  %and = and i32 %conv2, %conv13
  %mul.masked = and i32 %mul, 252
  %conv17 = xor i32 %mul.masked, %conv11
  %mul18 = mul nuw nsw i32 %conv17, %and
  %conv19 = trunc i32 %mul18 to i8
  %arrayidx21 = getelementptr inbounds i8, i8* %q, i64 %indvars.iv
  store i8 %conv19, i8* %arrayidx21
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %len
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: @add_f
; CHECK: load <8 x i16>
; CHECK: trunc <8 x i16>
; CHECK: shl <8 x i8>
; CHECK: add nsw <8 x i8>
; CHECK: or <8 x i8>
; CHECK: mul nuw nsw <8 x i8>
; CHECK: and <8 x i8>
; CHECK: xor <8 x i8>
; CHECK: mul nuw nsw <8 x i8>
; CHECK: store <8 x i8>
define void @add_f(i16* noalias nocapture readonly %p, i8* noalias nocapture %q, i8 %arg1, i8 %arg2, i32 %len) #0 {
entry:
  %cmp.32 = icmp sgt i32 %len, 0
  br i1 %cmp.32, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry
  %conv11 = zext i8 %arg2 to i32
  %conv13 = zext i8 %arg1 to i32
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i16, i16* %p, i64 %indvars.iv
  %0 = load i16, i16* %arrayidx
  %conv = sext i16 %0 to i32
  %add = shl i32 %conv, 4
  %conv2 = add nsw i32 %add, 32
  %or = and i32 %conv, 204
  %conv8 = or i32 %or, 51
  %mul = mul nuw nsw i32 %conv8, 60
  %and = and i32 %conv2, %conv13
  %mul.masked = and i32 %mul, 252
  %conv17 = xor i32 %mul.masked, %conv11
  %mul18 = mul nuw nsw i32 %conv17, %and
  %conv19 = trunc i32 %mul18 to i8
  %arrayidx21 = getelementptr inbounds i8, i8* %q, i64 %indvars.iv
  store i8 %conv19, i8* %arrayidx21
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %len
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: @add_g
; CHECK: load <16 x i8>
; CHECK: xor <16 x i8>
; CHECK: icmp ult <16 x i8>
; CHECK: select <16 x i1> {{.*}}, <16 x i8>
; CHECK: store <16 x i8>
define void @add_g(i8* noalias nocapture readonly %p, i8* noalias nocapture readonly %q, i8* noalias nocapture %r, i8 %arg1, i32 %len) #0 {
  %1 = icmp sgt i32 %len, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0
  %2 = sext i8 %arg1 to i64
  br label %3

._crit_edge:                                      ; preds = %3, %0
  ret void

; <label>:3                                       ; preds = %3, %.lr.ph
  %indvars.iv = phi i64 [ 0, %.lr.ph ], [ %indvars.iv.next, %3 ]
  %x4 = getelementptr inbounds i8, i8* %p, i64 %indvars.iv
  %x5 = load i8, i8* %x4
  %x7 = getelementptr inbounds i8, i8* %q, i64 %indvars.iv
  %x8 = load i8, i8* %x7
  %x9 = zext i8 %x5 to i32
  %x10 = xor i32 %x9, 255
  %x11 = icmp ult i32 %x10, 24
  %x12 = select i1 %x11, i32 %x10, i32 24
  %x13 = trunc i32 %x12 to i8
  store i8 %x13, i8* %x4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %len
  br i1 %exitcond, label %._crit_edge, label %3
}

attributes #0 = { nounwind }
