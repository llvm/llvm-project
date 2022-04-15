; RUN: opt -S < %s -basic-aa -loop-vectorize -force-vector-interleave=1 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

; CHECK-LABEL: @add_a(
; CHECK: load <16 x i8>, <16 x i8>*
; CHECK: add <16 x i8>
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

; Ensure that we preserve nuw/nsw if we're not shrinking the values we're
; working with.
; CHECK-LABEL: @add_a1(
; CHECK: load <16 x i8>, <16 x i8>*
; CHECK: add nuw nsw <16 x i8>
; CHECK: store <16 x i8>
; Function Attrs: nounwind
define void @add_a1(i8* noalias nocapture readonly %p, i8* noalias nocapture %q, i32 %len) #0 {
entry:
  %cmp8 = icmp sgt i32 %len, 0
  br i1 %cmp8, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i8, i8* %p, i64 %indvars.iv
  %0 = load i8, i8* %arrayidx
  %add = add nuw nsw i8 %0, 2
  %arrayidx3 = getelementptr inbounds i8, i8* %q, i64 %indvars.iv
  store i8 %add, i8* %arrayidx3
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %len
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: @add_b(
; CHECK: load <8 x i16>, <8 x i16>*
; CHECK: add <8 x i16>
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
; CHECK: add <8 x i16>
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
; CHECK: add <16 x i8>
; CHECK: or <16 x i8>
; CHECK: mul <16 x i8>
; CHECK: and <16 x i8>
; CHECK: xor <16 x i8>
; CHECK: mul <16 x i8>
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
; CHECK: add <8 x i8>
; CHECK: or <8 x i8>
; CHECK: mul <8 x i8>
; CHECK: and <8 x i8>
; CHECK: xor <8 x i8>
; CHECK: mul <8 x i8>
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

; CHECK-LABEL: @add_phifail(
; CHECK: load <16 x i8>, <16 x i8>*
; CHECK: add nuw nsw <16 x i32>
; CHECK: store <16 x i8>
; Function Attrs: nounwind
define void @add_phifail(i8* noalias nocapture readonly %p, i8* noalias nocapture %q, i32 %len) #0 {
entry:
  %cmp8 = icmp sgt i32 %len, 0
  br i1 %cmp8, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %a_phi = phi i32 [ %conv, %for.body ], [ 0, %entry ]
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

; Function Attrs: nounwind
; When we vectorize this loop, we generate correct code
; even when %len exactly divides VF (since we extract from the second last index
; and pass this to the for.cond.cleanup block). Vectorized loop returns 
; the correct value a_phi = p[len -2]
define i8 @add_phifail2(i8* noalias nocapture readonly %p, i8* noalias nocapture %q, i32 %len) #0 {
; CHECK-LABEL: @add_phifail2(
; CHECK: vector.body:
; CHECK:   %wide.load = load <16 x i8>, <16 x i8>*
; CHECK:   %[[L1:.+]] = zext <16 x i8> %wide.load to <16 x i32>
; CHECK:   add nuw nsw <16 x i32>
; CHECK:   store <16 x i8>
; CHECK:   add nuw i64 %index, 16
; CHECK:   icmp eq i64 %index.next, %n.vec
; CHECK: middle.block:
; CHECK:   %vector.recur.extract = extractelement <16 x i32> %[[L1]], i32 15
; CHECK:   %vector.recur.extract.for.phi = extractelement <16 x i32> %[[L1]], i32 14
; CHECK: for.cond.cleanup:
; CHECK:   %a_phi.lcssa = phi i32 [ %scalar.recur, %for.body ], [ %vector.recur.extract.for.phi, %middle.block ]
; CHECK:   %ret = trunc i32 %a_phi.lcssa to i8
; CHECK:   ret i8 %ret
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %ret = trunc i32 %a_phi to i8
  ret i8 %ret

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %a_phi = phi i32 [ %conv, %for.body ], [ 0, %entry ]
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

attributes #0 = { nounwind }

