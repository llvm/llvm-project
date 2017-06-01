; REQUIRES: asserts
; RUN: opt < %s -S -debug -loop-vectorize -mcpu=slm 2>&1 | FileCheck %s --check-prefix=SLM

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i8 @mul_i8(i8* %dataA, i8* %dataB, i32 %N) {
entry:
  %cmp12 = icmp eq i32 %N, 0
  br i1 %cmp12, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %N to i64
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  %phitmp = trunc i32 %add4 to i8
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  %acc.0.lcssa = phi i8 [ 0, %entry ], [ %phitmp, %for.cond.cleanup.loopexit ]
  ret i8 %acc.0.lcssa

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %acc.013 = phi i32 [ %add4, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i8, i8* %dataA, i64 %indvars.iv
  %0 = load i8, i8* %arrayidx, align 1
  %conv = sext i8 %0 to i32
  %arrayidx2 = getelementptr inbounds i8, i8* %dataB, i64 %indvars.iv
  %1 = load i8, i8* %arrayidx2, align 1
  %conv3 = sext i8 %1 to i32
; sources of the mul is sext\sext from i8 
; use pmullw\sext seq.   
; SLM:  cost of 3 for VF 4 {{.*}} mul nsw i32  
  %mul = mul nsw i32 %conv3, %conv
; sources of the mul is zext\sext from i8
; use pmulhw\pmullw\pshuf
; SLM:  cost of 5 for VF 4 {{.*}} mul nsw i32
  %conv4 = zext i8 %1 to i32
  %mul2 = mul nsw i32 %conv4, %conv
  %sum0 = add i32 %mul, %mul2
; sources of the mul is zext\zext from i8
; use pmullw\zext
; SLM:  cost of 3 for VF 4 {{.*}} mul nsw i32
  %conv5 = zext i8 %0 to i32
  %mul3 = mul nsw i32 %conv5, %conv4
  %sum1 = add i32 %sum0, %mul3
; sources of the mul is sext\-120
; use pmullw\sext
; SLM:  cost of 3 for VF 4 {{.*}} mul nsw i32
  %mul4 = mul nsw i32 -120, %conv3
  %sum2 = add i32 %sum1, %mul4
; sources of the mul is sext\250
; use pmulhw\pmullw\pshuf
; SLM:  cost of 5 for VF 4 {{.*}} mul nsw i32
  %mul5 = mul nsw i32 250, %conv3
  %sum3 = add i32 %sum2, %mul5
; sources of the mul is zext\-120
; use pmulhw\pmullw\pshuf
; SLM:  cost of 5 for VF 4 {{.*}} mul nsw i32
  %mul6 = mul nsw i32 -120, %conv4
  %sum4 = add i32 %sum3, %mul6
; sources of the mul is zext\250
; use pmullw\zext
; SLM:  cost of 3 for VF 4 {{.*}} mul nsw i32
  %mul7 = mul nsw i32 250, %conv4
  %sum5 = add i32 %sum4, %mul7
  %add = add i32 %acc.013, 5
  %add4 = add i32 %add, %sum5
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body
}

define i16 @mul_i16(i16* %dataA, i16* %dataB, i32 %N) {
entry:
  %cmp12 = icmp eq i32 %N, 0
  br i1 %cmp12, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %N to i64
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  %phitmp = trunc i32 %add4 to i16
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  %acc.0.lcssa = phi i16 [ 0, %entry ], [ %phitmp, %for.cond.cleanup.loopexit ]
  ret i16 %acc.0.lcssa

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %acc.013 = phi i32 [ %add4, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i16, i16* %dataA, i64 %indvars.iv
  %0 = load i16, i16* %arrayidx, align 1
  %conv = sext i16 %0 to i32
  %arrayidx2 = getelementptr inbounds i16, i16* %dataB, i64 %indvars.iv
  %1 = load i16, i16* %arrayidx2, align 1
  %conv3 = sext i16 %1 to i32
; sources of the mul is sext\sext from i16 
; use pmulhw\pmullw\pshuf seq.   
; SLM:  cost of 5 for VF 4 {{.*}} mul nsw i32  
  %mul = mul nsw i32 %conv3, %conv
; sources of the mul is zext\sext from i16
; use pmulld
; SLM:  cost of 11 for VF 4 {{.*}} mul nsw i32
  %conv4 = zext i16 %1 to i32
  %mul2 = mul nsw i32 %conv4, %conv
  %sum0 = add i32 %mul, %mul2
; sources of the mul is zext\zext from i16
; use pmulhw\pmullw\zext
; SLM:  cost of 5 for VF 4 {{.*}} mul nsw i32
  %conv5 = zext i16 %0 to i32
  %mul3 = mul nsw i32 %conv5, %conv4
  %sum1 = add i32 %sum0, %mul3
; sources of the mul is sext\-32000
; use pmulhw\pmullw\sext
; SLM:  cost of 5 for VF 4 {{.*}} mul nsw i32
  %mul4 = mul nsw i32 -32000, %conv3
  %sum2 = add i32 %sum1, %mul4
; sources of the mul is sext\64000
; use pmulld
; SLM:  cost of 11 for VF 4 {{.*}} mul nsw i32
  %mul5 = mul nsw i32 64000, %conv3
  %sum3 = add i32 %sum2, %mul5
; sources of the mul is zext\-32000
; use pmulld
; SLM:  cost of 11 for VF 4 {{.*}} mul nsw i32
  %mul6 = mul nsw i32 -32000, %conv4
  %sum4 = add i32 %sum3, %mul6
; sources of the mul is zext\64000
; use pmulhw\pmullw\zext
; SLM:  cost of 5 for VF 4 {{.*}} mul nsw i32
  %mul7 = mul nsw i32 250, %conv4
  %sum5 = add i32 %sum4, %mul7
  %add = add i32 %acc.013, 5
  %add4 = add i32 %add, %sum5
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body
}


