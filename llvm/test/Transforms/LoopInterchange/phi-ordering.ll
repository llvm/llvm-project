; REQUIRES: asserts
; RUN: opt < %s -loop-interchange -verify-dom-info -S -debug 2>&1 | FileCheck %s
;; Checks the order of the inner phi nodes does not cause havoc.
;; The inner loop has a reduction into c. The IV is not the first phi.

; CHECK: Not interchanging loops. Cannot prove legality.

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv8--linux-gnueabihf"

; Function Attrs: norecurse nounwind
define void @test(i32 %T, [90 x i32]* noalias nocapture %C, i16* noalias nocapture readonly %A, i16* noalias nocapture readonly %B) local_unnamed_addr #0 {
entry:
  %cmp45 = icmp sgt i32 %T, 0
  br i1 %cmp45, label %for.body3.lr.ph.preheader, label %for.end21

for.body3.lr.ph.preheader:                        ; preds = %entry
  br label %for.body3.lr.ph

for.body3.lr.ph:                                  ; preds = %for.body3.lr.ph.preheader, %for.inc19
  %i.046 = phi i32 [ %inc20, %for.inc19 ], [ 0, %for.body3.lr.ph.preheader ]
  %mul = mul nsw i32 %i.046, %T
  br label %for.body6.lr.ph

for.body6.lr.ph:                                  ; preds = %for.inc16, %for.body3.lr.ph
  %j.043 = phi i32 [ 0, %for.body3.lr.ph ], [ %inc17, %for.inc16 ]
  %arrayidx14 = getelementptr inbounds [90 x i32], [90 x i32]* %C, i32 %i.046, i32 %j.043
  %arrayidx14.promoted = load i32, i32* %arrayidx14, align 4
  br label %for.body6

for.body6:                                        ; preds = %for.body6, %for.body6.lr.ph
  %add1541 = phi i32 [ %arrayidx14.promoted, %for.body6.lr.ph ], [ %add15, %for.body6 ]
  %k.040 = phi i32 [ 0, %for.body6.lr.ph ], [ %inc, %for.body6 ]
  %add = add nsw i32 %k.040, %mul
  %arrayidx = getelementptr inbounds i16, i16* %A, i32 %add
  %0 = load i16, i16* %arrayidx, align 2
  %conv = sext i16 %0 to i32
  %mul7 = mul nsw i32 %k.040, %T
  %add8 = add nsw i32 %mul7, %j.043
  %arrayidx9 = getelementptr inbounds i16, i16* %B, i32 %add8
  %1 = load i16, i16* %arrayidx9, align 2
  %conv10 = sext i16 %1 to i32
  %mul11 = mul nsw i32 %conv10, %conv
  %add15 = add nsw i32 %mul11, %add1541
  %inc = add nuw nsw i32 %k.040, 1
  %exitcond = icmp eq i32 %inc, %T
  br i1 %exitcond, label %for.inc16, label %for.body6

for.inc16:                                        ; preds = %for.body6
  %add15.lcssa = phi i32 [ %add15, %for.body6 ]
  store i32 %add15.lcssa, i32* %arrayidx14, align 4
  %inc17 = add nuw nsw i32 %j.043, 1
  %exitcond47 = icmp eq i32 %inc17, %T
  br i1 %exitcond47, label %for.inc19, label %for.body6.lr.ph

for.inc19:                                        ; preds = %for.inc16
  %inc20 = add nuw nsw i32 %i.046, 1
  %exitcond48 = icmp eq i32 %inc20, %T
  br i1 %exitcond48, label %for.end21.loopexit, label %for.body3.lr.ph

for.end21.loopexit:                               ; preds = %for.inc19
  br label %for.end21

for.end21:                                        ; preds = %for.end21.loopexit, %entry
  ret void
}
