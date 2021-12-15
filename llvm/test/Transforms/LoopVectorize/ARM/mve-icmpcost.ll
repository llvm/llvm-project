; RUN: opt -loop-vectorize -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8.1m.main-arm-none-eabi"

; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %i.016 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.inc ]
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %arrayidx = getelementptr inbounds i16, i16* %s, i32 %i.016
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %1 = load i16, i16* %arrayidx, align 2
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %conv = sext i16 %1 to i32
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %cmp2 = icmp sgt i32 %conv, %conv1
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   br i1 %cmp2, label %if.then, label %for.inc
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %conv6 = add i16 %1, %0
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %arrayidx7 = getelementptr inbounds i16, i16* %d, i32 %i.016
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   store i16 %conv6, i16* %arrayidx7, align 2
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   br label %for.inc
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %inc = add nuw nsw i32 %i.016, 1
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %exitcond.not = icmp eq i32 %inc, %n
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body
; CHECK: LV: Scalar loop costs: 5.
; CHECK: LV: Found an estimated cost of 0 for VF 2 For instruction:   %i.016 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.inc ]
; CHECK: LV: Found an estimated cost of 0 for VF 2 For instruction:   %arrayidx = getelementptr inbounds i16, i16* %s, i32 %i.016
; CHECK: LV: Found an estimated cost of 18 for VF 2 For instruction:   %1 = load i16, i16* %arrayidx, align 2
; CHECK: LV: Found an estimated cost of 4 for VF 2 For instruction:   %conv = sext i16 %1 to i32
; CHECK: LV: Found an estimated cost of 20 for VF 2 For instruction:   %cmp2 = icmp sgt i32 %conv, %conv1
; CHECK: LV: Found an estimated cost of 0 for VF 2 For instruction:   br i1 %cmp2, label %if.then, label %for.inc
; CHECK: LV: Found an estimated cost of 26 for VF 2 For instruction:   %conv6 = add i16 %1, %0
; CHECK: LV: Found an estimated cost of 0 for VF 2 For instruction:   %arrayidx7 = getelementptr inbounds i16, i16* %d, i32 %i.016
; CHECK: LV: Found an estimated cost of 16 for VF 2 For instruction:   store i16 %conv6, i16* %arrayidx7, align 2
; CHECK: LV: Found an estimated cost of 0 for VF 2 For instruction:   br label %for.inc
; CHECK: LV: Found an estimated cost of 1 for VF 2 For instruction:   %inc = add nuw nsw i32 %i.016, 1
; CHECK: LV: Found an estimated cost of 1 for VF 2 For instruction:   %exitcond.not = icmp eq i32 %inc, %n
; CHECK: LV: Found an estimated cost of 0 for VF 2 For instruction:   br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body
; CHECK: LV: Vector loop of width 2 costs: 43.
; CHECK: LV: Found an estimated cost of 0 for VF 4 For instruction:   %i.016 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.inc ]
; CHECK: LV: Found an estimated cost of 0 for VF 4 For instruction:   %arrayidx = getelementptr inbounds i16, i16* %s, i32 %i.016
; CHECK: LV: Found an estimated cost of 2 for VF 4 For instruction:   %1 = load i16, i16* %arrayidx, align 2
; CHECK: LV: Found an estimated cost of 0 for VF 4 For instruction:   %conv = sext i16 %1 to i32
; CHECK: LV: Found an estimated cost of 2 for VF 4 For instruction:   %cmp2 = icmp sgt i32 %conv, %conv1
; CHECK: LV: Found an estimated cost of 0 for VF 4 For instruction:   br i1 %cmp2, label %if.then, label %for.inc
; CHECK: LV: Found an estimated cost of 2 for VF 4 For instruction:   %conv6 = add i16 %1, %0
; CHECK: LV: Found an estimated cost of 0 for VF 4 For instruction:   %arrayidx7 = getelementptr inbounds i16, i16* %d, i32 %i.016
; CHECK: LV: Found an estimated cost of 2 for VF 4 For instruction:   store i16 %conv6, i16* %arrayidx7, align 2
; CHECK: LV: Found an estimated cost of 0 for VF 4 For instruction:   br label %for.inc
; CHECK: LV: Found an estimated cost of 1 for VF 4 For instruction:   %inc = add nuw nsw i32 %i.016, 1
; CHECK: LV: Found an estimated cost of 1 for VF 4 For instruction:   %exitcond.not = icmp eq i32 %inc, %n
; CHECK: LV: Found an estimated cost of 0 for VF 4 For instruction:   br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body
; CHECK: LV: Vector loop of width 4 costs: 2.
; CHECK: LV: Found an estimated cost of 0 for VF 8 For instruction:   %i.016 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.inc ]
; CHECK: LV: Found an estimated cost of 0 for VF 8 For instruction:   %arrayidx = getelementptr inbounds i16, i16* %s, i32 %i.016
; CHECK: LV: Found an estimated cost of 2 for VF 8 For instruction:   %1 = load i16, i16* %arrayidx, align 2
; CHECK: LV: Found an estimated cost of 2 for VF 8 For instruction:   %conv = sext i16 %1 to i32
; CHECK: LV: Found an estimated cost of 36 for VF 8 For instruction:   %cmp2 = icmp sgt i32 %conv, %conv1
; CHECK: LV: Found an estimated cost of 0 for VF 8 For instruction:   br i1 %cmp2, label %if.then, label %for.inc
; CHECK: LV: Found an estimated cost of 2 for VF 8 For instruction:   %conv6 = add i16 %1, %0
; CHECK: LV: Found an estimated cost of 0 for VF 8 For instruction:   %arrayidx7 = getelementptr inbounds i16, i16* %d, i32 %i.016
; CHECK: LV: Found an estimated cost of 2 for VF 8 For instruction:   store i16 %conv6, i16* %arrayidx7, align 2
; CHECK: LV: Found an estimated cost of 0 for VF 8 For instruction:   br label %for.inc
; CHECK: LV: Found an estimated cost of 1 for VF 8 For instruction:   %inc = add nuw nsw i32 %i.016, 1
; CHECK: LV: Found an estimated cost of 1 for VF 8 For instruction:   %exitcond.not = icmp eq i32 %inc, %n
; CHECK: LV: Found an estimated cost of 0 for VF 8 For instruction:   br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body
; CHECK: LV: Vector loop of width 8 costs: 5.
; CHECK: LV: Selecting VF: 4.
define void @expensive_icmp(i16* noalias nocapture %d, i16* nocapture readonly %s, i32 %n, i16 zeroext %m) #0 {
entry:
  %cmp15 = icmp sgt i32 %n, 0
  br i1 %cmp15, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry
  %conv1 = zext i16 %m to i32
  %0 = trunc i32 %n to i16
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.inc, %entry
  ret void

for.body:                                         ; preds = %for.body.lr.ph, %for.inc
  %i.016 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.inc ]
  %arrayidx = getelementptr inbounds i16, i16* %s, i32 %i.016
  %1 = load i16, i16* %arrayidx, align 2
  %conv = sext i16 %1 to i32
  %cmp2 = icmp sgt i32 %conv, %conv1
  br i1 %cmp2, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %conv6 = add i16 %1, %0
  %arrayidx7 = getelementptr inbounds i16, i16* %d, i32 %i.016
  store i16 %conv6, i16* %arrayidx7, align 2
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %inc = add nuw nsw i32 %i.016, 1
  %exitcond.not = icmp eq i32 %inc, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %blkCnt.012 = phi i32 [ %dec, %while.body ], [ %blockSize, %while.body.preheader ]
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %pSrcA.addr.011 = phi i8* [ %incdec.ptr, %while.body ], [ %pSrcA, %while.body.preheader ]
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %pDst.addr.010 = phi i8* [ %incdec.ptr5, %while.body ], [ %pDst, %while.body.preheader ]
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %pSrcB.addr.09 = phi i8* [ %incdec.ptr2, %while.body ], [ %pSrcB, %while.body.preheader ]
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %incdec.ptr = getelementptr inbounds i8, i8* %pSrcA.addr.011, i32 1
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %0 = load i8, i8* %pSrcA.addr.011, align 1
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %conv1 = sext i8 %0 to i32
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %incdec.ptr2 = getelementptr inbounds i8, i8* %pSrcB.addr.09, i32 1
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %1 = load i8, i8* %pSrcB.addr.09, align 1
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %conv3 = sext i8 %1 to i32
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %mul = mul nsw i32 %conv3, %conv1
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %shr = ashr i32 %mul, 7
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %2 = icmp slt i32 %shr, 127
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %spec.select.i = select i1 %2, i32 %shr, i32 127
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %conv4 = trunc i32 %spec.select.i to i8
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %incdec.ptr5 = getelementptr inbounds i8, i8* %pDst.addr.010, i32 1
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   store i8 %conv4, i8* %pDst.addr.010, align 1
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %dec = add i32 %blkCnt.012, -1
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %cmp.not = icmp eq i32 %dec, 0
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   br i1 %cmp.not, label %while.end.loopexit, label %while.body
; CHECK: LV: Scalar loop costs: 9.
; CHECK: LV: Found an estimated cost of 0 for VF 2 For instruction:   %blkCnt.012 = phi i32 [ %dec, %while.body ], [ %blockSize, %while.body.preheader ]
; CHECK: LV: Found an estimated cost of 0 for VF 2 For instruction:   %pSrcA.addr.011 = phi i8* [ %incdec.ptr, %while.body ], [ %pSrcA, %while.body.preheader ]
; CHECK: LV: Found an estimated cost of 0 for VF 2 For instruction:   %pDst.addr.010 = phi i8* [ %incdec.ptr5, %while.body ], [ %pDst, %while.body.preheader ]
; CHECK: LV: Found an estimated cost of 0 for VF 2 For instruction:   %pSrcB.addr.09 = phi i8* [ %incdec.ptr2, %while.body ], [ %pSrcB, %while.body.preheader ]
; CHECK: LV: Found an estimated cost of 0 for VF 2 For instruction:   %incdec.ptr = getelementptr inbounds i8, i8* %pSrcA.addr.011, i32 1
; CHECK: LV: Found an estimated cost of 18 for VF 2 For instruction:   %0 = load i8, i8* %pSrcA.addr.011, align 1
; CHECK: LV: Found an estimated cost of 4 for VF 2 For instruction:   %conv1 = sext i8 %0 to i32
; CHECK: LV: Found an estimated cost of 0 for VF 2 For instruction:   %incdec.ptr2 = getelementptr inbounds i8, i8* %pSrcB.addr.09, i32 1
; CHECK: LV: Found an estimated cost of 18 for VF 2 For instruction:   %1 = load i8, i8* %pSrcB.addr.09, align 1
; CHECK: LV: Found an estimated cost of 4 for VF 2 For instruction:   %conv3 = sext i8 %1 to i32
; CHECK: LV: Found an estimated cost of 26 for VF 2 For instruction:   %mul = mul nsw i32 %conv3, %conv1
; CHECK: LV: Found an estimated cost of 18 for VF 2 For instruction:   %shr = ashr i32 %mul, 7
; CHECK: LV: Found an estimated cost of 0 for VF 2 For instruction:   %2 = icmp slt i32 %shr, 127
; CHECK: LV: Found an estimated cost of 22 for VF 2 For instruction:   %spec.select.i = select i1 %2, i32 %shr, i32 127
; CHECK: LV: Found an estimated cost of 0 for VF 2 For instruction:   %conv4 = trunc i32 %spec.select.i to i8
; CHECK: LV: Found an estimated cost of 0 for VF 2 For instruction:   %incdec.ptr5 = getelementptr inbounds i8, i8* %pDst.addr.010, i32 1
; CHECK: LV: Found an estimated cost of 18 for VF 2 For instruction:   store i8 %conv4, i8* %pDst.addr.010, align 1
; CHECK: LV: Found an estimated cost of 1 for VF 2 For instruction:   %dec = add i32 %blkCnt.012, -1
; CHECK: LV: Found an estimated cost of 1 for VF 2 For instruction:   %cmp.not = icmp eq i32 %dec, 0
; CHECK: LV: Found an estimated cost of 0 for VF 2 For instruction:   br i1 %cmp.not, label %while.end.loopexit, label %while.body
; CHECK: LV: Vector loop of width 2 costs: 65.
; CHECK: LV: Found an estimated cost of 0 for VF 4 For instruction:   %blkCnt.012 = phi i32 [ %dec, %while.body ], [ %blockSize, %while.body.preheader ]
; CHECK: LV: Found an estimated cost of 0 for VF 4 For instruction:   %pSrcA.addr.011 = phi i8* [ %incdec.ptr, %while.body ], [ %pSrcA, %while.body.preheader ]
; CHECK: LV: Found an estimated cost of 0 for VF 4 For instruction:   %pDst.addr.010 = phi i8* [ %incdec.ptr5, %while.body ], [ %pDst, %while.body.preheader ]
; CHECK: LV: Found an estimated cost of 0 for VF 4 For instruction:   %pSrcB.addr.09 = phi i8* [ %incdec.ptr2, %while.body ], [ %pSrcB, %while.body.preheader ]
; CHECK: LV: Found an estimated cost of 0 for VF 4 For instruction:   %incdec.ptr = getelementptr inbounds i8, i8* %pSrcA.addr.011, i32 1
; CHECK: LV: Found an estimated cost of 2 for VF 4 For instruction:   %0 = load i8, i8* %pSrcA.addr.011, align 1
; CHECK: LV: Found an estimated cost of 0 for VF 4 For instruction:   %conv1 = sext i8 %0 to i32
; CHECK: LV: Found an estimated cost of 0 for VF 4 For instruction:   %incdec.ptr2 = getelementptr inbounds i8, i8* %pSrcB.addr.09, i32 1
; CHECK: LV: Found an estimated cost of 2 for VF 4 For instruction:   %1 = load i8, i8* %pSrcB.addr.09, align 1
; CHECK: LV: Found an estimated cost of 0 for VF 4 For instruction:   %conv3 = sext i8 %1 to i32
; CHECK: LV: Found an estimated cost of 2 for VF 4 For instruction:   %mul = mul nsw i32 %conv3, %conv1
; CHECK: LV: Found an estimated cost of 2 for VF 4 For instruction:   %shr = ashr i32 %mul, 7
; CHECK: LV: Found an estimated cost of 0 for VF 4 For instruction:   %2 = icmp slt i32 %shr, 127
; CHECK: LV: Found an estimated cost of 2 for VF 4 For instruction:   %spec.select.i = select i1 %2, i32 %shr, i32 127
; CHECK: LV: Found an estimated cost of 0 for VF 4 For instruction:   %conv4 = trunc i32 %spec.select.i to i8
; CHECK: LV: Found an estimated cost of 0 for VF 4 For instruction:   %incdec.ptr5 = getelementptr inbounds i8, i8* %pDst.addr.010, i32 1
; CHECK: LV: Found an estimated cost of 2 for VF 4 For instruction:   store i8 %conv4, i8* %pDst.addr.010, align 1
; CHECK: LV: Found an estimated cost of 1 for VF 4 For instruction:   %dec = add i32 %blkCnt.012, -1
; CHECK: LV: Found an estimated cost of 1 for VF 4 For instruction:   %cmp.not = icmp eq i32 %dec, 0
; CHECK: LV: Found an estimated cost of 0 for VF 4 For instruction:   br i1 %cmp.not, label %while.end.loopexit, label %while.body
; CHECK: LV: Vector loop of width 4 costs: 3.
; CHECK: LV: Found an estimated cost of 0 for VF 8 For instruction:   %blkCnt.012 = phi i32 [ %dec, %while.body ], [ %blockSize, %while.body.preheader ]
; CHECK: LV: Found an estimated cost of 0 for VF 8 For instruction:   %pSrcA.addr.011 = phi i8* [ %incdec.ptr, %while.body ], [ %pSrcA, %while.body.preheader ]
; CHECK: LV: Found an estimated cost of 0 for VF 8 For instruction:   %pDst.addr.010 = phi i8* [ %incdec.ptr5, %while.body ], [ %pDst, %while.body.preheader ]
; CHECK: LV: Found an estimated cost of 0 for VF 8 For instruction:   %pSrcB.addr.09 = phi i8* [ %incdec.ptr2, %while.body ], [ %pSrcB, %while.body.preheader ]
; CHECK: LV: Found an estimated cost of 0 for VF 8 For instruction:   %incdec.ptr = getelementptr inbounds i8, i8* %pSrcA.addr.011, i32 1
; CHECK: LV: Found an estimated cost of 2 for VF 8 For instruction:   %0 = load i8, i8* %pSrcA.addr.011, align 1
; CHECK: LV: Found an estimated cost of 2 for VF 8 For instruction:   %conv1 = sext i8 %0 to i32
; CHECK: LV: Found an estimated cost of 0 for VF 8 For instruction:   %incdec.ptr2 = getelementptr inbounds i8, i8* %pSrcB.addr.09, i32 1
; CHECK: LV: Found an estimated cost of 2 for VF 8 For instruction:   %1 = load i8, i8* %pSrcB.addr.09, align 1
; CHECK: LV: Found an estimated cost of 2 for VF 8 For instruction:   %conv3 = sext i8 %1 to i32
; CHECK: LV: Found an estimated cost of 4 for VF 8 For instruction:   %mul = mul nsw i32 %conv3, %conv1
; CHECK: LV: Found an estimated cost of 4 for VF 8 For instruction:   %shr = ashr i32 %mul, 7
; CHECK: LV: Found an estimated cost of 0 for VF 8 For instruction:   %2 = icmp slt i32 %shr, 127
; CHECK: LV: Found an estimated cost of 4 for VF 8 For instruction:   %spec.select.i = select i1 %2, i32 %shr, i32 127
; CHECK: LV: Found an estimated cost of 2 for VF 8 For instruction:   %conv4 = trunc i32 %spec.select.i to i8
; CHECK: LV: Found an estimated cost of 0 for VF 8 For instruction:   %incdec.ptr5 = getelementptr inbounds i8, i8* %pDst.addr.010, i32 1
; CHECK: LV: Found an estimated cost of 2 for VF 8 For instruction:   store i8 %conv4, i8* %pDst.addr.010, align 1
; CHECK: LV: Found an estimated cost of 1 for VF 8 For instruction:   %dec = add i32 %blkCnt.012, -1
; CHECK: LV: Found an estimated cost of 1 for VF 8 For instruction:   %cmp.not = icmp eq i32 %dec, 0
; CHECK: LV: Found an estimated cost of 0 for VF 8 For instruction:   br i1 %cmp.not, label %while.end.loopexit, label %while.body
; CHECK: LV: Vector loop of width 8 costs: 3.
; CHECK: LV: Found an estimated cost of 0 for VF 16 For instruction:   %blkCnt.012 = phi i32 [ %dec, %while.body ], [ %blockSize, %while.body.preheader ]
; CHECK: LV: Found an estimated cost of 0 for VF 16 For instruction:   %pSrcA.addr.011 = phi i8* [ %incdec.ptr, %while.body ], [ %pSrcA, %while.body.preheader ]
; CHECK: LV: Found an estimated cost of 0 for VF 16 For instruction:   %pDst.addr.010 = phi i8* [ %incdec.ptr5, %while.body ], [ %pDst, %while.body.preheader ]
; CHECK: LV: Found an estimated cost of 0 for VF 16 For instruction:   %pSrcB.addr.09 = phi i8* [ %incdec.ptr2, %while.body ], [ %pSrcB, %while.body.preheader ]
; CHECK: LV: Found an estimated cost of 0 for VF 16 For instruction:   %incdec.ptr = getelementptr inbounds i8, i8* %pSrcA.addr.011, i32 1
; CHECK: LV: Found an estimated cost of 2 for VF 16 For instruction:   %0 = load i8, i8* %pSrcA.addr.011, align 1
; CHECK: LV: Found an estimated cost of 6 for VF 16 For instruction:   %conv1 = sext i8 %0 to i32
; CHECK: LV: Found an estimated cost of 0 for VF 16 For instruction:   %incdec.ptr2 = getelementptr inbounds i8, i8* %pSrcB.addr.09, i32 1
; CHECK: LV: Found an estimated cost of 2 for VF 16 For instruction:   %1 = load i8, i8* %pSrcB.addr.09, align 1
; CHECK: LV: Found an estimated cost of 6 for VF 16 For instruction:   %conv3 = sext i8 %1 to i32
; CHECK: LV: Found an estimated cost of 8 for VF 16 For instruction:   %mul = mul nsw i32 %conv3, %conv1
; CHECK: LV: Found an estimated cost of 8 for VF 16 For instruction:   %shr = ashr i32 %mul, 7
; CHECK: LV: Found an estimated cost of 0 for VF 16 For instruction:   %2 = icmp slt i32 %shr, 127
; CHECK: LV: Found an estimated cost of 8 for VF 16 For instruction:   %spec.select.i = select i1 %2, i32 %shr, i32 127
; CHECK: LV: Found an estimated cost of 6 for VF 16 For instruction:   %conv4 = trunc i32 %spec.select.i to i8
; CHECK: LV: Found an estimated cost of 0 for VF 16 For instruction:   %incdec.ptr5 = getelementptr inbounds i8, i8* %pDst.addr.010, i32 1
; CHECK: LV: Found an estimated cost of 2 for VF 16 For instruction:   store i8 %conv4, i8* %pDst.addr.010, align 1
; CHECK: LV: Found an estimated cost of 1 for VF 16 For instruction:   %dec = add i32 %blkCnt.012, -1
; CHECK: LV: Found an estimated cost of 1 for VF 16 For instruction:   %cmp.not = icmp eq i32 %dec, 0
; CHECK: LV: Found an estimated cost of 0 for VF 16 For instruction:   br i1 %cmp.not, label %while.end.loopexit, label %while.body
; CHECK: LV: Vector loop of width 16 costs: 3.
; CHECK: LV: Selecting VF: 16.
define void @cheap_icmp(i8* nocapture readonly %pSrcA, i8* nocapture readonly %pSrcB, i8* nocapture %pDst, i32 %blockSize) #0 {
entry:
  %cmp.not8 = icmp eq i32 %blockSize, 0
  br i1 %cmp.not8, label %while.end, label %while.body.preheader

while.body.preheader:                             ; preds = %entry
  br label %while.body

while.body:                                       ; preds = %while.body.preheader, %while.body
  %blkCnt.012 = phi i32 [ %dec, %while.body ], [ %blockSize, %while.body.preheader ]
  %pSrcA.addr.011 = phi i8* [ %incdec.ptr, %while.body ], [ %pSrcA, %while.body.preheader ]
  %pDst.addr.010 = phi i8* [ %incdec.ptr5, %while.body ], [ %pDst, %while.body.preheader ]
  %pSrcB.addr.09 = phi i8* [ %incdec.ptr2, %while.body ], [ %pSrcB, %while.body.preheader ]
  %incdec.ptr = getelementptr inbounds i8, i8* %pSrcA.addr.011, i32 1
  %0 = load i8, i8* %pSrcA.addr.011, align 1
  %conv1 = sext i8 %0 to i32
  %incdec.ptr2 = getelementptr inbounds i8, i8* %pSrcB.addr.09, i32 1
  %1 = load i8, i8* %pSrcB.addr.09, align 1
  %conv3 = sext i8 %1 to i32
  %mul = mul nsw i32 %conv3, %conv1
  %shr = ashr i32 %mul, 7
  %2 = icmp slt i32 %shr, 127
  %spec.select.i = select i1 %2, i32 %shr, i32 127
  %conv4 = trunc i32 %spec.select.i to i8
  %incdec.ptr5 = getelementptr inbounds i8, i8* %pDst.addr.010, i32 1
  store i8 %conv4, i8* %pDst.addr.010, align 1
  %dec = add i32 %blkCnt.012, -1
  %cmp.not = icmp eq i32 %dec, 0
  br i1 %cmp.not, label %while.end.loopexit, label %while.body

while.end.loopexit:                               ; preds = %while.body
  br label %while.end

while.end:                                        ; preds = %while.end.loopexit, %entry
  ret void
}

; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %cmp1 = fcmp
; CHECK: LV: Found an estimated cost of 12 for VF 2 For instruction:   %cmp1 = fcmp
; CHECK: LV: Found an estimated cost of 24 for VF 4 For instruction:   %cmp1 = fcmp
define void @floatcmp(float* nocapture readonly %pSrc, i32* nocapture %pDst, i32 %blockSize) #0 {
entry:
  %cmp.not7 = icmp eq i32 %blockSize, 0
  br i1 %cmp.not7, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %while.body
  %pSrc.addr.010 = phi float* [ %incdec.ptr2, %while.body ], [ %pSrc, %entry ]
  %blockSize.addr.09 = phi i32 [ %dec, %while.body ], [ %blockSize, %entry ]
  %pDst.addr.08 = phi i32* [ %incdec.ptr, %while.body ], [ %pDst, %entry ]
  %0 = load float, float* %pSrc.addr.010, align 4
  %cmp1 = fcmp nnan ninf nsz olt float %0, 0.000000e+00
  %cond = select nnan ninf nsz i1 %cmp1, float 1.000000e+01, float %0
  %conv = fptosi float %cond to i32
  %incdec.ptr = getelementptr inbounds i32, i32* %pDst.addr.08, i32 1
  store i32 %conv, i32* %pDst.addr.08, align 4
  %incdec.ptr2 = getelementptr inbounds float, float* %pSrc.addr.010, i32 1
  %dec = add i32 %blockSize.addr.09, -1
  %cmp.not = icmp eq i32 %dec, 0
  br i1 %cmp.not, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  ret void
}

attributes #0 = { "target-features"="+mve" }
