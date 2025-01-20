; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8.1m.main-arm-none-eabi"

; CHECK-LABEL: LV: Checking a loop in 'expensive_icmp'
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %i.016 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.inc ]
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %arrayidx = getelementptr inbounds i16, ptr %s, i32 %i.016
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %1 = load i16, ptr %arrayidx, align 2
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %conv = sext i16 %1 to i32
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %cmp2 = icmp sgt i32 %conv, %conv1
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   br i1 %cmp2, label %if.then, label %for.inc
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %conv6 = add i16 %1, %0
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %arrayidx7 = getelementptr inbounds i16, ptr %d, i32 %i.016
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   store i16 %conv6, ptr %arrayidx7, align 2
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   br label %for.inc
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %inc = add nuw nsw i32 %i.016, 1
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %exitcond.not = icmp eq i32 %inc, %n
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body
; CHECK: LV: Scalar loop costs: 5.
; CHECK: Cost of 1 for VF 2: induction instruction   %inc = add nuw nsw i32 %i.016, 1
; CHECK: Cost of 0 for VF 2: induction instruction   %i.016 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.inc ]
; CHECK: Cost of 1 for VF 2: exit condition instruction   %exitcond.not = icmp eq i32 %inc, %n
; CHECK: Cost of 0 for VF 2: EMIT vp<%2> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; CHECK: Cost of 0 for VF 2: vp<%3> = SCALAR-STEPS vp<%2>, ir<1>
; CHECK: Cost of 0 for VF 2: CLONE ir<%arrayidx> = getelementptr inbounds ir<%s>, vp<%3>
; CHECK: Cost of 0 for VF 2: vp<%4> = vector-pointer ir<%arrayidx>
; CHECK: Cost of 18 for VF 2: WIDEN ir<%1> = load vp<%4>
; CHECK: Cost of 4 for VF 2: WIDEN-CAST ir<%conv> = sext ir<%1> to i32
; CHECK: Cost of 20 for VF 2: WIDEN ir<%cmp2> = icmp sgt ir<%conv>, ir<%conv1>
; CHECK: Cost of 26 for VF 2: WIDEN ir<%conv6> = add ir<%1>, ir<%0>
; CHECK: Cost of 0 for VF 2: CLONE ir<%arrayidx7> = getelementptr ir<%d>, vp<%3>
; CHECK: Cost of 0 for VF 2: vp<%5> = vector-pointer ir<%arrayidx7>
; CHECK: Cost of 16 for VF 2: WIDEN store vp<%5>, ir<%conv6>, ir<%cmp2>
; CHECK: Cost of 0 for VF 2: EMIT vp<%index.next> = add nuw vp<%2>, vp<%0>
; CHECK: Cost of 0 for VF 2: EMIT branch-on-count vp<%index.next>, vp<%1>
; CHECK: Cost for VF 2: 86 (Estimated cost per lane: 43.
; CHECK: Cost of 1 for VF 4: induction instruction   %inc = add nuw nsw i32 %i.016, 1
; CHECK: Cost of 0 for VF 4: induction instruction   %i.016 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.inc ]
; CHECK: Cost of 1 for VF 4: exit condition instruction   %exitcond.not = icmp eq i32 %inc, %n
; CHECK: Cost of 0 for VF 4: EMIT vp<%2> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; CHECK: Cost of 0 for VF 4: vp<%3> = SCALAR-STEPS vp<%2>, ir<1>
; CHECK: Cost of 0 for VF 4: CLONE ir<%arrayidx> = getelementptr inbounds ir<%s>, vp<%3>
; CHECK: Cost of 0 for VF 4: vp<%4> = vector-pointer ir<%arrayidx>
; CHECK: Cost of 2 for VF 4: WIDEN ir<%1> = load vp<%4>
; CHECK: Cost of 0 for VF 4: WIDEN-CAST ir<%conv> = sext ir<%1> to i32
; CHECK: Cost of 2 for VF 4: WIDEN ir<%cmp2> = icmp sgt ir<%conv>, ir<%conv1>
; CHECK: Cost of 2 for VF 4: WIDEN ir<%conv6> = add ir<%1>, ir<%0>
; CHECK: Cost of 0 for VF 4: CLONE ir<%arrayidx7> = getelementptr ir<%d>, vp<%3>
; CHECK: Cost of 0 for VF 4: vp<%5> = vector-pointer ir<%arrayidx7>
; CHECK: Cost of 2 for VF 4: WIDEN store vp<%5>, ir<%conv6>, ir<%cmp2>
; CHECK: Cost of 0 for VF 4: EMIT vp<%index.next> = add nuw vp<%2>, vp<%0>
; CHECK: Cost of 0 for VF 4: EMIT branch-on-count vp<%index.next>, vp<%1>
; CHECK: Cost for VF 4: 10 (Estimated cost per lane: 2.
; CHECK: Cost of 1 for VF 8: induction instruction   %inc = add nuw nsw i32 %i.016, 1
; CHECK: Cost of 0 for VF 8: induction instruction   %i.016 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.inc ]
; CHECK: Cost of 1 for VF 8: exit condition instruction   %exitcond.not = icmp eq i32 %inc, %n
; CHECK: Cost of 0 for VF 8: EMIT vp<%2> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; CHECK: Cost of 0 for VF 8: vp<%3> = SCALAR-STEPS vp<%2>, ir<1>
; CHECK: Cost of 0 for VF 8: CLONE ir<%arrayidx> = getelementptr inbounds ir<%s>, vp<%3>
; CHECK: Cost of 0 for VF 8: vp<%4> = vector-pointer ir<%arrayidx>
; CHECK: Cost of 2 for VF 8: WIDEN ir<%1> = load vp<%4>
; CHECK: Cost of 2 for VF 8: WIDEN-CAST ir<%conv> = sext ir<%1> to i32
; CHECK: Cost of 36 for VF 8: WIDEN ir<%cmp2> = icmp sgt ir<%conv>, ir<%conv1>
; CHECK: Cost of 2 for VF 8: WIDEN ir<%conv6> = add ir<%1>, ir<%0>
; CHECK: Cost of 0 for VF 8: CLONE ir<%arrayidx7> = getelementptr ir<%d>, vp<%3>
; CHECK: Cost of 0 for VF 8: vp<%5> = vector-pointer ir<%arrayidx7>
; CHECK: Cost of 2 for VF 8: WIDEN store vp<%5>, ir<%conv6>, ir<%cmp2>
; CHECK: Cost of 0 for VF 8: EMIT vp<%index.next> = add nuw vp<%2>, vp<%0>
; CHECK: Cost of 0 for VF 8: EMIT branch-on-count vp<%index.next>, vp<%1>
; CHECK: Cost for VF 8: 46 (Estimated cost per lane: 5.
; CHECK: LV: Selecting VF: 4.
define void @expensive_icmp(ptr noalias nocapture %d, ptr nocapture readonly %s, i32 %n, i16 zeroext %m) #0 {
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
  %arrayidx = getelementptr inbounds i16, ptr %s, i32 %i.016
  %1 = load i16, ptr %arrayidx, align 2
  %conv = sext i16 %1 to i32
  %cmp2 = icmp sgt i32 %conv, %conv1
  br i1 %cmp2, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %conv6 = add i16 %1, %0
  %arrayidx7 = getelementptr inbounds i16, ptr %d, i32 %i.016
  store i16 %conv6, ptr %arrayidx7, align 2
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %inc = add nuw nsw i32 %i.016, 1
  %exitcond.not = icmp eq i32 %inc, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: LV: Checking a loop in 'cheap_icmp'
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %blkCnt.012 = phi i32 [ %dec, %while.body ], [ %blockSize, %while.body.preheader ]
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %pSrcA.addr.011 = phi ptr [ %incdec.ptr, %while.body ], [ %pSrcA, %while.body.preheader ]
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %pDst.addr.010 = phi ptr [ %incdec.ptr5, %while.body ], [ %pDst, %while.body.preheader ]
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %pSrcB.addr.09 = phi ptr [ %incdec.ptr2, %while.body ], [ %pSrcB, %while.body.preheader ]
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %incdec.ptr = getelementptr inbounds i8, ptr %pSrcA.addr.011, i32 1
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %0 = load i8, ptr %pSrcA.addr.011, align 1
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %conv1 = sext i8 %0 to i32
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %incdec.ptr2 = getelementptr inbounds i8, ptr %pSrcB.addr.09, i32 1
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %1 = load i8, ptr %pSrcB.addr.09, align 1
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %conv3 = sext i8 %1 to i32
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %mul = mul nsw i32 %conv3, %conv1
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %shr = ashr i32 %mul, 7
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %2 = icmp slt i32 %shr, 127
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %spec.select.i = select i1 %2, i32 %shr, i32 127
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %conv4 = trunc i32 %spec.select.i to i8
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %incdec.ptr5 = getelementptr inbounds i8, ptr %pDst.addr.010, i32 1
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   store i8 %conv4, ptr %pDst.addr.010, align 1
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %dec = add i32 %blkCnt.012, -1
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %cmp.not = icmp eq i32 %dec, 0
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   br i1 %cmp.not, label %while.end.loopexit, label %while.body
; CHECK: LV: Scalar loop costs: 9.
; CHECK: Cost of 1 for VF 2: induction instruction   %dec = add i32 %blkCnt.012, -1
; CHECK: Cost of 0 for VF 2: induction instruction   %blkCnt.012 = phi i32 [ %dec, %while.body ], [ %blockSize, %while.body.preheader ]
; CHECK: Cost of 0 for VF 2: induction instruction   %incdec.ptr = getelementptr inbounds i8, ptr %pSrcA.addr.011, i32 1
; CHECK: Cost of 0 for VF 2: induction instruction   %pSrcA.addr.011 = phi ptr [ %incdec.ptr, %while.body ], [ %pSrcA, %while.body.preheader ]
; CHECK: Cost of 0 for VF 2: induction instruction   %incdec.ptr5 = getelementptr inbounds i8, ptr %pDst.addr.010, i32 1
; CHECK: Cost of 0 for VF 2: induction instruction   %pDst.addr.010 = phi ptr [ %incdec.ptr5, %while.body ], [ %pDst, %while.body.preheader ]
; CHECK: Cost of 0 for VF 2: induction instruction   %incdec.ptr2 = getelementptr inbounds i8, ptr %pSrcB.addr.09, i32 1
; CHECK: Cost of 0 for VF 2: induction instruction   %pSrcB.addr.09 = phi ptr [ %incdec.ptr2, %while.body ], [ %pSrcB, %while.body.preheader ]
; CHECK: Cost of 1 for VF 2: exit condition instruction   %cmp.not = icmp eq i32 %dec, 0
; CHECK: Cost of 0 for VF 2: EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; CHECK: Cost of 0 for VF 2: vp<[[STEPS1:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK: Cost of 0 for VF 2: EMIT vp<%next.gep> = ptradd ir<%pSrcA>, vp<[[STEPS1]]>
; CHECK: Cost of 0 for VF 2: vp<[[STEPS2:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK: Cost of 0 for VF 2: EMIT vp<%next.gep>.1 = ptradd ir<%pDst>, vp<[[STEPS2]]>
; CHECK: Cost of 0 for VF 2: vp<[[STEPS3:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK: Cost of 0 for VF 2: EMIT vp<%next.gep>.2 = ptradd ir<%pSrcB>, vp<[[STEPS3]]>
; CHECK: Cost of 0 for VF 2: vp<[[VEC_PTR:%.+]]> = vector-pointer vp<%next.gep>
; CHECK: Cost of 18 for VF 2: WIDEN ir<%0> = load vp<[[VEC_PTR]]>
; CHECK: Cost of 4 for VF 2: WIDEN-CAST ir<%conv1> = sext ir<%0> to i32
; CHECK: Cost of 0 for VF 2: vp<[[VEC_PTR2:%.+]]> = vector-pointer vp<%next.gep>.2
; CHECK: Cost of 18 for VF 2: WIDEN ir<%1> = load vp<[[VEC_PTR2]]>
; CHECK: Cost of 4 for VF 2: WIDEN-CAST ir<%conv3> = sext ir<%1> to i32
; CHECK: Cost of 26 for VF 2: WIDEN ir<%mul> = mul nsw ir<%conv3>, ir<%conv1>
; CHECK: Cost of 18 for VF 2: WIDEN ir<%shr> = ashr ir<%mul>, ir<7>
; CHECK: Cost of 0 for VF 2: WIDEN ir<%2> = icmp slt ir<%shr>, ir<127>
; CHECK: Cost of 22 for VF 2: WIDEN-SELECT ir<%spec.select.i> = select ir<%2>, ir<%shr>, ir<127>
; CHECK: Cost of 0 for VF 2: WIDEN-CAST ir<%conv4> = trunc ir<%spec.select.i> to i8
; CHECK: Cost of 0 for VF 2: vp<[[VEC_PTR3:%.+]]> = vector-pointer vp<%next.gep>.1
; CHECK: Cost of 18 for VF 2: WIDEN store vp<[[VEC_PTR3]]>, ir<%conv4>
; CHECK: Cost of 0 for VF 2: EMIT vp<%index.next> = add nuw vp<[[CAN_IV]]>, vp<%0>
; CHECK: Cost of 0 for VF 2: EMIT branch-on-count vp<%index.next>, vp<{{.+}}>
; CHECK: Cost for VF 2: 130 (Estimated cost per lane: 65.
; CHECK: Cost of 1 for VF 4: induction instruction   %dec = add i32 %blkCnt.012, -1
; CHECK: Cost of 0 for VF 4: induction instruction   %blkCnt.012 = phi i32 [ %dec, %while.body ], [ %blockSize, %while.body.preheader ]
; CHECK: Cost of 0 for VF 4: induction instruction   %incdec.ptr = getelementptr inbounds i8, ptr %pSrcA.addr.011, i32 1
; CHECK: Cost of 0 for VF 4: induction instruction   %pSrcA.addr.011 = phi ptr [ %incdec.ptr, %while.body ], [ %pSrcA, %while.body.preheader ]
; CHECK: Cost of 0 for VF 4: induction instruction   %incdec.ptr5 = getelementptr inbounds i8, ptr %pDst.addr.010, i32 1
; CHECK: Cost of 0 for VF 4: induction instruction   %pDst.addr.010 = phi ptr [ %incdec.ptr5, %while.body ], [ %pDst, %while.body.preheader ]
; CHECK: Cost of 0 for VF 4: induction instruction   %incdec.ptr2 = getelementptr inbounds i8, ptr %pSrcB.addr.09, i32 1
; CHECK: Cost of 0 for VF 4: induction instruction   %pSrcB.addr.09 = phi ptr [ %incdec.ptr2, %while.body ], [ %pSrcB, %while.body.preheader ]
; CHECK: Cost of 1 for VF 4: exit condition instruction   %cmp.not = icmp eq i32 %dec, 0
; CHECK: Cost of 0 for VF 4: EMIT vp<[[CAN_IV:%.]]> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; CHECK: Cost of 0 for VF 4: vp<[[STEPS1:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK: Cost of 0 for VF 4: EMIT vp<%next.gep> = ptradd ir<%pSrcA>, vp<[[STEPS1]]>
; CHECK: Cost of 0 for VF 4: vp<[[STEPS2:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK: Cost of 0 for VF 4: EMIT vp<%next.gep>.1 = ptradd ir<%pDst>, vp<[[STEPS2]]>
; CHECK: Cost of 0 for VF 4: vp<[[STEPS3:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK: Cost of 0 for VF 4: EMIT vp<%next.gep>.2 = ptradd ir<%pSrcB>, vp<[[STEPS3]]>
; CHECK: Cost of 0 for VF 4: vp<[[VEC_PTR1:%.+]]> = vector-pointer vp<%next.gep>
; CHECK: Cost of 2 for VF 4: WIDEN ir<%0> = load vp<[[VEC_PTR1]]>
; CHECK: Cost of 0 for VF 4: WIDEN-CAST ir<%conv1> = sext ir<%0> to i32
; CHECK: Cost of 0 for VF 4: vp<[[VEC_PTR2:%.+]]> = vector-pointer vp<%next.gep>.2
; CHECK: Cost of 2 for VF 4: WIDEN ir<%1> = load vp<[[VEC_PTR2]]>
; CHECK: Cost of 0 for VF 4: WIDEN-CAST ir<%conv3> = sext ir<%1> to i32
; CHECK: Cost of 2 for VF 4: WIDEN ir<%mul> = mul nsw ir<%conv3>, ir<%conv1>
; CHECK: Cost of 2 for VF 4: WIDEN ir<%shr> = ashr ir<%mul>, ir<7>
; CHECK: Cost of 0 for VF 4: WIDEN ir<%2> = icmp slt ir<%shr>, ir<127>
; CHECK: Cost of 2 for VF 4: WIDEN-SELECT ir<%spec.select.i> = select ir<%2>, ir<%shr>, ir<127>
; CHECK: Cost of 0 for VF 4: WIDEN-CAST ir<%conv4> = trunc ir<%spec.select.i> to i8
; CHECK: Cost of 0 for VF 4: vp<[[VEC_PTR2:%.+]]> = vector-pointer vp<%next.gep>.1
; CHECK: Cost of 2 for VF 4: WIDEN store vp<[[VEC_PTR2]]>, ir<%conv4>
; CHECK: Cost of 0 for VF 4: EMIT vp<%index.next> = add nuw vp<[[CAN_IV]]>, vp<%0>
; CHECK: Cost of 0 for VF 4: EMIT branch-on-count vp<%index.next>, vp<{{.+}}>
; CHECK: Cost for VF 4: 14 (Estimated cost per lane: 3.
; CHECK: Cost of 1 for VF 8: induction instruction   %dec = add i32 %blkCnt.012, -1
; CHECK: Cost of 0 for VF 8: induction instruction   %blkCnt.012 = phi i32 [ %dec, %while.body ], [ %blockSize, %while.body.preheader ]
; CHECK: Cost of 0 for VF 8: induction instruction   %incdec.ptr = getelementptr inbounds i8, ptr %pSrcA.addr.011, i32 1
; CHECK: Cost of 0 for VF 8: induction instruction   %pSrcA.addr.011 = phi ptr [ %incdec.ptr, %while.body ], [ %pSrcA, %while.body.preheader ]
; CHECK: Cost of 0 for VF 8: induction instruction   %incdec.ptr5 = getelementptr inbounds i8, ptr %pDst.addr.010, i32 1
; CHECK: Cost of 0 for VF 8: induction instruction   %pDst.addr.010 = phi ptr [ %incdec.ptr5, %while.body ], [ %pDst, %while.body.preheader ]
; CHECK: Cost of 0 for VF 8: induction instruction   %incdec.ptr2 = getelementptr inbounds i8, ptr %pSrcB.addr.09, i32 1
; CHECK: Cost of 0 for VF 8: induction instruction   %pSrcB.addr.09 = phi ptr [ %incdec.ptr2, %while.body ], [ %pSrcB, %while.body.preheader ]
; CHECK: Cost of 1 for VF 8: exit condition instruction   %cmp.not = icmp eq i32 %dec, 0
; CHECK: Cost of 0 for VF 8: EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; CHECK: Cost of 0 for VF 8: vp<[[STEPS1:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK: Cost of 0 for VF 8: EMIT vp<%next.gep> = ptradd ir<%pSrcA>, vp<[[STEPS1]]>
; CHECK: Cost of 0 for VF 8: vp<[[STEPS2:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK: Cost of 0 for VF 8: EMIT vp<%next.gep>.1 = ptradd ir<%pDst>, vp<[[STEPS2]]>
; CHECK: Cost of 0 for VF 8: vp<[[STEPS3:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK: Cost of 0 for VF 8: EMIT vp<%next.gep>.2 = ptradd ir<%pSrcB>, vp<[[STEPS3]]>
; CHECK: Cost of 0 for VF 8: vp<[[VEC_PTR1:%.+]]> = vector-pointer vp<%next.gep>
; CHECK: Cost of 2 for VF 8: WIDEN ir<%0> = load vp<[[VEC_PTR1]]>
; CHECK: Cost of 2 for VF 8: WIDEN-CAST ir<%conv1> = sext ir<%0> to i32
; CHECK: Cost of 0 for VF 8: vp<[[VEC_PTR2:%.+]]> = vector-pointer vp<%next.gep>.2
; CHECK: Cost of 2 for VF 8: WIDEN ir<%1> = load vp<[[VEC_PTR2]]>
; CHECK: Cost of 2 for VF 8: WIDEN-CAST ir<%conv3> = sext ir<%1> to i32
; CHECK: Cost of 4 for VF 8: WIDEN ir<%mul> = mul nsw ir<%conv3>, ir<%conv1>
; CHECK: Cost of 4 for VF 8: WIDEN ir<%shr> = ashr ir<%mul>, ir<7>
; CHECK: Cost of 0 for VF 8: WIDEN ir<%2> = icmp slt ir<%shr>, ir<127>
; CHECK: Cost of 4 for VF 8: WIDEN-SELECT ir<%spec.select.i> = select ir<%2>, ir<%shr>, ir<127>
; CHECK: Cost of 2 for VF 8: WIDEN-CAST ir<%conv4> = trunc ir<%spec.select.i> to i8
; CHECK: Cost of 0 for VF 8: vp<[[VEC_PTR3:%.+]]> = vector-pointer vp<%next.gep>.1
; CHECK: Cost of 2 for VF 8: WIDEN store vp<[[VEC_PTR3]]>, ir<%conv4>
; CHECK: Cost of 0 for VF 8: EMIT vp<%index.next> = add nuw vp<[[CAN_IV]]>, vp<{{.+}}
; CHECK: Cost of 0 for VF 8: EMIT branch-on-count vp<%index.next>, vp<{{.+}}>
; CHECK: Cost for VF 8: 26 (Estimated cost per lane: 3.
; CHECK: Cost of 1 for VF 16: induction instruction   %dec = add i32 %blkCnt.012, -1
; CHECK: Cost of 0 for VF 16: induction instruction   %blkCnt.012 = phi i32 [ %dec, %while.body ], [ %blockSize, %while.body.preheader ]
; CHECK: Cost of 0 for VF 16: induction instruction   %incdec.ptr = getelementptr inbounds i8, ptr %pSrcA.addr.011, i32 1
; CHECK: Cost of 0 for VF 16: induction instruction   %pSrcA.addr.011 = phi ptr [ %incdec.ptr, %while.body ], [ %pSrcA, %while.body.preheader ]
; CHECK: Cost of 0 for VF 16: induction instruction   %incdec.ptr5 = getelementptr inbounds i8, ptr %pDst.addr.010, i32 1
; CHECK: Cost of 0 for VF 16: induction instruction   %pDst.addr.010 = phi ptr [ %incdec.ptr5, %while.body ], [ %pDst, %while.body.preheader ]
; CHECK: Cost of 0 for VF 16: induction instruction   %incdec.ptr2 = getelementptr inbounds i8, ptr %pSrcB.addr.09, i32 1
; CHECK: Cost of 0 for VF 16: induction instruction   %pSrcB.addr.09 = phi ptr [ %incdec.ptr2, %while.body ], [ %pSrcB, %while.body.preheader ]
; CHECK: Cost of 1 for VF 16: exit condition instruction   %cmp.not = icmp eq i32 %dec, 0
; CHECK: Cost of 0 for VF 16: EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; CHECK: Cost of 0 for VF 16: vp<[[STEPS1:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK: Cost of 0 for VF 16: EMIT vp<%next.gep> = ptradd ir<%pSrcA>, vp<[[STEPS1]]>
; CHECK: Cost of 0 for VF 16: vp<[[STEPS2:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK: Cost of 0 for VF 16: EMIT vp<%next.gep>.1 = ptradd ir<%pDst>, vp<[[STEPS2]]>
; CHECK: Cost of 0 for VF 16: vp<[[STEPS3:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK: Cost of 0 for VF 16: EMIT vp<%next.gep>.2 = ptradd ir<%pSrcB>, vp<[[STEPS3]]>
; CHECK: Cost of 0 for VF 16: vp<[[VEC_PTR:%.+]]> = vector-pointer vp<%next.gep>
; CHECK: Cost of 2 for VF 16: WIDEN ir<%0> = load vp<[[VEC_PTR]]>
; CHECK: Cost of 6 for VF 16: WIDEN-CAST ir<%conv1> = sext ir<%0> to i32
; CHECK: Cost of 0 for VF 16: vp<[[VEC_PTR1:%.+]]> = vector-pointer vp<%next.gep>.2
; CHECK: Cost of 2 for VF 16: WIDEN ir<%1> = load vp<[[VEC_PTR1]]>
; CHECK: Cost of 6 for VF 16: WIDEN-CAST ir<%conv3> = sext ir<%1> to i32
; CHECK: Cost of 8 for VF 16: WIDEN ir<%mul> = mul nsw ir<%conv3>, ir<%conv1>
; CHECK: Cost of 8 for VF 16: WIDEN ir<%shr> = ashr ir<%mul>, ir<7>
; CHECK: Cost of 0 for VF 16: WIDEN ir<%2> = icmp slt ir<%shr>, ir<127>
; CHECK: Cost of 8 for VF 16: WIDEN-SELECT ir<%spec.select.i> = select ir<%2>, ir<%shr>, ir<127>
; CHECK: Cost of 6 for VF 16: WIDEN-CAST ir<%conv4> = trunc ir<%spec.select.i> to i8
; CHECK: Cost of 0 for VF 16: vp<[[VEC_PTR2:%.+]]> = vector-pointer vp<%next.gep>.1
; CHECK: Cost of 2 for VF 16: WIDEN store vp<[[VEC_PTR2]]>, ir<%conv4>
; CHECK: Cost of 0 for VF 16: EMIT vp<%index.next> = add nuw vp<[[CAN_IV]]>, vp<{{.+}}>
; CHECK: Cost of 0 for VF 16: EMIT branch-on-count vp<%index.next>, vp<{{.+}}>
; CHECK: Cost for VF 16: 50
; CHECK: LV: Selecting VF: 16.
define void @cheap_icmp(ptr nocapture readonly %pSrcA, ptr nocapture readonly %pSrcB, ptr nocapture %pDst, i32 %blockSize) #0 {
entry:
  %cmp.not8 = icmp eq i32 %blockSize, 0
  br i1 %cmp.not8, label %while.end, label %while.body.preheader

while.body.preheader:                             ; preds = %entry
  br label %while.body

while.body:                                       ; preds = %while.body.preheader, %while.body
  %blkCnt.012 = phi i32 [ %dec, %while.body ], [ %blockSize, %while.body.preheader ]
  %pSrcA.addr.011 = phi ptr [ %incdec.ptr, %while.body ], [ %pSrcA, %while.body.preheader ]
  %pDst.addr.010 = phi ptr [ %incdec.ptr5, %while.body ], [ %pDst, %while.body.preheader ]
  %pSrcB.addr.09 = phi ptr [ %incdec.ptr2, %while.body ], [ %pSrcB, %while.body.preheader ]
  %incdec.ptr = getelementptr inbounds i8, ptr %pSrcA.addr.011, i32 1
  %0 = load i8, ptr %pSrcA.addr.011, align 1
  %conv1 = sext i8 %0 to i32
  %incdec.ptr2 = getelementptr inbounds i8, ptr %pSrcB.addr.09, i32 1
  %1 = load i8, ptr %pSrcB.addr.09, align 1
  %conv3 = sext i8 %1 to i32
  %mul = mul nsw i32 %conv3, %conv1
  %shr = ashr i32 %mul, 7
  %2 = icmp slt i32 %shr, 127
  %spec.select.i = select i1 %2, i32 %shr, i32 127
  %conv4 = trunc i32 %spec.select.i to i8
  %incdec.ptr5 = getelementptr inbounds i8, ptr %pDst.addr.010, i32 1
  store i8 %conv4, ptr %pDst.addr.010, align 1
  %dec = add i32 %blkCnt.012, -1
  %cmp.not = icmp eq i32 %dec, 0
  br i1 %cmp.not, label %while.end.loopexit, label %while.body

while.end.loopexit:                               ; preds = %while.body
  br label %while.end

while.end:                                        ; preds = %while.end.loopexit, %entry
  ret void
}

; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %cmp1 = fcmp
; CHECK: Cost of 12 for VF 2: WIDEN ir<%cmp1> = fcmp olt ir<%0>, ir<0.000000e+00>
; CHECK: Cost of 24 for VF 4: WIDEN ir<%cmp1> = fcmp olt ir<%0>, ir<0.000000e+00>
define void @floatcmp(ptr nocapture readonly %pSrc, ptr nocapture %pDst, i32 %blockSize) #0 {
entry:
  %cmp.not7 = icmp eq i32 %blockSize, 0
  br i1 %cmp.not7, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %while.body
  %pSrc.addr.010 = phi ptr [ %incdec.ptr2, %while.body ], [ %pSrc, %entry ]
  %blockSize.addr.09 = phi i32 [ %dec, %while.body ], [ %blockSize, %entry ]
  %pDst.addr.08 = phi ptr [ %incdec.ptr, %while.body ], [ %pDst, %entry ]
  %0 = load float, ptr %pSrc.addr.010, align 4
  %cmp1 = fcmp nnan ninf nsz olt float %0, 0.000000e+00
  %cond = select nnan ninf nsz i1 %cmp1, float 1.000000e+01, float %0
  %conv = fptosi float %cond to i32
  %incdec.ptr = getelementptr inbounds i32, ptr %pDst.addr.08, i32 1
  store i32 %conv, ptr %pDst.addr.08, align 4
  %incdec.ptr2 = getelementptr inbounds float, ptr %pSrc.addr.010, i32 1
  %dec = add i32 %blockSize.addr.09, -1
  %cmp.not = icmp eq i32 %dec, 0
  br i1 %cmp.not, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  ret void
}

attributes #0 = { "target-features"="+mve" }
