; RUN: opt -loop-vectorize -enable-interleaved-mem-accesses -prefer-predicate-over-epilogue=predicate-dont-vectorize -S -mcpu=skx --debug-only=loop-vectorize < %s 2>&1 | FileCheck %s --check-prefix=DISABLED_MASKED_STRIDED
; RUN: opt -loop-vectorize -enable-interleaved-mem-accesses -enable-masked-interleaved-mem-accesses -prefer-predicate-over-epilogue=predicate-dont-vectorize -S -mcpu=skx --debug-only=loop-vectorize < %s 2>&1 | FileCheck %s --check-prefix=ENABLED_MASKED_STRIDED
; REQUIRES: asserts

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; (1) Interleave-group with factor 4, storing only 2 members out of the 4.
; Check that when we allow masked-memops to support interleave-group with gaps,
; the store is vectorized using a wide masked store, with a 1,1,0,0,1,1,0,0,... mask.
; Check that when we don't allow masked-memops to support interleave-group with gaps,
; the store is scalarized.
; The input IR was generated from this source:
;     for(i=0;i<1024;i++){
;       x[i] = points[i*4];
;       y[i] = points[i*4 + 1];
;     }
; (relates to the testcase in PR50566)

; DISABLED_MASKED_STRIDED: LV: Checking a loop in "test1"
;
; DISABLED_MASKED_STRIDED: LV: Found an estimated cost of 1 for VF 1 For instruction:   %i2 = load i16, i16* %arrayidx2, align 2
; DISABLED_MASKED_STRIDED: LV: Found an estimated cost of 1 for VF 1 For instruction:   %i4 = load i16, i16* %arrayidx7, align 2
;
; DISABLED_MASKED_STRIDED: LV: Found an estimated cost of 6 for VF 2 For instruction:   %i2 = load i16, i16* %arrayidx2, align 2
; DISABLED_MASKED_STRIDED: LV: Found an estimated cost of 6 for VF 2 For instruction:   %i4 = load i16, i16* %arrayidx7, align 2
;
; DISABLED_MASKED_STRIDED: LV: Found an estimated cost of 14 for VF 4 For instruction:   %i2 = load i16, i16* %arrayidx2, align 2
; DISABLED_MASKED_STRIDED: LV: Found an estimated cost of 14 for VF 4 For instruction:   %i4 = load i16, i16* %arrayidx7, align 2
;
; DISABLED_MASKED_STRIDED: LV: Found an estimated cost of 30 for VF 8 For instruction:   %i2 = load i16, i16* %arrayidx2, align 2
; DISABLED_MASKED_STRIDED: LV: Found an estimated cost of 30 for VF 8 For instruction:   %i4 = load i16, i16* %arrayidx7, align 2
;
; DISABLED_MASKED_STRIDED: LV: Found an estimated cost of 62 for VF 16 For instruction:   %i2 = load i16, i16* %arrayidx2, align 2
; DISABLED_MASKED_STRIDED: LV: Found an estimated cost of 62 for VF 16 For instruction:   %i4 = load i16, i16* %arrayidx7, align 2

; ENABLED_MASKED_STRIDED: LV: Checking a loop in "test1"
;
; ENABLED_MASKED_STRIDED: LV: Found an estimated cost of 1 for VF 1 For instruction:   %i2 = load i16, i16* %arrayidx2, align 2
; ENABLED_MASKED_STRIDED: LV: Found an estimated cost of 1 for VF 1 For instruction:   %i4 = load i16, i16* %arrayidx7, align 2
;
; ENABLED_MASKED_STRIDED: LV: Found an estimated cost of 8 for VF 2 For instruction:   %i2 = load i16, i16* %arrayidx2, align 2
; ENABLED_MASKED_STRIDED: LV: Found an estimated cost of 0 for VF 2 For instruction:   %i4 = load i16, i16* %arrayidx7, align 2
;
; ENABLED_MASKED_STRIDED: LV: Found an estimated cost of 11 for VF 4 For instruction:   %i2 = load i16, i16* %arrayidx2, align 2
; ENABLED_MASKED_STRIDED: LV: Found an estimated cost of 0 for VF 4 For instruction:   %i4 = load i16, i16* %arrayidx7, align 2
;
; ENABLED_MASKED_STRIDED: LV: Found an estimated cost of 11 for VF 8 For instruction:   %i2 = load i16, i16* %arrayidx2, align 2
; ENABLED_MASKED_STRIDED: LV: Found an estimated cost of 0 for VF 8 For instruction:   %i4 = load i16, i16* %arrayidx7, align 2
;
; ENABLED_MASKED_STRIDED: LV: Found an estimated cost of 17 for VF 16 For instruction:   %i2 = load i16, i16* %arrayidx2, align 2
; ENABLED_MASKED_STRIDED: LV: Found an estimated cost of 0 for VF 16 For instruction:   %i4 = load i16, i16* %arrayidx7, align 2

define void @test1(i16* noalias nocapture %points, i16* noalias nocapture readonly %x, i16* noalias nocapture readonly %y) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %i1 = shl nuw nsw i64 %indvars.iv, 2
  %arrayidx2 = getelementptr inbounds i16, i16* %points, i64 %i1
  %i2 = load i16, i16* %arrayidx2, align 2
  %i3 = or i64 %i1, 1
  %arrayidx7 = getelementptr inbounds i16, i16* %points, i64 %i3
  %i4 = load i16, i16* %arrayidx7, align 2
  %arrayidx = getelementptr inbounds i16, i16* %x, i64 %indvars.iv
  store i16 %i2, i16* %arrayidx, align 2
  %arrayidx4 = getelementptr inbounds i16, i16* %y, i64 %indvars.iv
  store i16 %i4, i16* %arrayidx4, align 2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:
  ret void
}

; (2) Same as above, but this time the gaps mask of the store is also And-ed with the
; fold-tail mask. If using masked memops to vectorize interleaved-group with gaps is
; not allowed, the store is scalarized and predicated.
; The input IR was generated from this source:
;     for(i=0;i<numPoints;i++){
;       x[i] = points[i*4];
;       y[i] = points[i*4 + 1];
;     }

; DISABLED_MASKED_STRIDED: LV: Checking a loop in "test2"
;
; DISABLED_MASKED_STRIDED: LV: Found an estimated cost of 1 for VF 1 For instruction:   %i2 = load i16, i16* %arrayidx2, align 2
; DISABLED_MASKED_STRIDED: LV: Found an estimated cost of 1 for VF 1 For instruction:   %i4 = load i16, i16* %arrayidx7, align 2
;
; DISABLED_MASKED_STRIDED: LV: Found an estimated cost of 3000000 for VF 2 For instruction:   %i2 = load i16, i16* %arrayidx2, align 2
; DISABLED_MASKED_STRIDED: LV: Found an estimated cost of 3000000 for VF 2 For instruction:   %i4 = load i16, i16* %arrayidx7, align 2
;
; DISABLED_MASKED_STRIDED: LV: Found an estimated cost of 3000000 for VF 4 For instruction:   %i2 = load i16, i16* %arrayidx2, align 2
; DISABLED_MASKED_STRIDED: LV: Found an estimated cost of 3000000 for VF 4 For instruction:   %i4 = load i16, i16* %arrayidx7, align 2
;
; DISABLED_MASKED_STRIDED: LV: Found an estimated cost of 3000000 for VF 8 For instruction:   %i2 = load i16, i16* %arrayidx2, align 2
; DISABLED_MASKED_STRIDED: LV: Found an estimated cost of 3000000 for VF 8 For instruction:   %i4 = load i16, i16* %arrayidx7, align 2
;
; DISABLED_MASKED_STRIDED: LV: Found an estimated cost of 3000000 for VF 16 For instruction:   %i2 = load i16, i16* %arrayidx2, align 2
; DISABLED_MASKED_STRIDED: LV: Found an estimated cost of 3000000 for VF 16 For instruction:   %i4 = load i16, i16* %arrayidx7, align 2

; ENABLED_MASKED_STRIDED: LV: Checking a loop in "test2"
;
; ENABLED_MASKED_STRIDED: LV: Found an estimated cost of 1 for VF 1 For instruction:   %i2 = load i16, i16* %arrayidx2, align 2
; ENABLED_MASKED_STRIDED: LV: Found an estimated cost of 1 for VF 1 For instruction:   %i4 = load i16, i16* %arrayidx7, align 2
;
; ENABLED_MASKED_STRIDED: LV: Found an estimated cost of 8 for VF 2 For instruction:   %i2 = load i16, i16* %arrayidx2, align 2
; ENABLED_MASKED_STRIDED: LV: Found an estimated cost of 0 for VF 2 For instruction:   %i4 = load i16, i16* %arrayidx7, align 2
;
; ENABLED_MASKED_STRIDED: LV: Found an estimated cost of 11 for VF 4 For instruction:   %i2 = load i16, i16* %arrayidx2, align 2
; ENABLED_MASKED_STRIDED: LV: Found an estimated cost of 0 for VF 4 For instruction:   %i4 = load i16, i16* %arrayidx7, align 2
;
; ENABLED_MASKED_STRIDED: LV: Found an estimated cost of 11 for VF 8 For instruction:   %i2 = load i16, i16* %arrayidx2, align 2
; ENABLED_MASKED_STRIDED: LV: Found an estimated cost of 0 for VF 8 For instruction:   %i4 = load i16, i16* %arrayidx7, align 2
;
; ENABLED_MASKED_STRIDED: LV: Found an estimated cost of 17 for VF 16 For instruction:   %i2 = load i16, i16* %arrayidx2, align 2
; ENABLED_MASKED_STRIDED: LV: Found an estimated cost of 0 for VF 16 For instruction:   %i4 = load i16, i16* %arrayidx7, align 2

define void @test2(i16* noalias nocapture %points, i32 %numPoints, i16* noalias nocapture readonly %x, i16* noalias nocapture readonly %y) {
entry:
  %cmp15 = icmp sgt i32 %numPoints, 0
  br i1 %cmp15, label %for.body.preheader, label %for.end

for.body.preheader:
  %wide.trip.count = zext i32 %numPoints to i64
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %i1 = shl nuw nsw i64 %indvars.iv, 2
  %arrayidx2 = getelementptr inbounds i16, i16* %points, i64 %i1
  %i2 = load i16, i16* %arrayidx2, align 2
  %i3 = or i64 %i1, 1
  %arrayidx7 = getelementptr inbounds i16, i16* %points, i64 %i3
  %i4 = load i16, i16* %arrayidx7, align 2
  %arrayidx = getelementptr inbounds i16, i16* %x, i64 %indvars.iv
  store i16 %i2, i16* %arrayidx, align 2
  %arrayidx4 = getelementptr inbounds i16, i16* %y, i64 %indvars.iv
  store i16 %i4, i16* %arrayidx4, align 2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.end.loopexit, label %for.body

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}

; (3) Testing a scenario of a conditional store. The gaps mask of the store is also
; And-ed with the condition mask (x[i] > 0).
; If using masked memops to vectorize interleaved-group with gaps is
; not allowed, the store is scalarized and predicated.
; Here the Interleave-group is with factor 3, storing only 1 member out of the 3.
; The input IR was generated from this source:
;     for(i=0;i<1024;i++){
;       if (x[i] > 0)
;         x[i] = points[i*3];
;     }

; DISABLED_MASKED_STRIDED: LV: Checking a loop in "test"
;
; DISABLED_MASKED_STRIDED: LV: Found an estimated cost of 1 for VF 1 For instruction:   %i4 = load i16, i16* %arrayidx6, align 2
; DISABLED_MASKED_STRIDED: LV: Found an estimated cost of 3000000 for VF 2 For instruction:   %i4 = load i16, i16* %arrayidx6, align 2
; DISABLED_MASKED_STRIDED: LV: Found an estimated cost of 3000000 for VF 4 For instruction:   %i4 = load i16, i16* %arrayidx6, align 2
; DISABLED_MASKED_STRIDED: LV: Found an estimated cost of 3000000 for VF 8 For instruction:   %i4 = load i16, i16* %arrayidx6, align 2
; DISABLED_MASKED_STRIDED: LV: Found an estimated cost of 3000000 for VF 16 For instruction:   %i4 = load i16, i16* %arrayidx6, align 2

; ENABLED_MASKED_STRIDED: LV: Checking a loop in "test"
;
; ENABLED_MASKED_STRIDED: LV: Found an estimated cost of 1 for VF 1 For instruction:   %i4 = load i16, i16* %arrayidx6, align 2
; ENABLED_MASKED_STRIDED: LV: Found an estimated cost of 7 for VF 2 For instruction:   %i4 = load i16, i16* %arrayidx6, align 2
; ENABLED_MASKED_STRIDED: LV: Found an estimated cost of 9 for VF 4 For instruction:   %i4 = load i16, i16* %arrayidx6, align 2
; ENABLED_MASKED_STRIDED: LV: Found an estimated cost of 9 for VF 8 For instruction:   %i4 = load i16, i16* %arrayidx6, align 2
; ENABLED_MASKED_STRIDED: LV: Found an estimated cost of 14 for VF 16 For instruction:   %i4 = load i16, i16* %arrayidx6, align 2

define void @test(i16* noalias nocapture %points, i16* noalias nocapture readonly %x, i16* noalias nocapture readnone %y) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.inc ]
  %arrayidx = getelementptr inbounds i16, i16* %x, i64 %indvars.iv
  %i2 = load i16, i16* %arrayidx, align 2
  %cmp1 = icmp sgt i16 %i2, 0
  br i1 %cmp1, label %if.then, label %for.inc

if.then:
  %i1 = mul nuw nsw i64 %indvars.iv, 3
  %arrayidx6 = getelementptr inbounds i16, i16* %points, i64 %i1
  %i4 = load i16, i16* %arrayidx6, align 2
  store i16 %i4, i16* %arrayidx, align 2
  br label %for.inc

for.inc:
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:
  ret void
}
