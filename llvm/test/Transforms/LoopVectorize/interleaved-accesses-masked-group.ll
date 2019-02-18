; REQUIRES: asserts
; RUN: opt -S -loop-vectorize -force-vector-width=8 -force-vector-interleave=1 -enable-interleaved-mem-accesses -debug-only=loop-vectorize,vectorutils -disable-output < %s 2>&1 | FileCheck %s -check-prefix=STRIDED_UNMASKED
; RUN: opt -S -loop-vectorize -force-vector-width=8 -force-vector-interleave=1 -enable-interleaved-mem-accesses -enable-masked-interleaved-mem-accesses -debug-only=loop-vectorize,vectorutils -disable-output < %s 2>&1 | FileCheck %s -check-prefix=STRIDED_MASKED

target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"

; We test here that the loop-vectorizer forms an interleave-groups from 
; predicated memory accesses only if they are both in the same (predicated)
; block (first scenario below).
; If the accesses are not in the same predicated block, an interleave-group
; is not formed (scenarios 2,3 below).

; Scenario 1: Check the case where it is legal to create masked interleave-
; groups. Altogether two groups are created (one for loads and one for stores)
; when masked-interleaved-acceses are enabled. When masked-interleaved-acceses
; are disabled we do not create any interleave-group.
;
; void masked_strided1(const unsigned char* restrict p,
;                     unsigned char* restrict q,
;                     unsigned char guard) {
; for(ix=0; ix < 1024; ++ix) {
;     if (ix > guard) {
;         char left = p[2*ix];
;         char right = p[2*ix + 1];
;         char max = max(left, right);
;         q[2*ix] = max;
;         q[2*ix+1] = 0 - max;
;     }
; }
;}


; STRIDED_UNMASKED: LV: Checking a loop in "masked_strided1" 
; STRIDED_UNMASKED: LV: Analyzing interleaved accesses...
; STRIDED_UNMASKED-NOT: LV: Creating an interleave group 

; STRIDED_MASKED: LV: Checking a loop in "masked_strided1" 
; STRIDED_MASKED: LV: Analyzing interleaved accesses...
; STRIDED_MASKED-NEXT: LV: Creating an interleave group with:  store i8 %{{.*}}, i8* %{{.*}}, align 1
; STRIDED_MASKED-NEXT: LV: Inserted:  store i8  %{{.*}}, i8* %{{.*}}, align 1
; STRIDED_MASKED-NEXT:     into the interleave group with  store i8 %{{.*}}, i8* %{{.*}}, align 1
; STRIDED_MASKED-NEXT: LV: Creating an interleave group with:   %{{.*}} = load i8, i8* %{{.*}}, align 1
; STRIDED_MASKED-NEXT: LV: Inserted:  %{{.*}} = load i8, i8* %{{.*}}, align 1
; STRIDED_MASKED-NEXT:     into the interleave group with   %{{.*}} = load i8, i8* %{{.*}}, align 1

; Scenario 2: Check the case where it is illegal to create a masked interleave-
; group because the first access is predicated, and the second isn't.
; We therefore create a separate interleave-group with gaps for each of the
; stores (if masked-interleaved-accesses are enabled) and these are later
; invalidated because interleave-groups of stores with gaps are not supported. 
; If masked-interleaved-accesses is not enabled we create only one interleave
; group of stores (for the non-predicated store) and it is later invalidated
; due to gaps.
;
; void masked_strided2(const unsigned char* restrict p,
;                     unsigned char* restrict q,
;                     unsigned char guard1,
;                     unsigned char guard2) {
; for(ix=0; ix < 1024; ++ix) {
;     if (ix > guard1) {
;         q[2*ix] = 1;
;     }
;     q[2*ix+1] = 2;
; }
;}

; STRIDED_UNMASKED: LV: Checking a loop in "masked_strided2" 
; STRIDED_UNMASKED: LV: Analyzing interleaved accesses...
; STRIDED_UNMASKED-NEXT: LV: Creating an interleave group with:  store i8 1, i8* %{{.*}}, align 1
; STRIDED_UNMASKED-NEXT: LV: Invalidate candidate interleaved store group due to gaps.
; STRIDED_UNMASKED-NOT: LV: Creating an interleave group 

; STRIDED_MASKED: LV: Checking a loop in "masked_strided2" 
; STRIDED_MASKED: LV: Analyzing interleaved accesses...
; STRIDED_MASKED-NEXT: LV: Creating an interleave group with:  store i8 2, i8* %{{.*}}, align 1
; STRIDED_MASKED-NEXT: LV: Creating an interleave group with:  store i8 1, i8* %{{.*}}, align 1
; STRIDED_MASKED-NEXT: LV: Invalidate candidate interleaved store group due to gaps.
; STRIDED_MASKED-NEXT: LV: Invalidate candidate interleaved store group due to gaps.


; Scenario 3: Check the case where it is illegal to create a masked interleave-
; group because the two accesses are in separate predicated blocks.
; We therefore create a separate interleave-group with gaps for each of the accesses,
; (which are later invalidated because interleave-groups of stores with gaps are 
; not supported).
; If masked-interleaved-accesses is not enabled we don't create any interleave
; group because all accesses are predicated.
;
; void masked_strided3(const unsigned char* restrict p,
;                     unsigned char* restrict q,
;                     unsigned char guard1,
;                     unsigned char guard2) {
; for(ix=0; ix < 1024; ++ix) {
;     if (ix > guard1) {
;         q[2*ix] = 1;
;     }
;     if (ix > guard2) {
;         q[2*ix+1] = 2;
;     }
; }
;}


; STRIDED_UNMASKED: LV: Checking a loop in "masked_strided3" 
; STRIDED_UNMASKED: LV: Analyzing interleaved accesses...
; STRIDED_UNMASKED-NOT: LV: Creating an interleave group 

; STRIDED_MASKED: LV: Checking a loop in "masked_strided3" 
; STRIDED_MASKED: LV: Analyzing interleaved accesses...
; STRIDED_MASKED-NEXT: LV: Creating an interleave group with:  store i8 2, i8* %{{.*}}, align 1
; STRIDED_MASKED-NEXT: LV: Creating an interleave group with:  store i8 1, i8* %{{.*}}, align 1
; STRIDED_MASKED-NEXT: LV: Invalidate candidate interleaved store group due to gaps.
; STRIDED_MASKED-NEXT: LV: Invalidate candidate interleaved store group due to gaps.


; ModuleID = 'test.c'
source_filename = "test.c"
target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

define dso_local void @masked_strided1(i8* noalias nocapture readonly %p, i8* noalias nocapture %q, i8 zeroext %guard) local_unnamed_addr #0 {
entry:
  %conv = zext i8 %guard to i32
  br label %for.body

for.body:
  %ix.024 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp1 = icmp ugt i32 %ix.024, %conv
  br i1 %cmp1, label %if.then, label %for.inc

if.then:
  %mul = shl nuw nsw i32 %ix.024, 1
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 %mul
  %0 = load i8, i8* %arrayidx, align 1
  %add = or i32 %mul, 1
  %arrayidx4 = getelementptr inbounds i8, i8* %p, i32 %add
  %1 = load i8, i8* %arrayidx4, align 1
  %cmp.i = icmp slt i8 %0, %1
  %spec.select.i = select i1 %cmp.i, i8 %1, i8 %0
  %arrayidx6 = getelementptr inbounds i8, i8* %q, i32 %mul
  store i8 %spec.select.i, i8* %arrayidx6, align 1
  %sub = sub i8 0, %spec.select.i
  %arrayidx11 = getelementptr inbounds i8, i8* %q, i32 %add
  store i8 %sub, i8* %arrayidx11, align 1
  br label %for.inc

for.inc:
  %inc = add nuw nsw i32 %ix.024, 1
  %exitcond = icmp eq i32 %inc, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}


define dso_local void @masked_strided2(i8* noalias nocapture readnone %p, i8* noalias nocapture %q, i8 zeroext %guard) local_unnamed_addr #0 {
entry:
  %conv = zext i8 %guard to i32
  br label %for.body

for.body:
  %ix.012 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %mul = shl nuw nsw i32 %ix.012, 1
  %arrayidx = getelementptr inbounds i8, i8* %q, i32 %mul
  store i8 1, i8* %arrayidx, align 1
  %cmp1 = icmp ugt i32 %ix.012, %conv
  br i1 %cmp1, label %if.then, label %for.inc

if.then:
  %add = or i32 %mul, 1
  %arrayidx3 = getelementptr inbounds i8, i8* %q, i32 %add
  store i8 2, i8* %arrayidx3, align 1
  br label %for.inc

for.inc:
  %inc = add nuw nsw i32 %ix.012, 1
  %exitcond = icmp eq i32 %inc, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}


define dso_local void @masked_strided3(i8* noalias nocapture readnone %p, i8* noalias nocapture %q, i8 zeroext %guard1, i8 zeroext %guard2) local_unnamed_addr #0 {
entry:
  %conv = zext i8 %guard1 to i32
  %conv3 = zext i8 %guard2 to i32
  br label %for.body

for.body:
  %ix.018 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %mul = shl nuw nsw i32 %ix.018, 1
  %cmp1 = icmp ugt i32 %ix.018, %conv
  br i1 %cmp1, label %if.then, label %if.end

if.then:
  %arrayidx = getelementptr inbounds i8, i8* %q, i32 %mul
  store i8 1, i8* %arrayidx, align 1
  br label %if.end

if.end:
  %cmp4 = icmp ugt i32 %ix.018, %conv3
  br i1 %cmp4, label %if.then6, label %for.inc

if.then6:
  %add = or i32 %mul, 1
  %arrayidx7 = getelementptr inbounds i8, i8* %q, i32 %add
  store i8 2, i8* %arrayidx7, align 1
  br label %for.inc

for.inc:
  %inc = add nuw nsw i32 %ix.018, 1
  %exitcond = icmp eq i32 %inc, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

attributes #0 = {  "target-features"="+fxsr,+mmx,+sse,+sse2,+x87"  }
