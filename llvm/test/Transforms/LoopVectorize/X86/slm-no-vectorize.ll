; RUN: opt < %s -passes=loop-vectorize -mtriple=x86_64-unknown-linux -S -mcpu=slm -debug 2>&1 | FileCheck -check-prefix=MSG %s
; REQUIRES: asserts
; This test should not be vectorized in X86\SLM arch
; Vectorizing the 64bit multiply in this case is wrong since
; it can be done with a lower bit mode (notice that the sources is 16bit)
; Also addq\subq (quad word) has a high cost on SLM arch.
; this test has a bad performance (regression of -70%) if vectorized on SLM arch
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @no_vec(i32 %LastIndex, i16* nocapture readonly %InputData, i16 signext %lag, i16 signext %Scale) {
entry:
; MSG: LV: Selecting VF: 1.
  %cmp17 = icmp sgt i32 %LastIndex, 0
  br i1 %cmp17, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry
  %conv5 = sext i16 %Scale to i64
  %sh_prom = and i64 %conv5, 4294967295
  %0 = sext i16 %lag to i64
  %wide.trip.count = zext i32 %LastIndex to i64
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  %conv8 = trunc i64 %add7 to i32
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  %Accumulator.0.lcssa = phi i32 [ 0, %entry ], [ %conv8, %for.cond.cleanup.loopexit ]
  ret i32 %Accumulator.0.lcssa

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %Accumulator.018 = phi i64 [ 0, %for.body.lr.ph ], [ %add7, %for.body ]
  %arrayidx = getelementptr inbounds i16, i16* %InputData, i64 %indvars.iv
  %1 = load i16, i16* %arrayidx, align 2
  %conv = sext i16 %1 to i64
  %2 = add nsw i64 %indvars.iv, %0
  %arrayidx3 = getelementptr inbounds i16, i16* %InputData, i64 %2
  %3 = load i16, i16* %arrayidx3, align 2
  %conv4 = sext i16 %3 to i64
  %mul = mul nsw i64 %conv4, %conv
  %shr = ashr i64 %mul, %sh_prom
  %add7 = add i64 %shr, %Accumulator.018
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body
}

