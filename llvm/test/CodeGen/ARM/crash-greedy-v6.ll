; RUN: llc -frame-pointer=all -relocation-model=pic < %s
; RUN: llc -frame-pointer=all -relocation-model=pic -O0 -pre-RA-sched=source < %s | FileCheck %s --check-prefix=SOURCE-SCHED
target triple = "armv6-apple-ios"

; Reduced from 177.mesa. This test causes a live range split before an LDR_POST instruction.
; That requires leaveIntvBefore to be very accurate about the redefined value number.
define internal void @sample_nearest_3d(i8* nocapture %tObj, i32 %n, float* nocapture %s, float* nocapture %t, float* nocapture %u, float* nocapture %lambda, i8* nocapture %red, i8* nocapture %green, i8* nocapture %blue, i8* nocapture %alpha) nounwind ssp {
entry:
  br i1 undef, label %for.end, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph
; SOURCE-SCHED: ldr
; SOURCE-SCHED: ldr
; SOURCE-SCHED: add
; SOURCE-SCHED: ldr
; SOURCE-SCHED: add
; SOURCE-SCHED: ldr
; SOURCE-SCHED: add
; SOURCE-SCHED: ldr
; SOURCE-SCHED: add
; SOURCE-SCHED: str
; SOURCE-SCHED: str
; SOURCE-SCHED: str
; SOURCE-SCHED: str
; SOURCE-SCHED: ldr
; SOURCE-SCHED: bl
; SOURCE-SCHED: add
; SOURCE-SCHED: ldr
; SOURCE-SCHED: cmp
; SOURCE-SCHED: bne
  %i.031 = phi i32 [ 0, %for.body.lr.ph ], [ %0, %for.body ]
  %arrayidx11 = getelementptr float, float* %t, i32 %i.031
  %arrayidx15 = getelementptr float, float* %u, i32 %i.031
  %arrayidx19 = getelementptr i8, i8* %red, i32 %i.031
  %arrayidx22 = getelementptr i8, i8* %green, i32 %i.031
  %arrayidx25 = getelementptr i8, i8* %blue, i32 %i.031
  %arrayidx28 = getelementptr i8, i8* %alpha, i32 %i.031
  %tmp12 = load float, float* %arrayidx11, align 4
  tail call fastcc void @sample_3d_nearest(i8* %tObj, i8* undef, float undef, float %tmp12, float undef, i8* %arrayidx19, i8* %arrayidx22, i8* %arrayidx25, i8* %arrayidx28)
  %0 = add i32 %i.031, 1
  %exitcond = icmp eq i32 %0, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare fastcc void @sample_3d_nearest(i8* nocapture, i8* nocapture, float, float, float, i8* nocapture, i8* nocapture, i8* nocapture, i8* nocapture) nounwind ssp

