; RUN: llc -frame-pointer=all -relocation-model=pic < %s
; RUN: llc -frame-pointer=all -relocation-model=pic -O0 -pre-RA-sched=source < %s | FileCheck %s --check-prefix=SOURCE-SCHED
target triple = "armv6-apple-ios"

; Reduced from 177.mesa. This test causes a live range split before an LDR_POST instruction.
; That requires leaveIntvBefore to be very accurate about the redefined value number.
define internal void @sample_nearest_3d(ptr nocapture %tObj, i32 %n, ptr nocapture %s, ptr nocapture %t, ptr nocapture %u, ptr nocapture %lambda, ptr nocapture %red, ptr nocapture %green, ptr nocapture %blue, ptr nocapture %alpha) nounwind ssp {
entry:
  br i1 undef, label %for.end, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph
; SOURCE-SCHED: ldr
; SOURCE-SCHED: ldr
; SOURCE-SCHED: ldr
; SOURCE-SCHED: ldr
; SOURCE-SCHED: ldr
; SOURCE-SCHED: add
; SOURCE-SCHED: add
; SOURCE-SCHED: add
; SOURCE-SCHED: add
; SOURCE-SCHED: ldr
; SOURCE-SCHED: str
; SOURCE-SCHED: str
; SOURCE-SCHED: str
; SOURCE-SCHED: str
; SOURCE-SCHED: bl
; SOURCE-SCHED: ldr
; SOURCE-SCHED: add
; SOURCE-SCHED: cmp
; SOURCE-SCHED: bne
  %i.031 = phi i32 [ 0, %for.body.lr.ph ], [ %0, %for.body ]
  %arrayidx11 = getelementptr float, ptr %t, i32 %i.031
  %arrayidx15 = getelementptr float, ptr %u, i32 %i.031
  %arrayidx19 = getelementptr i8, ptr %red, i32 %i.031
  %arrayidx22 = getelementptr i8, ptr %green, i32 %i.031
  %arrayidx25 = getelementptr i8, ptr %blue, i32 %i.031
  %arrayidx28 = getelementptr i8, ptr %alpha, i32 %i.031
  %tmp12 = load float, ptr %arrayidx11, align 4
  tail call fastcc void @sample_3d_nearest(ptr %tObj, ptr undef, float undef, float %tmp12, float undef, ptr %arrayidx19, ptr %arrayidx22, ptr %arrayidx25, ptr %arrayidx28)
  %0 = add i32 %i.031, 1
  %exitcond = icmp eq i32 %0, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare fastcc void @sample_3d_nearest(ptr nocapture, ptr nocapture, float, float, float, ptr nocapture, ptr nocapture, ptr nocapture, ptr nocapture) nounwind ssp

