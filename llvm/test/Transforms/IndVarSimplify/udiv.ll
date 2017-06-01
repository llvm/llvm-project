; RUN: opt -indvars -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

@main.flags = internal global [8193 x i8] zeroinitializer, align 1 ; <[8193 x i8]*> [#uses=5]
@.str = private constant [11 x i8] c"Count: %d\0A\00" ; <[11 x i8]*> [#uses=1]

; Indvars shouldn't emit a udiv here, because there's no udiv in the
; original code. This comes from SingleSource/Benchmarks/Shootout/sieve.c.

; CHECK-LABEL: @main(
; CHECK-NOT: div

define i32 @main(i32 %argc, i8** nocapture %argv) nounwind {
entry:
  %cmp = icmp eq i32 %argc, 2                     ; <i1> [#uses=1]
  br i1 %cmp, label %cond.true, label %while.cond.preheader

cond.true:                                        ; preds = %entry
  %arrayidx = getelementptr inbounds i8*, i8** %argv, i64 1 ; <i8**> [#uses=1]
  %tmp2 = load i8*, i8** %arrayidx                     ; <i8*> [#uses=1]
  %call = tail call i32 @atoi(i8* %tmp2) nounwind readonly ; <i32> [#uses=1]
  br label %while.cond.preheader

while.cond.preheader:                             ; preds = %entry, %cond.true
  %NUM.0.ph = phi i32 [ %call, %cond.true ], [ 170000, %entry ] ; <i32> [#uses=2]
  %tobool18 = icmp eq i32 %NUM.0.ph, 0            ; <i1> [#uses=1]
  br i1 %tobool18, label %while.end, label %bb.nph30

while.cond.loopexit:                              ; preds = %for.cond12.while.cond.loopexit_crit_edge, %for.cond12.loopexit
  %count.2.lcssa = phi i32 [ %count.1.lcssa, %for.cond12.while.cond.loopexit_crit_edge ], [ 0, %for.cond12.loopexit ] ; <i32> [#uses=1]
  br label %while.cond

while.cond:                                       ; preds = %while.cond.loopexit
  %tobool = icmp eq i32 %dec19, 0                 ; <i1> [#uses=1]
  br i1 %tobool, label %while.cond.while.end_crit_edge, label %for.cond.preheader

while.cond.while.end_crit_edge:                   ; preds = %while.cond
  %count.2.lcssa.lcssa = phi i32 [ %count.2.lcssa, %while.cond ] ; <i32> [#uses=1]
  br label %while.end

bb.nph30:                                         ; preds = %while.cond.preheader
  br label %for.cond.preheader

for.cond.preheader:                               ; preds = %bb.nph30, %while.cond
  %dec19.in = phi i32 [ %NUM.0.ph, %bb.nph30 ], [ %dec19, %while.cond ] ; <i32> [#uses=1]
  %dec19 = add i32 %dec19.in, -1                  ; <i32> [#uses=2]
  br i1 true, label %bb.nph, label %for.cond12.loopexit

for.cond:                                         ; preds = %for.body
  %cmp8 = icmp slt i64 %inc, 8193                 ; <i1> [#uses=1]
  br i1 %cmp8, label %for.body, label %for.cond.for.cond12.loopexit_crit_edge

for.cond.for.cond12.loopexit_crit_edge:           ; preds = %for.cond
  br label %for.cond12.loopexit

bb.nph:                                           ; preds = %for.cond.preheader
  br label %for.body

for.body:                                         ; preds = %bb.nph, %for.cond
  %i.02 = phi i64 [ 2, %bb.nph ], [ %inc, %for.cond ] ; <i64> [#uses=2]
  %arrayidx10 = getelementptr inbounds [8193 x i8], [8193 x i8]* @main.flags, i64 0, i64 %i.02 ; <i8*> [#uses=1]
  store i8 1, i8* %arrayidx10
  %inc = add nsw i64 %i.02, 1                     ; <i64> [#uses=2]
  br label %for.cond

for.cond12.loopexit:                              ; preds = %for.cond.for.cond12.loopexit_crit_edge, %for.cond.preheader
  br i1 true, label %bb.nph16, label %while.cond.loopexit

for.cond12:                                       ; preds = %for.inc35
  %cmp14 = icmp slt i64 %inc37, 8193              ; <i1> [#uses=1]
  br i1 %cmp14, label %for.body15, label %for.cond12.while.cond.loopexit_crit_edge

for.cond12.while.cond.loopexit_crit_edge:         ; preds = %for.cond12
  %count.1.lcssa = phi i32 [ %count.1, %for.cond12 ] ; <i32> [#uses=1]
  br label %while.cond.loopexit

bb.nph16:                                         ; preds = %for.cond12.loopexit
  br label %for.body15

for.body15:                                       ; preds = %bb.nph16, %for.cond12
  %count.212 = phi i32 [ 0, %bb.nph16 ], [ %count.1, %for.cond12 ] ; <i32> [#uses=2]
  %i.17 = phi i64 [ 2, %bb.nph16 ], [ %inc37, %for.cond12 ] ; <i64> [#uses=4]
  %arrayidx17 = getelementptr inbounds [8193 x i8], [8193 x i8]* @main.flags, i64 0, i64 %i.17 ; <i8*> [#uses=1]
  %tmp18 = load i8, i8* %arrayidx17                   ; <i8> [#uses=1]
  %tobool19 = icmp eq i8 %tmp18, 0                ; <i1> [#uses=1]
  br i1 %tobool19, label %for.inc35, label %if.then

if.then:                                          ; preds = %for.body15
  %add = shl i64 %i.17, 1                         ; <i64> [#uses=2]
  %cmp243 = icmp slt i64 %add, 8193               ; <i1> [#uses=1]
  br i1 %cmp243, label %bb.nph5, label %for.end32

for.cond22:                                       ; preds = %for.body25
  %cmp24 = icmp slt i64 %add31, 8193              ; <i1> [#uses=1]
  br i1 %cmp24, label %for.body25, label %for.cond22.for.end32_crit_edge

for.cond22.for.end32_crit_edge:                   ; preds = %for.cond22
  br label %for.end32

bb.nph5:                                          ; preds = %if.then
  br label %for.body25

for.body25:                                       ; preds = %bb.nph5, %for.cond22
  %k.04 = phi i64 [ %add, %bb.nph5 ], [ %add31, %for.cond22 ] ; <i64> [#uses=2]
  %arrayidx27 = getelementptr inbounds [8193 x i8], [8193 x i8]* @main.flags, i64 0, i64 %k.04 ; <i8*> [#uses=1]
  store i8 0, i8* %arrayidx27
  %add31 = add nsw i64 %k.04, %i.17               ; <i64> [#uses=2]
  br label %for.cond22

for.end32:                                        ; preds = %for.cond22.for.end32_crit_edge, %if.then
  %inc34 = add nsw i32 %count.212, 1              ; <i32> [#uses=1]
  br label %for.inc35

for.inc35:                                        ; preds = %for.body15, %for.end32
  %count.1 = phi i32 [ %inc34, %for.end32 ], [ %count.212, %for.body15 ] ; <i32> [#uses=2]
  %inc37 = add nsw i64 %i.17, 1                   ; <i64> [#uses=2]
  br label %for.cond12

while.end:                                        ; preds = %while.cond.while.end_crit_edge, %while.cond.preheader
  %count.0.lcssa = phi i32 [ %count.2.lcssa.lcssa, %while.cond.while.end_crit_edge ], [ 0, %while.cond.preheader ] ; <i32> [#uses=1]
  %call40 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i64 0, i64 0), i32 %count.0.lcssa) nounwind ; <i32> [#uses=0]
  ret i32 0
}

declare i32 @atoi(i8* nocapture) nounwind readonly

declare i32 @printf(i8* nocapture, ...) nounwind

; IndVars doesn't emit a udiv in for.body.preheader since SCEVExpander::expand will
; find out there's already a udiv in the original code.

; CHECK-LABEL: @foo(
; CHECK: for.body.preheader:
; CHECK-NOT: udiv

define void @foo(double* %p, i64 %n) nounwind {
entry:
  %div0 = udiv i64 %n, 7                          ; <i64> [#uses=1]
  %div1 = add i64 %div0, 1
  %cmp2 = icmp ult i64 0, %div1                   ; <i1> [#uses=1]
  br i1 %cmp2, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.03 = phi i64 [ %inc, %for.body ], [ 0, %for.body.preheader ] ; <i64> [#uses=2]
  %arrayidx = getelementptr inbounds double, double* %p, i64 %i.03 ; <double*> [#uses=1]
  store double 0.000000e+00, double* %arrayidx
  %inc = add i64 %i.03, 1                         ; <i64> [#uses=2]
  %divx = udiv i64 %n, 7                           ; <i64> [#uses=1]
  %div = add i64 %divx, 1
  %cmp = icmp ult i64 %inc, %div                  ; <i1> [#uses=1]
  br i1 %cmp, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}
