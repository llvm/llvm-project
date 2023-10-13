; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -debug -disable-output %s 2>&1 < %s | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

; CHECK-LABEL: 'kernel_reference'
; CHECK:      VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF>=1' {
; CHECK-NEXT: Live-in vp<%0> = vector-trip-count
; CHECK-NEXT: vp<%1> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ph:
; CHECK-NEXT:   EMIT vp<%1> = EXPAND SCEV (zext i32 %N to i64)
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT vp<%2> = CANONICAL-INDUCTION
; CHECK-NEXT:     COMPACT-PHI ir<%n.013> = phi ir<0>, ir<%n.1>
; CHECK-NEXT:     vp<%4>    = SCALAR-STEPS vp<%2>, ir<1>
; CHECK-NEXT:     CLONE ir<%arrayidx> = getelementptr inbounds ir<%comp>, vp<%4>
; CHECK-NEXT:     WIDEN ir<%0> = load ir<%arrayidx>
; CHECK-NEXT:     WIDEN ir<%cmp1> = icmp slt ir<%0>, ir<%a>
; CHECK-NEXT:     CLONE ir<%arrayidx3> = getelementptr inbounds ir<%B>, vp<%4>
; CHECK-NEXT:     WIDEN ir<%1> = load ir<%arrayidx3>, ir<%cmp1>
; CHECK-NEXT:     COMPACT ir<%inc> = add ir<%n.013>, ir<1>
; CHECK-NEXT:     CLONE ir<%idxprom4> = sext ir<%n.013>
; CHECK-NEXT:     CLONE ir<%arrayidx5> = getelementptr inbounds ir<%Out_ref>, ir<%idxprom4>
; CHECK-NEXT:     COMPACT store ir<%1>, ir<%arrayidx5>
; CHECK-NEXT:     COMPACT ir<%n.1> = phi ir<%inc>, ir<%n.013>
; CHECK-NEXT:     EMIT vp<%15> = VF * UF + nuw vp<%2>
; CHECK-NEXT:     EMIT branch-on-count vp<%15>, vp<%0>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
;
; Function Attrs: argmemonly nofree norecurse nosync nounwind uwtable vscale_range(1,16)
define dso_local i32 @kernel_reference(i32 noundef %N, i32 noundef %a, ptr noalias nocapture noundef readonly %comp, ptr noalias nocapture noundef writeonly %Out_ref, ptr nocapture noundef readonly %B, ptr noalias nocapture noundef readnone %Out1) #0 {
entry:
  %cmp11 = icmp sgt i32 %N, 0
  br i1 %cmp11, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %N to i64
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.inc
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.inc ]
  %n.013 = phi i32 [ 0, %for.body.preheader ], [ %n.1, %for.inc ]
  %arrayidx = getelementptr inbounds i32, ptr %comp, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp1 = icmp slt i32 %0, %a
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %arrayidx3 = getelementptr inbounds i32, ptr %B, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx3, align 4
  %inc = add nsw i32 %n.013, 1
  %idxprom4 = sext i32 %n.013 to i64
  %arrayidx5 = getelementptr inbounds i32, ptr %Out_ref, i64 %idxprom4
  store i32 %1, ptr %arrayidx5, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %n.1 = phi i32 [ %inc, %if.then ], [ %n.013, %for.body ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc, %entry
  %n.0.lcssa = phi i32 [ 0, %entry ], [ %n.1, %for.inc ]
  ret i32 %n.0.lcssa
}

attributes #0 = { argmemonly nofree norecurse nosync nounwind uwtable vscale_range(1,16) "target-cpu"="generic" "target-features"="+neon,+sve,+v8.2a"}
