; REQUIRES: asserts
; RUN: opt < %s -passes=loop-vectorize,dce,instcombine -mcpu=core-axv2 -force-vector-interleave=1 -debug-only=loop-vectorize -S < %s 2>&1  | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Make sure we ignore the costs of the redundant reduction casts
; char reduction_i8(char *a, char *b, int n) {
;   char sum = 0;
;   for (int i = 0; i < n; ++i)
;     sum += (a[i] + b[i]);
;   return sum;
; }
;

; CHECK-LABEL: reduction_i8
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 1 For instruction:   %{{.*}} = phi
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 1 For instruction:   %{{.*}} = phi
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 1 For instruction:   %{{.*}} = getelementptr
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 1 For instruction:   %{{.*}} = load
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 1 For instruction:   %{{.*}} = zext i8 %{{.*}} to i32
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 1 For instruction:   %{{.*}} = getelementptr
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 1 For instruction:   %{{.*}} = load
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 1 For instruction:   %{{.*}} = zext i8 %{{.*}} to i32
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 1 For instruction:   %{{.*}} = and i32 %{{.*}}, 255
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 1 For instruction:   %{{.*}} = add
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 1 For instruction:   %{{.*}} = add
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 1 For instruction:   %{{.*}} = add
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 1 For instruction:   %{{.*}} = trunc
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 1 For instruction:   %{{.*}} = icmp
; CHECK: LV: Found an estimated cost of {{[0-9]+}} for VF 1 For instruction:   br
; CHECK: Cost of 1 for VF 2: induction instruction   %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
; CHECK: Cost of 1 for VF 2: induction instruction   %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
; CHECK: Cost of 0 for VF 2: EMIT vp<%3> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; CHECK: Cost of 1 for VF 2: WIDEN-REDUCTION-PHI ir<%sum.013> = phi ir<0>, vp<%8>
; CHECK: Cost of 0 for VF 2: vp<%4> = SCALAR-STEPS vp<%3>, ir<1>
; CHECK: Cost of 0 for VF 2: CLONE ir<%arrayidx> = getelementptr inbounds ir<%a>, vp<%4>
; CHECK: Cost of 0 for VF 2: vp<%5> = vector-pointer ir<%arrayidx>
; CHECK: Cost of 1 for VF 2: WIDEN ir<%0> = load vp<%5>
; CHECK: Cost of 0 for VF 2: WIDEN-CAST ir<%conv> = zext ir<%0> to i32
; CHECK: Cost of 0 for VF 2: CLONE ir<%arrayidx2> = getelementptr inbounds ir<%b>, vp<%4>
; CHECK: Cost of 0 for VF 2: vp<%6> = vector-pointer ir<%arrayidx2>
; CHECK: Cost of 1 for VF 2: WIDEN ir<%1> = load vp<%6>
; CHECK: Cost of 0 for VF 2: WIDEN-CAST ir<%conv3> = zext ir<%1> to i32
; CHECK: Cost of 0 for VF 2: WIDEN ir<%conv4> = and ir<%sum.013>, ir<255>
; CHECK: Cost of 1 for VF 2: WIDEN ir<%add> = add ir<%conv>, ir<%conv4>
; CHECK: Cost of 1 for VF 2: WIDEN ir<%add5> = add ir<%add>, ir<%conv3>
; CHECK: Cost of 0 for VF 2: WIDEN-CAST vp<%7> = trunc ir<%add5> to i8
; CHECK: Cost of 0 for VF 2: WIDEN-CAST vp<%8> = zext vp<%7> to i32
; CHECK: Cost of 0 for VF 2: EMIT vp<%index.next> = add nuw vp<%3>, vp<%0>
; CHECK: Cost of 1 for VF 2: EMIT branch-on-count vp<%index.next>, vp<%1>
;
define i8 @reduction_i8(ptr nocapture readonly %a, ptr nocapture readonly %b, i32 %n) {
entry:
  %cmp.12 = icmp sgt i32 %n, 0
  br i1 %cmp.12, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  br label %for.body

for.cond.for.cond.cleanup_crit_edge:
  %add5.lcssa = phi i32 [ %add5, %for.body ]
  %conv6 = trunc i32 %add5.lcssa to i8
  br label %for.cond.cleanup

for.cond.cleanup:
  %sum.0.lcssa = phi i8 [ %conv6, %for.cond.for.cond.cleanup_crit_edge ], [ 0, %entry ]
  ret i8 %sum.0.lcssa

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %sum.013 = phi i32 [ %add5, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i8, ptr %a, i64 %indvars.iv
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %arrayidx2 = getelementptr inbounds i8, ptr %b, i64 %indvars.iv
  %1 = load i8, ptr %arrayidx2, align 1
  %conv3 = zext i8 %1 to i32
  %conv4 = and i32 %sum.013, 255
  %add = add nuw nsw i32 %conv, %conv4
  %add5 = add nuw nsw i32 %add, %conv3
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.cond.for.cond.cleanup_crit_edge, label %for.body
}
