; RUN: opt < %s  -O1  -S -loop-versioning-licm -debug-only=loop-versioning-licm  2>&1 | FileCheck %s
; RUN: opt < %s  -S -passes='loop-versioning-licm' -debug-only=loop-versioning-licm  2>&1 | FileCheck %s
; REQUIRES: asserts
;
; Test to confirm loop is not a candidate for LoopVersioningLICM.
;
; CHECK: Loop: Loop at depth 2 containing: %for.body3<header><latch><exiting>
; CHECK-NEXT:    LAA: Runtime check not found !!
; CHECK-NEXT:    Loop instructions not suitable for LoopVersioningLICM

define i32 @foo(i32* nocapture %var1, i32 %itr) #0 {
entry:
  %cmp18 = icmp eq i32 %itr, 0
  br i1 %cmp18, label %for.end8, label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %entry, %for.inc6
  %j.020 = phi i32 [ %j.1.lcssa, %for.inc6 ], [ 0, %entry ]
  %i.019 = phi i32 [ %inc7, %for.inc6 ], [ 0, %entry ]
  %cmp216 = icmp ult i32 %j.020, %itr
  br i1 %cmp216, label %for.body3.lr.ph, label %for.inc6

for.body3.lr.ph:                                  ; preds = %for.cond1.preheader
  %0 = zext i32 %j.020 to i64
  br label %for.body3

for.body3:                                        ; preds = %for.body3, %for.body3.lr.ph
  %indvars.iv = phi i64 [ %0, %for.body3.lr.ph ], [ %indvars.iv.next, %for.body3 ]
  %arrayidx = getelementptr inbounds i32, i32* %var1, i64 %indvars.iv
  %1 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %1, %itr
  store i32 %add, i32* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %itr
  br i1 %exitcond, label %for.inc6, label %for.body3

for.inc6:                                         ; preds = %for.body3, %for.cond1.preheader
  %j.1.lcssa = phi i32 [ %j.020, %for.cond1.preheader ], [ %itr, %for.body3 ]
  %inc7 = add nuw i32 %i.019, 1
  %exitcond21 = icmp eq i32 %inc7, %itr
  br i1 %exitcond21, label %for.end8, label %for.cond1.preheader

for.end8:                                         ; preds = %for.inc6, %entry
  ret i32 0
}

