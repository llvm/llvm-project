; REQUIRES: asserts
; RUN: opt < %s -disable-output -stats -passes=loop-unroll -unroll-runtime -unroll-partial-threshold=200 -unroll-threshold=400 -info-output-file - | FileCheck %s --check-prefix=STATS
; RUN: opt < %s -disable-output -stats -passes='require<opt-remark-emit>,loop-unroll' -unroll-runtime -unroll-partial-threshold=200 -unroll-threshold=400 -info-output-file - | FileCheck %s --check-prefix=STATS

; Test that nested loops can be unrolled.  We need to increase threshold to do it

; STATS: 2 loop-unroll - Number of loops unrolled (completely or otherwise)

define i32 @nested(i32* nocapture %a, i32 %n, i32 %m) nounwind uwtable readonly {
entry:
  %cmp11 = icmp sgt i32 %n, 0
  br i1 %cmp11, label %for.cond1.preheader.lr.ph, label %for.end7

for.cond1.preheader.lr.ph:                        ; preds = %entry
  %cmp28 = icmp sgt i32 %m, 0
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.inc5, %for.cond1.preheader.lr.ph
  %indvars.iv16 = phi i64 [ 0, %for.cond1.preheader.lr.ph ], [ %indvars.iv.next17, %for.inc5 ]
  %sum.012 = phi i32 [ 0, %for.cond1.preheader.lr.ph ], [ %sum.1.lcssa, %for.inc5 ]
  br i1 %cmp28, label %for.body3, label %for.inc5

for.body3:                                        ; preds = %for.cond1.preheader, %for.body3
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body3 ], [ 0, %for.cond1.preheader ]
  %sum.19 = phi i32 [ %add4, %for.body3 ], [ %sum.012, %for.cond1.preheader ]
  %0 = add nsw i64 %indvars.iv, %indvars.iv16
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %0
  %1 = load i32, i32* %arrayidx, align 4
  %add4 = add nsw i32 %1, %sum.19
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %m
  br i1 %exitcond, label %for.inc5, label %for.body3

for.inc5:                                         ; preds = %for.body3, %for.cond1.preheader
  %sum.1.lcssa = phi i32 [ %sum.012, %for.cond1.preheader ], [ %add4, %for.body3 ]
  %indvars.iv.next17 = add i64 %indvars.iv16, 1
  %lftr.wideiv18 = trunc i64 %indvars.iv.next17 to i32
  %exitcond19 = icmp eq i32 %lftr.wideiv18, %n
  br i1 %exitcond19, label %for.end7, label %for.cond1.preheader

for.end7:                                         ; preds = %for.inc5, %entry
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %sum.1.lcssa, %for.inc5 ]
  ret i32 %sum.0.lcssa
}

