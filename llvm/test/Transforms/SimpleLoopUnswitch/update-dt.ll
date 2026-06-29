; RUN: opt < %s -passes='loop-mssa(simple-loop-unswitch)' -verify-memoryssa -disable-output

; Ensure that MemorySSA and DomTree are correctly updated after a
; continue-to-break transformation during loop unswitching.
; This addresses the bug that caused PR #193989 to be reverted.

define dso_local void @a() {
entry:
  %c = alloca [4 x i32], align 4
  br label %for.cond

for.cond:                                         ; preds = %for.end5, %entry
  call void @llvm.lifetime.start.p0(ptr nonnull %c)
  %cmp3 = icmp ult ptr inttoptr (i64 2 to ptr), @a
  br label %for.cond2.preheader

for.cond2.preheader:                              ; preds = %for.cond, %for.inc
  %b.08 = phi i32 [ 0, %for.cond ], [ %inc, %for.inc ]
  br i1 %cmp3, label %for.body4.lr.ph, label %for.inc

for.body4.lr.ph:                                  ; preds = %for.cond2.preheader
  %idxprom = zext nneg i32 %b.08 to i64
  %arrayidx = getelementptr inbounds nuw [4 x i8], ptr %c, i64 %idxprom
  %arrayidx.promoted = load i32, ptr %arrayidx, align 4
  br i1 %cmp3, label %for.body4.lr.ph.split.us, label %for.body4.lr.ph.split

for.body4.lr.ph.split.us:                         ; preds = %for.body4.lr.ph
  %arrayidx.promoted.lcssa = phi i32 [ %arrayidx.promoted, %for.body4.lr.ph ]
  br label %for.body4.us

for.body4.us:                                     ; preds = %for.body4.us, %for.body4.lr.ph.split.us
  %0 = phi i32 [ %arrayidx.promoted.lcssa, %for.body4.lr.ph.split.us ], [ %add.us, %for.body4.us ]
  %add.us = add nsw i32 %0, 1
  br label %for.body4.us

for.body4.lr.ph.split:                            ; preds = %for.body4.lr.ph
  %add = add nsw i32 %arrayidx.promoted, 1
  store i32 %add, ptr %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body4.lr.ph.split, %for.cond2.preheader
  %inc = add nuw nsw i32 %b.08, 1
  %cmp = icmp samesign ult i32 %inc, 4
  br i1 %cmp, label %for.cond2.preheader, label %for.end5

for.end5:                                         ; preds = %for.inc
  call void @llvm.lifetime.end.p0(ptr nonnull %c)
  br label %for.cond
}


declare void @llvm.lifetime.start.p0(ptr captures(none))
