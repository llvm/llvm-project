; RUN: opt -S < %s -loop-idiom | FileCheck %s
; CHECK-NOT: memset
; check that memset is not generated (for stores) because that will result
; in udiv hoisted out of the loop by the SCEV Expander
; TODO: ideally we should be able to generate memset
; if SCEV expander is taught to generate the dependencies
; at the right point.

@a = global i32 0, align 4
@b = global i32 0, align 4
@c = external local_unnamed_addr global [1 x i8], align 1

define void @e() local_unnamed_addr {
entry:
  %d0 = load i32, i32* @a, align 4
  %d1 = load i32, i32* @b, align 4
  br label %for.cond1thread-pre-split

for.cond1thread-pre-split:                        ; preds = %for.body5, %entry
  %div = udiv i32 %d0, %d1
  br label %for.body5

for.body5:                                        ; preds = %for.body5, %for.cond1thread-pre-split
  %indvars.iv = phi i64 [ 0, %for.cond1thread-pre-split ], [ %indvars.iv.next, %for.body5 ]
  %divx = sext i32 %div to i64
  %0 = add nsw i64 %divx, %indvars.iv
  %arrayidx = getelementptr inbounds [1 x i8], [1 x i8]* @c, i64 0, i64 %0
  store i8 0, i8* %arrayidx, align 1
  %indvars.iv.next = add nsw i64 %indvars.iv, 1
  %1 = trunc i64 %indvars.iv.next to i32
  %tobool4 = icmp eq i32 %1, 0
  br i1 %tobool4, label %for.cond1thread-pre-split, label %for.body5
}

; The loop's trip count is depending on an unsafe operation
; udiv. SCEV expander hoists it out of the loop, so loop-idiom
; should check that the memset is not generated in this case.
define void @f(i32 %a, i32 %b, i8* nocapture %x) local_unnamed_addr {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body6, %entry
  %div = udiv i32 %a, %b
  %conv = zext i32 %div to i64
  br label %for.body6

for.body6:                                        ; preds = %for.body6, %for.body
  %i.09 = phi i64 [ %inc, %for.body6 ], [ 0, %for.body ]
  %arrayidx = getelementptr inbounds i8, i8* %x, i64 %i.09
  store i8 0, i8* %arrayidx, align 1
  %inc = add nuw nsw i64 %i.09, 1
  %cmp3 = icmp slt i64 %inc, %conv
  br i1 %cmp3, label %for.body6, label %for.body
}

