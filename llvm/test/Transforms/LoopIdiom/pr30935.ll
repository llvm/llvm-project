; RUN: opt -loop-idiom -S < %s | FileCheck %s

; CHECK-LABEL: define i32 @main(
; CHECK: udiv
; CHECK-NOT: udiv
; CHECK: call void @llvm.memset.p0i8.i64

@a = common local_unnamed_addr global [4 x i8] zeroinitializer, align 1
@b = common local_unnamed_addr global i32 0, align 4
@c = common local_unnamed_addr global i32 0, align 4
@d = common local_unnamed_addr global i32 0, align 4
@e = common local_unnamed_addr global i32 0, align 4
@f = common local_unnamed_addr global i32 0, align 4
@g = common local_unnamed_addr global i32 0, align 4
@h = common local_unnamed_addr global i64 0, align 8


define i32 @main() local_unnamed_addr #0 {
entry:
  %0 = load i32, i32* @e, align 4
  %tobool19 = icmp eq i32 %0, 0
  %1 = load i32, i32* @c, align 4
  %cmp10 = icmp eq i32 %1, 0
  %2 = load i32, i32* @g, align 4
  %3 = load i32, i32* @b, align 4
  %tobool = icmp eq i32 %0, 0
  br label %for.cond

for.cond.loopexit:                                ; preds = %for.inc14
  br label %for.cond.backedge

for.cond:                                         ; preds = %for.cond.backedge, %entry
  %.pr = load i32, i32* @f, align 4
  %cmp20 = icmp eq i32 %.pr, 0
  br i1 %cmp20, label %for.cond2.preheader.preheader, label %for.cond.backedge

for.cond.backedge:                                ; preds = %for.cond, %for.cond.loopexit
  br label %for.cond

for.cond2.preheader.preheader:                    ; preds = %for.cond
  br label %for.cond2.preheader

for.cond2.preheader:                              ; preds = %for.cond2.preheader.preheader, %for.inc14
  br i1 %tobool19, label %for.cond9, label %for.body3.lr.ph

for.body3.lr.ph:                                  ; preds = %for.cond2.preheader
  %div = udiv i32 %2, %3
  %conv = zext i32 %div to i64
  br label %for.body3

for.cond4.for.cond2.loopexit_crit_edge:           ; preds = %for.body6
  store i32 0, i32* @d, align 4
  br label %for.cond2.loopexit

for.cond2.loopexit:                               ; preds = %for.cond4.for.cond2.loopexit_crit_edge, %for.body3
  br i1 %tobool, label %for.cond2.for.cond9_crit_edge, label %for.body3

for.body3:                                        ; preds = %for.body3.lr.ph, %for.cond2.loopexit
  %.pr17 = load i32, i32* @d, align 4
  %tobool518 = icmp eq i32 %.pr17, 0
  br i1 %tobool518, label %for.cond2.loopexit, label %for.body6.preheader

for.body6.preheader:                              ; preds = %for.body3
  %4 = zext i32 %.pr17 to i64
  br label %for.body6

for.body6:                                        ; preds = %for.body6.preheader, %for.body6
  %indvars.iv = phi i64 [ %4, %for.body6.preheader ], [ %indvars.iv.next, %for.body6 ]
  %add = add nuw nsw i64 %conv, %indvars.iv
  %arrayidx = getelementptr inbounds [4 x i8], [4 x i8]* @a, i64 0, i64 %add
  store i8 1, i8* %arrayidx, align 1
  %5 = trunc i64 %indvars.iv to i32
  %inc = add i32 %5, 1
  %tobool5 = icmp eq i32 %inc, 0
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br i1 %tobool5, label %for.cond4.for.cond2.loopexit_crit_edge, label %for.body6

for.cond2.for.cond9_crit_edge:                    ; preds = %for.cond2.loopexit
  store i64 %conv, i64* @h, align 8
  br label %for.cond9

for.cond9:                                        ; preds = %for.cond2.for.cond9_crit_edge, %for.cond2.preheader
  br i1 %cmp10, label %for.body12, label %for.inc14

for.body12:                                       ; preds = %for.cond9
  ret i32 0

for.inc14:                                        ; preds = %for.cond9
  %6 = load i32, i32* @f, align 4
  %inc15 = add i32 %6, 1
  store i32 %inc15, i32* @f, align 4
  %cmp = icmp eq i32 %inc15, 0
  br i1 %cmp, label %for.cond2.preheader, label %for.cond.loopexit
}
