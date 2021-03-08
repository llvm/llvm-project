; REQUIRES: asserts
; RUN: opt -S -passes='loop-interchange' -debug-only=loop-interchange < %s 2>&1 > /dev/null | FileCheck %s

; Test case of PR48113.
; The loops should not be interchanged becuase they are not tightly nested.

@a = dso_local local_unnamed_addr global i8 0, align 1
@b = dso_local local_unnamed_addr global [1 x [2 x i32]] zeroinitializer, align 4
@c = dso_local local_unnamed_addr global [1 x [9 x i8]] zeroinitializer, align 1
@d = dso_local local_unnamed_addr global i8 0, align 1
@e = dso_local local_unnamed_addr global i8* null, align 8
@f = internal unnamed_addr global i32 0, align 4
@g = dso_local local_unnamed_addr global i16 0, align 2

; CHECK: Processing InnerLoopId = 1 and OuterLoopId = 0
; CHECK-NEXT: Checking if loops are tightly nested
; CHECK-NEXT: Loops not tightly nested
; CHECK-NEXT: Not interchanging loops. Cannot prove legality

define dso_local i32 @main() local_unnamed_addr {
entry:
  %.pr.i = load i32, i32* @f, align 4
  %cmp3.i = icmp ult i32 %.pr.i, 3
  br i1 %cmp3.i, label %for.cond1.preheader.lr.ph.i, label %h.exit

for.cond1.preheader.lr.ph.i:                      ; preds = %entry
  %0 = load i8, i8* @a, align 1
  %tobool.not.i = icmp eq i8 %0, 0
  %1 = load i8*, i8** @e, align 8
  %2 = zext i32 %.pr.i to i64
  br label %for.cond1.preheader.i

for.cond1.preheader.i:                            ; preds = %if.end.i, %for.cond1.preheader.lr.ph.i
  %indvars.iv4.i = phi i64 [ %2, %for.cond1.preheader.lr.ph.i ], [ %indvars.iv.next5.i, %if.end.i ]
  br label %for.body3.i

for.body3.i:                                      ; preds = %for.body3.i, %for.cond1.preheader.i
  %indvars.iv.i = phi i64 [ 0, %for.cond1.preheader.i ], [ %indvars.iv.next.i, %for.body3.i ]
  %arrayidx5.i = getelementptr inbounds [1 x [9 x i8]], [1 x [9 x i8]]* @c, i64 0, i64 %indvars.iv.i, i64 %indvars.iv4.i
  %3 = load i8, i8* %arrayidx5.i, align 1
  %conv.i = sext i8 %3 to i32
  %arrayidx8.i = getelementptr inbounds [1 x [2 x i32]], [1 x [2 x i32]]* @b, i64 0, i64 %indvars.iv.i, i64 1
  store i32 %conv.i, i32* %arrayidx8.i, align 4
  %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.i, 1
  %exitcond.not.i = icmp eq i64 %indvars.iv.next.i, 3
  br i1 %exitcond.not.i, label %for.end.i, label %for.body3.i

for.end.i:                                        ; preds = %for.body3.i
  %4 = load i8, i8* @d, align 1
  %inc9.i = add i8 %4, 1
  store i8 %inc9.i, i8* @d, align 1
  br i1 %tobool.not.i, label %if.end.i, label %if.then.i

if.then.i:                                        ; preds = %for.end.i
  %5 = load i8, i8* %1, align 1
  %conv10.i = sext i8 %5 to i16
  store i16 %conv10.i, i16* @g, align 2
  br label %if.end.i

if.end.i:                                         ; preds = %if.then.i, %for.end.i
  %indvars.iv.next5.i = add nuw nsw i64 %indvars.iv4.i, 1
  %exitcond6.not.i = icmp eq i64 %indvars.iv.next5.i, 3
  br i1 %exitcond6.not.i, label %for.cond.for.end13_crit_edge.i, label %for.cond1.preheader.i

for.cond.for.end13_crit_edge.i:                   ; preds = %if.end.i
  store i32 3, i32* @f, align 4
  br label %h.exit

h.exit:                                           ; preds = %entry, %for.cond.for.end13_crit_edge.i
  %6 = load i8, i8* @d, align 1
  %conv = sext i8 %6 to i32
  ret i32 %conv
}
