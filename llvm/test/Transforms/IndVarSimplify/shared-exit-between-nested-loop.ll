; RUN: opt -passes=indvars -S < %s | FileCheck %s

@__const.e.f = private unnamed_addr constant <{ <{ [2 x i32], [2 x i32], [8 x [2 x i32]] }>, [10 x [2 x i32]] }> <{ <{ [2 x i32], [2 x i32], [8 x [2 x i32]] }> <{ [2 x i32] [i32 4, i32 8], [2 x i32] [i32 3, i32 10], [8 x [2 x i32]] zeroinitializer }>, [10 x [2 x i32]] [[2 x i32] zeroinitializer, [2 x i32] zeroinitializer, [2 x i32] [i32 1, i32 5], [2 x i32] [i32 2080555007, i32 0], [2 x i32] zeroinitializer, [2 x i32] zeroinitializer, [2 x i32] zeroinitializer, [2 x i32] zeroinitializer, [2 x i32] zeroinitializer, [2 x i32] zeroinitializer] }>, align 4

define dso_local i8 @main() local_unnamed_addr {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.inc15, %entry
  %storemerge32 = phi i32 [ 0, %entry ], [ %inc16, %for.inc15 ]
  br label %for.cond4

for.cond4:                                        ; preds = %for.inc, %for.cond1.preheader
  %storemerge2531 = phi i32 [ 0, %for.cond1.preheader ], [ %inc, %for.inc ]
  %tobool = icmp eq i32 1, 0
  br i1 %tobool, label %cleanup, label %if.then

if.then:                                          ; preds = %for.cond4
; CHECK:        if.then:                                          ; preds = %for.cond4
; CHECK-NEXT:     br i1 false, label %cleanup, label %for.inc
  %add = add nuw nsw i32 %storemerge2531, 2
  %idxprom8 = zext i32 %add to i64
  %arrayidx10 = getelementptr inbounds <{ <{ [2 x i32], [2 x i32], [8 x [2 x i32]] }>, [10 x [2 x i32]] }>, <{ <{ [2 x i32], [2 x i32], [8 x [2 x i32]] }>, [10 x [2 x i32]] }>* @__const.e.f, i64 0, i32 1, i64 %idxprom8, i64 0
  %0 = load i32, i32* %arrayidx10, align 4
  %tobool11 = icmp eq i32 %0, 0
  br i1 %tobool11, label %cleanup, label %for.inc

for.inc:                                          ; preds = %if.then
  %inc = add nuw nsw i32 %storemerge2531, 1
  %cmp2 = icmp ult i32 %inc, 2
  br i1 %cmp2, label %for.cond4, label %for.inc15

cleanup:                                          ; preds = %if.then, %for.cond4
  br label %return

for.inc15:                                        ; preds = %for.inc
; CHECK:        for.inc15:
; CHECK-NEXT:     %inc16 = add nuw nsw i32 %storemerge32, 1
; CHECK-NEXT:     %cmp = icmp eq i32 %inc16, 4
; CHECK-NEXT:     br i1 %cmp, label %for.cond18thread-pre-split, label %for.cond1.preheader
  %inc16 = add nuw nsw i32 %storemerge32, 1
  %cmp = icmp eq i32 %inc16, 4
  br i1 %cmp, label %for.cond18thread-pre-split, label %for.cond1.preheader

for.cond18thread-pre-split:                       ; preds = %for.inc15
  br i1 1, label %return.loopexit, label %for.inc21.lr.ph

for.inc21.lr.ph:                                  ; preds = %for.cond18thread-pre-split
  br label %for.inc21

for.inc21:                                        ; preds = %for.inc21, %for.inc21.lr.ph
  %1 = phi i32 [ undef, %for.inc21.lr.ph ], [ %inc22, %for.inc21 ]
  %inc22 = add nsw i32 %1, 1
  br i1 true, label %for.cond18.return.loopexit_crit_edge, label %for.inc21

for.cond18.return.loopexit_crit_edge:             ; preds = %for.inc21
  br label %return.loopexit

return.loopexit:                                  ; preds = %for.cond18.return.loopexit_crit_edge, %for.cond18thread-pre-split
  br label %return

return:                                           ; preds = %return.loopexit, %cleanup
  ret i8 undef
}
