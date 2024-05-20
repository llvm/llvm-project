; RUN: opt -S -passes=licm < %s | FileCheck %s
@res = dso_local local_unnamed_addr global i32 0, align 4

define dso_local void @test(ptr noalias nocapture noundef readonly %a, i32 noundef signext %N) local_unnamed_addr #0 {
  ; Preheader:
  entry:
    br label %for.cond

  ; Loop:
  for.cond:                                         ; preds = %for.inc, %entry
    %i.0 = phi i32 [ 0, %entry ], [ %inc1, %for.inc ]
    %cmp = icmp slt i32 %i.0, %N
    br i1 %cmp, label %for.body, label %for.cond.cleanup

  for.body:                                         ; preds = %for.cond
    %idxprom = zext i32 %i.0 to i64
    %arrayidx = getelementptr inbounds i32, ptr %a, i64 %idxprom
    %0 = load i32, ptr %arrayidx, align 4
    %tobool.not = icmp eq i32 %0, 0
    br i1 %tobool.not, label %for.inc, label %if.then

  if.then:                                          ; preds = %for.body
    %1 = load i32, ptr @res, align 4
    %inc = add nsw i32 %1, 1
    store i32 %inc, ptr @res, align 4
    br label %for.inc

  for.inc:                                          ; preds = %for.body, %if.then
    %inc1 = add nuw nsw i32 %i.0, 1
    br label %for.cond 

  ; Exit blocks
  for.cond.cleanup:                                 ; preds = %for.cond
    ret void
}

; CHECK:  entry:
; CHECK:    %res.promoted = load i32, ptr @res, align 4
; CHECK:    br label %for.cond

; CHECK:  for.cond:
; CHECK:    %inc3 = phi i32 [ %res.promoted, %entry ], [ %inc2, %for.inc ]
; CHECK:    %i.0 = phi i32 [ 0, %entry ], [ %inc1, %for.inc ]
; CHECK:    %cmp = icmp slt i32 %i.0, %N
; CHECK:    br i1 %cmp, label %for.body, label %for.cond.cleanup

; CHECK:  for.body:
; CHECK:    %idxprom = zext i32 %i.0 to i64
; CHECK:    %arrayidx = getelementptr inbounds i32, ptr %a, i64 %idxprom
; CHECK:    %0 = load i32, ptr %arrayidx, align 4
; CHECK:    %tobool.not = icmp eq i32 %0, 0
; CHECK:    br i1 %tobool.not, label %for.inc, label %if.then

; CHECK:  if.then:
; CHECK:    %inc = add nsw i32 %inc3, 1
; CHECK:    store i32 %inc, ptr @res, align 4
; CHECK:    br label %for.inc

; CHECK:  for.inc:
; CHECK:    %inc2 = phi i32 [ %inc, %if.then ], [ %inc3, %for.body ]
; CHECK:    %inc1 = add nuw nsw i32 %i.0, 1
; CHECK:    br label %for.cond

; CHECK:  for.cond.cleanup:
  ; CHECK:    ret void
; CHECK:  }