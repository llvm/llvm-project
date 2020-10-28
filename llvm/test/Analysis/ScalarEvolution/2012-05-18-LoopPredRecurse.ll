; RUN: opt < %s -iv-users -S -disable-output
; RUN: opt < %s -passes='require<iv-users>' -S -disable-output
;
; PR12868: Infinite recursion:
; getUDivExpr()->getZeroExtendExpr()->isLoopBackedgeGuardedBy()
;
; We actually want SCEV simplification to fail gracefully in this
; case, so there's no output to check, just the absence of stack overflow.

@c = common global i8 0, align 1

define i32 @func() {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %storemerge = phi i8 [ -1, %entry ], [ %inc, %for.body ]
  %ui.0 = phi i32 [ undef, %entry ], [ %div, %for.body ]
  %tobool = icmp eq i8 %storemerge, 0
  br i1 %tobool, label %for.end, label %for.body

for.body:                                         ; preds = %for.cond
  %conv = sext i8 %storemerge to i32
  %div = lshr i32 %conv, 1
  %tobool2 = icmp eq i32 %div, 0
  %inc = add i8 %storemerge, 1
  br i1 %tobool2, label %for.cond, label %for.end

for.end:                                          ; preds = %for.body, %for.cond
  ret i32 0
}
