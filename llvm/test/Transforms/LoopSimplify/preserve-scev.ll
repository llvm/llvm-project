; RUN: opt -S < %s -indvars | opt -analyze -iv-users | grep {%cmp = icmp slt i32} | grep {= \{%\\.ph,+,1\}<%for.cond>}
; PR8079

; LoopSimplify should invalidate indvars when splitting out the
; inner loop.

@maxStat = external global i32

define i32 @test() nounwind {
entry:
  br label %for.cond

for.cond:                                         ; preds = %if.then5, %if.end, %entry
  %cuts.1 = phi i32 [ 0, %entry ], [ %inc, %if.then5 ], [ %cuts.1, %if.end ]
  %0 = phi i32 [ 0, %entry ], [ %add, %if.end ], [ %add, %if.then5 ]
  %add = add i32 %0, 1
  %cmp = icmp slt i32 %0, 1
  %tmp1 = load i32* @maxStat, align 4
  br i1 %cmp, label %for.body, label %for.cond14.preheader

for.cond14.preheader:                             ; preds = %for.cond
  %cmp1726 = icmp sgt i32 %tmp1, 0
  br i1 %cmp1726, label %for.body18, label %return

for.body:                                         ; preds = %for.cond
  %cmp2 = icmp sgt i32 %tmp1, 100
  br i1 %cmp2, label %return, label %if.end

if.end:                                           ; preds = %for.body
  %cmp4 = icmp sgt i32 %tmp1, -1
  br i1 %cmp4, label %if.then5, label %for.cond

if.then5:                                         ; preds = %if.end
  call void @foo() nounwind
  %inc = add i32 %cuts.1, 1
  br label %for.cond

for.body18:                                       ; preds = %for.body18, %for.cond14.preheader
  %i13.027 = phi i32 [ %1, %for.body18 ], [ 0, %for.cond14.preheader ]
  call void @foo() nounwind
  %1 = add nsw i32 %i13.027, 1
  %tmp16 = load i32* @maxStat, align 4
  %cmp17 = icmp slt i32 %1, %tmp16
  br i1 %cmp17, label %for.body18, label %return

return:                                           ; preds = %for.body18, %for.body, %for.cond14.preheader
  ret i32 0
}

declare void @foo() nounwind
