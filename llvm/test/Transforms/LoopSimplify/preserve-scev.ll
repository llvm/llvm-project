; RUN: opt -S < %s -indvars | opt -analyze -iv-users | grep "%cmp = icmp slt i32" | grep "= {%\.ph,+,1}<%for.cond>"
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

; Notify SCEV when removing an ExitingBlock.
; CHECK: @mergeExit
; CHECK: while.cond191:
; CHECK: br i1 %or.cond, label %while.body197
; CHECK-NOT: land.rhs:
; CHECK: ret
define void @mergeExit(i32 %MapAttrCount) nounwind uwtable ssp {
entry:
  br i1 undef, label %if.then124, label %if.end126

if.then124:                                       ; preds = %entry
  unreachable

if.end126:                                        ; preds = %entry
  br i1 undef, label %while.body.lr.ph, label %if.end591

while.body.lr.ph:                                 ; preds = %if.end126
  br i1 undef, label %if.end140, label %if.then137

if.then137:                                       ; preds = %while.body.lr.ph
  unreachable

if.end140:                                        ; preds = %while.body.lr.ph
  br i1 undef, label %while.cond191.outer, label %if.then148

if.then148:                                       ; preds = %if.end140
  unreachable

while.cond191.outer:                              ; preds = %if.then205, %if.end140
  br label %while.cond191

while.cond191:                                    ; preds = %while.body197, %while.cond191.outer
  %CppIndex.0 = phi i32 [ %inc, %while.body197 ], [ undef, %while.cond191.outer ]
  br i1 undef, label %land.rhs, label %if.then216

land.rhs:                                         ; preds = %while.cond191
  %inc = add i32 %CppIndex.0, 1
  %cmp196 = icmp ult i32 %inc, %MapAttrCount
  br i1 %cmp196, label %while.body197, label %if.then216

while.body197:                                    ; preds = %land.rhs
  br i1 undef, label %if.then205, label %while.cond191

if.then205:                                       ; preds = %while.body197
  br label %while.cond191.outer

if.then216:                                       ; preds = %land.rhs, %while.cond191
  br i1 undef, label %if.else, label %if.then221

if.then221:                                       ; preds = %if.then216
  unreachable

if.else:                                          ; preds = %if.then216
  br i1 undef, label %if.then266, label %if.end340

if.then266:                                       ; preds = %if.else
  switch i32 undef, label %if.else329 [
    i32 17, label %if.then285
    i32 19, label %if.then285
    i32 18, label %if.then285
    i32 15, label %if.then285
  ]

if.then285:                                       ; preds = %if.then266, %if.then266, %if.then266, %if.then266
  br i1 undef, label %if.then317, label %if.else324

if.then317:                                       ; preds = %if.then285
  br label %if.end340

if.else324:                                       ; preds = %if.then285
  unreachable

if.else329:                                       ; preds = %if.then266
  unreachable

if.end340:                                        ; preds = %if.then317, %if.else
  unreachable

if.end591:                                        ; preds = %if.end126
  br i1 undef, label %cond.end, label %cond.false

cond.false:                                       ; preds = %if.end591
  unreachable

cond.end:                                         ; preds = %if.end591
  ret void
}
