; RUN: llc -O2 -mcpu=hexagonv5 < %s | FileCheck %s
; Check if the three stores in the loop were predicated.
; CHECK: if{{.*}}memw
; CHECK: if{{.*}}memw
; CHECK: if{{.*}}memw

target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon"

define void @fred(i32 %n, i32* %bp) nounwind {
entry:
  %cmp16 = icmp eq i32 %n, 0
  br i1 %cmp16, label %for.end, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  %cmp2 = icmp ugt i32 %n, 32
  br label %for.body

for.body:                                         ; preds = %for.inc, %for.body.lr.ph
  %i.017 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.inc ]
  %call = tail call i32 @foo(i32* %bp) nounwind
  %call1 = tail call i32 @bar(i32* %bp) nounwind
  br i1 %cmp2, label %if.then, label %if.else

if.then:                                          ; preds = %for.body
  %arrayidx = getelementptr inbounds i32, i32* %bp, i32 %i.017
  store i32 %call, i32* %arrayidx, align 4, !tbaa !0
  %add = add i32 %i.017, 2
  %arrayidx3 = getelementptr inbounds i32, i32* %bp, i32 %add
  store i32 %call1, i32* %arrayidx3, align 4, !tbaa !0
  br label %for.inc

if.else:                                          ; preds = %for.body
  %or = or i32 %call1, %call
  %arrayidx4 = getelementptr inbounds i32, i32* %bp, i32 %i.017
  store i32 %or, i32* %arrayidx4, align 4, !tbaa !0
  br label %for.inc

for.inc:                                          ; preds = %if.then, %if.else
  %inc = add i32 %i.017, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.inc
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}

declare i32 @foo(i32*) nounwind

declare i32 @bar(i32*) nounwind

!0 = !{!"int", !1}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
