; RUN: llc -march=hexagon -O2 < %s | FileCheck %s
; Check if the three stores in the loop were predicated.
; CHECK: if{{.*}}memw
; CHECK: if{{.*}}memw
; CHECK: if{{.*}}memw

target triple = "hexagon"

define void @fred(i32 %n, ptr %bp) #0 {
entry:
  %cmp16 = icmp eq i32 %n, 0
  br i1 %cmp16, label %for.end, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  %cmp2 = icmp ugt i32 %n, 32
  br label %for.body

for.body:                                         ; preds = %for.inc, %for.body.lr.ph
  %i.017 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.inc ]
  %call = tail call i32 @foo(ptr %bp) nounwind
  %call1 = tail call i32 @bar(ptr %bp) nounwind
  br i1 %cmp2, label %if.then, label %if.else

if.then:                                          ; preds = %for.body
  %arrayidx = getelementptr inbounds i32, ptr %bp, i32 %i.017
  store i32 %call, ptr %arrayidx, align 4, !tbaa !0
  %add = add i32 %i.017, 2
  %arrayidx3 = getelementptr inbounds i32, ptr %bp, i32 %add
  store i32 %call1, ptr %arrayidx3, align 4, !tbaa !0
  br label %for.inc

if.else:                                          ; preds = %for.body
  %or = or i32 %call1, %call
  %arrayidx4 = getelementptr inbounds i32, ptr %bp, i32 %i.017
  store i32 %or, ptr %arrayidx4, align 4, !tbaa !0
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

declare i32 @foo(ptr) nounwind

declare i32 @bar(ptr) nounwind

attributes #0 = { nounwind "target-cpu"="hexagonv5" }

!0 = !{!"int", !1}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
