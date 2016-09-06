; RUN: llc -march=hexagon -hexagon-small-data-threshold=0 -disable-hexagon-amodeopt < %s | FileCheck %s --check-prefix=CHECK-NO-AMODE
; RUN: llc -march=hexagon -hexagon-small-data-threshold=0 -disable-hexagon-amodeopt=0 -hexagon-amode-growth-limit=4 < %s | FileCheck %s --check-prefix=CHECK-AMODE

; CHECK-NO-AMODE: [[REG0:(r[0-9]+)]] = ##global_2
; CHECK-NO-AMODE: memw([[REG0]] + {{.*}}<<#2) =

; CHECK-AMODE: [[REG1:(r[0-9]+)]] = memw(##global_1)
; CHECK-AMODE: memw([[REG1]]<<#2 + ##global_2) =

@global_1 = external global i32, align 4
@global_2 = external global [128 x i32], align 8

declare i32 @foo(i32, i32) #0

define i32 @fred(i32 %a0, i32 %a1, i32* %p) #0 {
entry:
  %call24 = tail call i32 @foo(i32 %a0, i32 1) #0
  %tobool26 = icmp eq i32 %call24, 0
  br i1 %tobool26, label %while.end, label %while.body.lr.ph

while.body.lr.ph:                                 ; preds = %entry
  %cmp3 = icmp sgt i32 %a1, 19
  %sub = sub nsw i32 19, %a0
  %xor = xor i32 %a0, 1
  br i1 %cmp3, label %while.body.us.preheader, label %while.body.preheader

while.body.preheader:                             ; preds = %while.body.lr.ph
  br label %while.body

while.body.us.preheader:                          ; preds = %while.body.lr.ph
  br label %while.body.us

while.body.us:                                    ; preds = %while.body.us.preheader, %while.cond.backedge.us
  %call27.us = phi i32 [ %call.us, %while.cond.backedge.us ], [ %call24, %while.body.us.preheader ]
  %x0 = load i32, i32* %p, align 4, !tbaa !4
  %cmp.us = icmp sgt i32 %x0, 0
  br i1 %cmp.us, label %if.then.us, label %if.end.us

if.then.us:                                       ; preds = %while.body.us
  %sext.us = shl i32 %call27.us, 24
  %conv2.us = ashr i32 %sext.us, 24
  %x10 = tail call i32 @foo(i32 %conv2.us, i32 %sext.us) #0
  br label %if.end.us

if.end.us:                                        ; preds = %if.then.us, %while.body.us
  %x1 = load i32, i32* %p, align 4, !tbaa !4
  %call8.us = tail call i32 @foo(i32 %sub, i32 %a1) #0
  %tobool11.us = icmp eq i32 %call8.us, 0
  br i1 %tobool11.us, label %while.cond.backedge.us, label %if.then12.us

if.then12.us:                                     ; preds = %if.end.us
  %x3 = load i32, i32* %p, align 4, !tbaa !4
  %sub13.us = sub i32 %x3, %x1
  %x4 = load i32, i32* @global_1, align 4, !tbaa !4
  %arrayidx.us = getelementptr inbounds [128 x i32], [128 x i32]* @global_2, i32 0, i32 %x4
  store i32 %sub13.us, i32* %arrayidx.us, align 4, !tbaa !4
  br label %while.cond.backedge.us

while.cond.backedge.us:                           ; preds = %if.then12.us, %if.end.us
  %call.us = tail call i32 @foo(i32 %a0, i32 2) #0
  %tobool.us = icmp eq i32 %call.us, 0
  br i1 %tobool.us, label %while.end.loopexit, label %while.body.us

while.body:                                       ; preds = %while.body.preheader, %while.cond.backedge
  %call27 = phi i32 [ %call, %while.cond.backedge ], [ %call24, %while.body.preheader ]
  %x5 = load i32, i32* %p, align 4, !tbaa !4
  %cmp = icmp sgt i32 %x5, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %while.body
  %sext = shl i32 %call27, 24
  %conv2 = ashr i32 %sext, 24
  %x11 = tail call i32 @foo(i32 %conv2, i32 %sext) #0
  br label %if.end

if.end:                                           ; preds = %if.then, %while.body
  %tobool11 = icmp eq i32 %call27, 0
  br i1 %tobool11, label %while.cond.backedge, label %if.then12

if.then12:                                        ; preds = %if.end
  %x7 = load i32, i32* @global_1, align 4, !tbaa !4
  %arrayidx = getelementptr inbounds [128 x i32], [128 x i32]* @global_2, i32 0, i32 %x7
  store i32 0, i32* %arrayidx, align 4, !tbaa !4
  br label %while.cond.backedge

while.cond.backedge:                              ; preds = %if.then12, %if.end
  %call = tail call i32 @foo(i32 %a0, i32 3) #0
  %tobool = icmp eq i32 %call, 0
  br i1 %tobool, label %while.end.loopexit33, label %while.body

while.end.loopexit:                               ; preds = %while.cond.backedge.us
  br label %while.end

while.end.loopexit33:                             ; preds = %while.cond.backedge
  br label %while.end

while.end:                                        ; preds = %while.end.loopexit33, %while.end.loopexit, %entry
  ret i32 0
}

attributes #0 = { nounwind }

!1 = !{!2, !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"int", !2, i64 0}
