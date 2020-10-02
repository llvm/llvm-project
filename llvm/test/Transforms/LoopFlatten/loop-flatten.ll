; RUN: opt < %s -S -loop-flatten -verify-loop-info -verify-dom-info -verify-scev -verify | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

; CHECK-LABEL: test1
; Simple loop where the IV's is constant
define i32 @test1(i32 %val, i16* nocapture %A) {
entry:
  br label %for.body
; CHECK: entry:
; CHECK:   %flatten.tripcount = mul i32 20, 10
; CHECK:   br label %for.body

for.body:                                         ; preds = %entry, %for.inc6
  %i.018 = phi i32 [ 0, %entry ], [ %inc7, %for.inc6 ]
  %mul = mul nuw nsw i32 %i.018, 20
  br label %for.body3
; CHECK: for.body:
; CHECK:   %i.018 = phi i32 [ 0, %entry ], [ %inc7, %for.inc6 ]
; CHECK:   %mul = mul nuw nsw i32 %i.018, 20
; CHECK:   br label %for.body3

for.body3:                                        ; preds = %for.body, %for.body3
  %j.017 = phi i32 [ 0, %for.body ], [ %inc, %for.body3 ]
  %add = add nuw nsw i32 %j.017, %mul
  %arrayidx = getelementptr inbounds i16, i16* %A, i32 %add
  %0 = load i16, i16* %arrayidx, align 2
  %conv16 = zext i16 %0 to i32
  %add4 = add i32 %conv16, %val
  %conv5 = trunc i32 %add4 to i16
  store i16 %conv5, i16* %arrayidx, align 2
  %inc = add nuw nsw i32 %j.017, 1
  %exitcond = icmp ne i32 %inc, 20
  br i1 %exitcond, label %for.body3, label %for.inc6
; CHECK: for.body3:
; CHECK:   %j.017 = phi i32 [ 0, %for.body ]
; CHECK:   %add = add nuw nsw i32 %j.017, %mul
; CHECK:   %arrayidx = getelementptr inbounds i16, i16* %A, i32 %i.018
; CHECK:   %0 = load i16, i16* %arrayidx, align 2
; CHECK:   %conv16 = zext i16 %0 to i32
; CHECK:   %add4 = add i32 %conv16, %val
; CHECK:   %conv5 = trunc i32 %add4 to i16
; CHECK:   store i16 %conv5, i16* %arrayidx, align 2
; CHECK:   %inc = add nuw nsw i32 %j.017, 1
; CHECK:   %exitcond = icmp ne i32 %inc, 20
; CHECK:   br label %for.inc6

for.inc6:                                         ; preds = %for.body3
  %inc7 = add nuw nsw i32 %i.018, 1
  %exitcond19 = icmp ne i32 %inc7, 10
  br i1 %exitcond19, label %for.body, label %for.end8
; CHECK: for.inc6:
; CHECK:   %inc7 = add nuw nsw i32 %i.018, 1
; CHECK:   %exitcond19 = icmp ne i32 %inc7, %flatten.tripcount
; CHECK:   br i1 %exitcond19, label %for.body, label %for.end8

for.end8:                                         ; preds = %for.inc6
  ret i32 10
}


; CHECK-LABEL: test2
; Same as above but non constant IV (which still cannot overflow)
define i32 @test2(i8 zeroext %I, i32 %val, i16* nocapture %A) {
entry:
  %conv = zext i8 %I to i32
  %cmp26 = icmp eq i8 %I, 0
  br i1 %cmp26, label %for.end13, label %for.body.lr.ph.split.us

for.body.lr.ph.split.us:                          ; preds = %entry
  br label %for.body.us
; CHECK: for.body.lr.ph.split.us:
; CHECK:   %flatten.tripcount = mul i32 %conv, %conv
; CHECK:   br label %for.body.us

for.body.us:                                      ; preds = %for.cond2.for.inc11_crit_edge.us, %for.body.lr.ph.split.us
  %i.027.us = phi i32 [ 0, %for.body.lr.ph.split.us ], [ %inc12.us, %for.cond2.for.inc11_crit_edge.us ]
  %mul.us = mul nuw nsw i32 %i.027.us, %conv
  br label %for.body6.us
; CHECK: for.body.us:
; CHECK:   %i.027.us = phi i32 [ 0, %for.body.lr.ph.split.us ], [ %inc12.us, %for.cond2.for.inc11_crit_edge.us ]
; CHECK:   %mul.us = mul nuw nsw i32 %i.027.us, %conv
; CHECK:   br label %for.body6.us

for.body6.us:                                     ; preds = %for.body.us, %for.body6.us
  %j.025.us = phi i32 [ 0, %for.body.us ], [ %inc.us, %for.body6.us ]
  %add.us = add nuw nsw i32 %j.025.us, %mul.us
  %arrayidx.us = getelementptr inbounds i16, i16* %A, i32 %add.us
  %0 = load i16, i16* %arrayidx.us, align 2
  %conv823.us = zext i16 %0 to i32
  %add9.us = add i32 %conv823.us, %val
  %conv10.us = trunc i32 %add9.us to i16
  store i16 %conv10.us, i16* %arrayidx.us, align 2
  %inc.us = add nuw nsw i32 %j.025.us, 1
  %exitcond = icmp ne i32 %inc.us, %conv
  br i1 %exitcond, label %for.body6.us, label %for.cond2.for.inc11_crit_edge.us
; CHECK: for.body6.us:
; CHECK:   %j.025.us = phi i32 [ 0, %for.body.us ]
; CHECK:   %add.us = add nuw nsw i32 %j.025.us, %mul.us
; CHECK:   %arrayidx.us = getelementptr inbounds i16, i16* %A, i32 %i.027.us
; CHECK:   %0 = load i16, i16* %arrayidx.us, align 2
; CHECK:   %conv823.us = zext i16 %0 to i32
; CHECK:   %add9.us = add i32 %conv823.us, %val
; CHECK:   %conv10.us = trunc i32 %add9.us to i16
; CHECK:   store i16 %conv10.us, i16* %arrayidx.us, align 2
; CHECK:   %inc.us = add nuw nsw i32 %j.025.us, 1
; CHECK:   %exitcond = icmp ne i32 %inc.us, %conv
; CHECK:   br label %for.cond2.for.inc11_crit_edge.us

for.cond2.for.inc11_crit_edge.us:                 ; preds = %for.body6.us
  %inc12.us = add nuw nsw i32 %i.027.us, 1
  %exitcond28 = icmp ne i32 %inc12.us, %conv
  br i1 %exitcond28, label %for.body.us, label %for.end13.loopexit
; CHECK: for.cond2.for.inc11_crit_edge.us:                 ; preds = %for.body6.us
; CHECK:   %inc12.us = add nuw nsw i32 %i.027.us, 1
; CHECK:   %exitcond28 = icmp ne i32 %inc12.us, %flatten.tripcount
; CHECK:   br i1 %exitcond28, label %for.body.us, label %for.end13.loopexit

for.end13.loopexit:                               ; preds = %for.cond2.for.inc11_crit_edge.us
  br label %for.end13

for.end13:                                        ; preds = %for.end13.loopexit, %entry
  %i.0.lcssa = phi i32 [ 0, %entry ], [ %conv, %for.end13.loopexit ]
  ret i32 %i.0.lcssa
}


; CHECK-LABEL: test3
; Same as above, uses load to determine it can't overflow
define i32 @test3(i32 %N, i32 %val, i16* nocapture %A) local_unnamed_addr #0 {
entry:
  %cmp21 = icmp eq i32 %N, 0
  br i1 %cmp21, label %for.end8, label %for.body.lr.ph.split.us

for.body.lr.ph.split.us:                          ; preds = %entry
  br label %for.body.us
; CHECK: for.body.lr.ph.split.us:
; CHECK:   %flatten.tripcount = mul i32 %N, %N
; CHECK:   br label %for.body.us

for.body.us:                                      ; preds = %for.cond1.for.inc6_crit_edge.us, %for.body.lr.ph.split.us
  %i.022.us = phi i32 [ 0, %for.body.lr.ph.split.us ], [ %inc7.us, %for.cond1.for.inc6_crit_edge.us ]
  %mul.us = mul i32 %i.022.us, %N
  br label %for.body3.us
; CHECK: for.body.us:
; CHECK:   %i.022.us = phi i32 [ 0, %for.body.lr.ph.split.us ], [ %inc7.us, %for.cond1.for.inc6_crit_edge.us ]
; CHECK:   %mul.us = mul i32 %i.022.us, %N
; CHECK:   br label %for.body3.us

for.body3.us:                                     ; preds = %for.body.us, %for.body3.us
  %j.020.us = phi i32 [ 0, %for.body.us ], [ %inc.us, %for.body3.us ]
  %add.us = add i32 %j.020.us, %mul.us
  %arrayidx.us = getelementptr inbounds i16, i16* %A, i32 %add.us
  %0 = load i16, i16* %arrayidx.us, align 2
  %conv18.us = zext i16 %0 to i32
  %add4.us = add i32 %conv18.us, %val
  %conv5.us = trunc i32 %add4.us to i16
  store i16 %conv5.us, i16* %arrayidx.us, align 2
  %inc.us = add nuw i32 %j.020.us, 1
  %exitcond = icmp ne i32 %inc.us, %N
  br i1 %exitcond, label %for.body3.us, label %for.cond1.for.inc6_crit_edge.us
; CHECK: for.body3.us:
; CHECK:   %j.020.us = phi i32 [ 0, %for.body.us ]
; CHECK:   %add.us = add i32 %j.020.us, %mul.us
; CHECK:   %arrayidx.us = getelementptr inbounds i16, i16* %A, i32 %i.022.us
; CHECK:   %0 = load i16, i16* %arrayidx.us, align 2
; CHECK:   %conv18.us = zext i16 %0 to i32
; CHECK:   %add4.us = add i32 %conv18.us, %val
; CHECK:   %conv5.us = trunc i32 %add4.us to i16
; CHECK:   store i16 %conv5.us, i16* %arrayidx.us, align 2
; CHECK:   %inc.us = add nuw i32 %j.020.us, 1
; CHECK:   %exitcond = icmp ne i32 %inc.us, %N
; CHECK:   br label %for.cond1.for.inc6_crit_edge.us

for.cond1.for.inc6_crit_edge.us:                  ; preds = %for.body3.us
  %inc7.us = add nuw i32 %i.022.us, 1
  %exitcond23 = icmp ne i32 %inc7.us, %N
  br i1 %exitcond23, label %for.body.us, label %for.end8.loopexit
; CHECK: for.cond1.for.inc6_crit_edge.us:
; CHECK:   %inc7.us = add nuw i32 %i.022.us, 1
; CHECK:   %exitcond23 = icmp ne i32 %inc7.us, %flatten.tripcount
; CHECK:   br i1 %exitcond23, label %for.body.us, label %for.end8.loopexit

for.end8.loopexit:                                ; preds = %for.cond1.for.inc6_crit_edge.us
  br label %for.end8

for.end8:                                         ; preds = %for.end8.loopexit, %entry
  %i.0.lcssa = phi i32 [ 0, %entry ], [ %N, %for.end8.loopexit ]
  ret i32 %i.0.lcssa
}


; CHECK-LABEL: test4
; Multiplication cannot overflow, so we can replace the original loop.
define void @test4(i16 zeroext %N, i32* nocapture %C, i32* nocapture readonly %A, i32 %scale) {
entry:
  %conv = zext i16 %N to i32
  %cmp30 = icmp eq i16 %N, 0
  br i1 %cmp30, label %for.cond.cleanup, label %for.body.lr.ph.split.us
; CHECK: entry:
; CHECK: %[[LIMIT:.*]] = zext i16 %N to i32
; CHECK: br i1 %{{.*}} label %for.cond.cleanup, label %for.body.lr.ph.split.us

for.body.lr.ph.split.us:                          ; preds = %entry
  br label %for.body.us
; CHECK: for.body.lr.ph.split.us:
; CHECK: %[[TRIPCOUNT:.*]] = mul i32 %[[LIMIT]], %[[LIMIT]]
; CHECK: br label %for.body.us

for.body.us:                                      ; preds = %for.cond2.for.cond.cleanup6_crit_edge.us, %for.body.lr.ph.split.us
  %i.031.us = phi i32 [ 0, %for.body.lr.ph.split.us ], [ %inc15.us, %for.cond2.for.cond.cleanup6_crit_edge.us ]
  %mul.us = mul nuw nsw i32 %i.031.us, %conv
  br label %for.body7.us
; CHECK: for.body.us:
; CHECK: %[[OUTER_IV:.*]] = phi i32
; CHECK: br label %for.body7.us

for.body7.us:                                     ; preds = %for.body.us, %for.body7.us
; CHECK: for.body7.us:
  %j.029.us = phi i32 [ 0, %for.body.us ], [ %inc.us, %for.body7.us ]
  %add.us = add nuw nsw i32 %j.029.us, %mul.us
  %arrayidx.us = getelementptr inbounds i32, i32* %A, i32 %add.us
; CHECK: getelementptr inbounds i32, i32* %A, i32 %[[OUTER_IV]]
  %0 = load i32, i32* %arrayidx.us, align 4
  %mul9.us = mul nsw i32 %0, %scale
; CHECK: getelementptr inbounds i32, i32* %C, i32 %[[OUTER_IV]]
  %arrayidx13.us = getelementptr inbounds i32, i32* %C, i32 %add.us
  store i32 %mul9.us, i32* %arrayidx13.us, align 4
  %inc.us = add nuw nsw i32 %j.029.us, 1
  %exitcond = icmp ne i32 %inc.us, %conv
  br i1 %exitcond, label %for.body7.us, label %for.cond2.for.cond.cleanup6_crit_edge.us
; CHECK: br label %for.cond2.for.cond.cleanup6_crit_edge.us

for.cond2.for.cond.cleanup6_crit_edge.us:         ; preds = %for.body7.us
  %inc15.us = add nuw nsw i32 %i.031.us, 1
  %exitcond32 = icmp ne i32 %inc15.us, %conv
  br i1 %exitcond32, label %for.body.us, label %for.cond.cleanup.loopexit
; CHECK: for.cond2.for.cond.cleanup6_crit_edge.us:
; CHECK: br i1 %exitcond32, label %for.body.us, label %for.cond.cleanup.loopexit

for.cond.cleanup.loopexit:                        ; preds = %for.cond2.for.cond.cleanup6_crit_edge.us
  br label %for.cond.cleanup
; CHECK: for.cond.cleanup.loopexit:
; CHECK: br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void
; CHECK: for.cond.cleanup:
; CHECK: ret void
}


; CHECK-LABEL: test5
define i32 @test5(i8 zeroext %I, i16 zeroext %J) {
entry:
  %0 = lshr i8 %I, 1
  %div = zext i8 %0 to i32
  %cmp30 = icmp eq i8 %0, 0
  br i1 %cmp30, label %for.cond.cleanup, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  %1 = lshr i16 %J, 1
  %div5 = zext i16 %1 to i32
  %cmp627 = icmp eq i16 %1, 0
  br i1 %cmp627, label %for.body.lr.ph.split, label %for.body.lr.ph.split.us

for.body.lr.ph.split.us:                          ; preds = %for.body.lr.ph
  br label %for.body.us
; CHECK: for.body.lr.ph.split.us:
; CHECK:   %flatten.tripcount = mul i32 %div5, %div
; CHECK:   br label %for.body.us

for.body.us:                                      ; preds = %for.cond3.for.cond.cleanup8_crit_edge.us, %for.body.lr.ph.split.us
  %i.032.us = phi i32 [ 0, %for.body.lr.ph.split.us ], [ %inc13.us, %for.cond3.for.cond.cleanup8_crit_edge.us ]
  %x.031.us = phi i32 [ 1, %for.body.lr.ph.split.us ], [ %xor.us.lcssa, %for.cond3.for.cond.cleanup8_crit_edge.us ]
  br label %for.body9.us
; CHECK: for.body.us:
; CHECK:   %i.032.us = phi i32 [ 0, %for.body.lr.ph.split.us ], [ %inc13.us, %for.cond3.for.cond.cleanup8_crit_edge.us ]
; CHECK:   %x.031.us = phi i32 [ 1, %for.body.lr.ph.split.us ], [ %xor.us.lcssa, %for.cond3.for.cond.cleanup8_crit_edge.us ]
; CHECK:   br label %for.body9.us

for.body9.us:                                     ; preds = %for.body.us, %for.body9.us
  %j.029.us = phi i32 [ 0, %for.body.us ], [ %inc.us, %for.body9.us ]
  %x.128.us = phi i32 [ %x.031.us, %for.body.us ], [ %xor.us, %for.body9.us ]
  %call.us = tail call i32 @func(i32 1)
  %sub.us = sub nsw i32 %call.us, %x.128.us
  %xor.us = xor i32 %sub.us, %x.128.us
  %inc.us = add nuw nsw i32 %j.029.us, 1
  %cmp6.us = icmp ult i32 %inc.us, %div5
  br i1 %cmp6.us, label %for.body9.us, label %for.cond3.for.cond.cleanup8_crit_edge.us
; CHECK: for.body9.us:
; CHECK:   %j.029.us = phi i32 [ 0, %for.body.us ]
; CHECK:   %x.128.us = phi i32 [ %x.031.us, %for.body.us ]
; CHECK:   %call.us = tail call i32 @func(i32 1)
; CHECK:   %sub.us = sub nsw i32 %call.us, %x.128.us
; CHECK:   %xor.us = xor i32 %sub.us, %x.128.us
; CHECK:   %inc.us = add nuw nsw i32 %j.029.us, 1
; CHECK:   %cmp6.us = icmp ult i32 %inc.us, %div5
; CHECK:   br label %for.cond3.for.cond.cleanup8_crit_edge.us

for.cond3.for.cond.cleanup8_crit_edge.us:         ; preds = %for.body9.us
  %xor.us.lcssa = phi i32 [ %xor.us, %for.body9.us ]
  %inc13.us = add nuw nsw i32 %i.032.us, 1
  %cmp.us = icmp ult i32 %inc13.us, %div
  br i1 %cmp.us, label %for.body.us, label %for.cond.cleanup.loopexit
; CHECK: for.cond3.for.cond.cleanup8_crit_edge.us:
; CHECK:   %xor.us.lcssa = phi i32 [ %xor.us, %for.body9.us ]
; CHECK:   %inc13.us = add nuw nsw i32 %i.032.us, 1
; CHECK:   %cmp.us = icmp ult i32 %inc13.us, %flatten.tripcount
; CHECK:   br i1 %cmp.us, label %for.body.us, label %for.cond.cleanup.loopexit

for.body.lr.ph.split:                             ; preds = %for.body.lr.ph
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.cond3.for.cond.cleanup8_crit_edge.us
  %xor.us.lcssa.lcssa = phi i32 [ %xor.us.lcssa, %for.cond3.for.cond.cleanup8_crit_edge.us ]
  br label %for.cond.cleanup

for.cond.cleanup.loopexit34:                      ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit34, %for.cond.cleanup.loopexit, %entry
  %x.0.lcssa = phi i32 [ 1, %entry ], [ %xor.us.lcssa.lcssa, %for.cond.cleanup.loopexit ], [ 1, %for.cond.cleanup.loopexit34 ]
  ret i32 %x.0.lcssa

for.body:                                         ; preds = %for.body.lr.ph.split, %for.body
  %i.032 = phi i32 [ 0, %for.body.lr.ph.split ], [ %inc13, %for.body ]
  %inc13 = add nuw nsw i32 %i.032, 1
  %cmp = icmp ult i32 %inc13, %div
  br i1 %cmp, label %for.body, label %for.cond.cleanup.loopexit34
}


; CHECK-LABEL: test6
define i32 @test6(i8 zeroext %I, i16 zeroext %J) {
entry:
  %0 = lshr i8 %I, 1
  %div = zext i8 %0 to i32
  %cmp30 = icmp eq i8 %0, 0
  br i1 %cmp30, label %for.cond.cleanup, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  %1 = lshr i16 %J, 1
  %div5 = zext i16 %1 to i32
  %cmp627 = icmp eq i16 %1, 0
  br i1 %cmp627, label %for.body.lr.ph.split, label %for.body.lr.ph.split.us

for.body.lr.ph.split.us:                          ; preds = %for.body.lr.ph
  br label %for.body.us
; CHECK: for.body.lr.ph.split.us:
; CHECK:   %flatten.tripcount = mul i32 %div5, %div
; CHECK:   br label %for.body.us

for.body.us:                                      ; preds = %for.cond3.for.cond.cleanup8_crit_edge.us, %for.body.lr.ph.split.us
  %i.032.us = phi i32 [ 0, %for.body.lr.ph.split.us ], [ %inc13.us, %for.cond3.for.cond.cleanup8_crit_edge.us ]
  %x.031.us = phi i32 [ 1, %for.body.lr.ph.split.us ], [ %xor.us.lcssa, %for.cond3.for.cond.cleanup8_crit_edge.us ]
  %mul.us = mul nuw nsw i32 %i.032.us, %div5
  br label %for.body9.us
; CHECK: for.body.us:
; CHECK:   %i.032.us = phi i32 [ 0, %for.body.lr.ph.split.us ], [ %inc13.us, %for.cond3.for.cond.cleanup8_crit_edge.us ]
; CHECK:   %x.031.us = phi i32 [ 1, %for.body.lr.ph.split.us ], [ %xor.us.lcssa, %for.cond3.for.cond.cleanup8_crit_edge.us ]
; CHECK:   %mul.us = mul nuw nsw i32 %i.032.us, %div5
; CHECK:   br label %for.body9.us

for.body9.us:                                     ; preds = %for.body.us, %for.body9.us
  %j.029.us = phi i32 [ 0, %for.body.us ], [ %inc.us, %for.body9.us ]
  %x.128.us = phi i32 [ %x.031.us, %for.body.us ], [ %xor.us, %for.body9.us ]
  %add.us = add nuw nsw i32 %j.029.us, %mul.us
  %call.us = tail call i32 @func(i32 %add.us)
  %sub.us = sub nsw i32 %call.us, %x.128.us
  %xor.us = xor i32 %sub.us, %x.128.us
  %inc.us = add nuw nsw i32 %j.029.us, 1
  %cmp6.us = icmp ult i32 %inc.us, %div5
  br i1 %cmp6.us, label %for.body9.us, label %for.cond3.for.cond.cleanup8_crit_edge.us
; CHECK: for.body9.us:
; CHECK:   %j.029.us = phi i32 [ 0, %for.body.us ]
; CHECK:   %x.128.us = phi i32 [ %x.031.us, %for.body.us ]
; CHECK:   %add.us = add nuw nsw i32 %j.029.us, %mul.us
; CHECK:   %call.us = tail call i32 @func(i32 %i.032.us)
; CHECK:   %sub.us = sub nsw i32 %call.us, %x.128.us
; CHECK:   %xor.us = xor i32 %sub.us, %x.128.us
; CHECK:   %inc.us = add nuw nsw i32 %j.029.us, 1
; CHECK:   %cmp6.us = icmp ult i32 %inc.us, %div5
; CHECK:   br label %for.cond3.for.cond.cleanup8_crit_edge.us

for.cond3.for.cond.cleanup8_crit_edge.us:         ; preds = %for.body9.us
  %xor.us.lcssa = phi i32 [ %xor.us, %for.body9.us ]
  %inc13.us = add nuw nsw i32 %i.032.us, 1
  %cmp.us = icmp ult i32 %inc13.us, %div
  br i1 %cmp.us, label %for.body.us, label %for.cond.cleanup.loopexit
; CHECK: for.cond3.for.cond.cleanup8_crit_edge.us:
; CHECK:   %xor.us.lcssa = phi i32 [ %xor.us, %for.body9.us ]
; CHECK:   %inc13.us = add nuw nsw i32 %i.032.us, 1
; CHECK:   %cmp.us = icmp ult i32 %inc13.us, %flatten.tripcount
; CHECK:   br i1 %cmp.us, label %for.body.us, label %for.cond.cleanup.loopexit

for.body.lr.ph.split:                             ; preds = %for.body.lr.ph
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.cond3.for.cond.cleanup8_crit_edge.us
  %xor.us.lcssa.lcssa = phi i32 [ %xor.us.lcssa, %for.cond3.for.cond.cleanup8_crit_edge.us ]
  br label %for.cond.cleanup

for.cond.cleanup.loopexit34:                      ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit34, %for.cond.cleanup.loopexit, %entry
  %x.0.lcssa = phi i32 [ 1, %entry ], [ %xor.us.lcssa.lcssa, %for.cond.cleanup.loopexit ], [ 1, %for.cond.cleanup.loopexit34 ]
  ret i32 %x.0.lcssa

for.body:                                         ; preds = %for.body.lr.ph.split, %for.body
  %i.032 = phi i32 [ 0, %for.body.lr.ph.split ], [ %inc13, %for.body ]
  %inc13 = add nuw nsw i32 %i.032, 1
  %cmp = icmp ult i32 %inc13, %div
  br i1 %cmp, label %for.body, label %for.cond.cleanup.loopexit34
}

; CHECK-LABEL: test7
; Various inner phis and conditions which we can still work with
define signext i16 @test7(i32 %I, i32 %J, i32* nocapture readonly %C, i16 signext %limit) {
entry:
  %cmp43 = icmp eq i32 %J, 0
  br i1 %cmp43, label %for.end17, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  %conv = sext i16 %limit to i32
  br label %for.body.us
; CHECK: for.body.lr.ph:
; CHECK:   %conv = sext i16 %limit to i32
; CHECK:   %flatten.tripcount = mul i32 %J, %J
; CHECK:   br label %for.body.us

for.body.us:                                      ; preds = %for.cond1.for.inc15_crit_edge.us, %for.body.lr.ph
  %i.047.us = phi i32 [ 0, %for.body.lr.ph ], [ %inc16.us, %for.cond1.for.inc15_crit_edge.us ]
  %ret.046.us = phi i16 [ 0, %for.body.lr.ph ], [ %ret.2.us.lcssa, %for.cond1.for.inc15_crit_edge.us ]
  %prev.045.us = phi i32 [ 0, %for.body.lr.ph ], [ %.lcssa, %for.cond1.for.inc15_crit_edge.us ]
  %tmp.044.us = phi i32 [ 0, %for.body.lr.ph ], [ %tmp.2.us.lcssa, %for.cond1.for.inc15_crit_edge.us ]
  %mul.us = mul i32 %i.047.us, %J
  br label %for.body3.us
; CHECK: for.body.us:
; CHECK:   %i.047.us = phi i32 [ 0, %for.body.lr.ph ], [ %inc16.us, %for.cond1.for.inc15_crit_edge.us ]
; CHECK:   %ret.046.us = phi i16 [ 0, %for.body.lr.ph ], [ %ret.2.us.lcssa, %for.cond1.for.inc15_crit_edge.us ]
; CHECK:   %prev.045.us = phi i32 [ 0, %for.body.lr.ph ], [ %.lcssa, %for.cond1.for.inc15_crit_edge.us ]
; CHECK:   %tmp.044.us = phi i32 [ 0, %for.body.lr.ph ], [ %tmp.2.us.lcssa, %for.cond1.for.inc15_crit_edge.us ]
; CHECK:   %mul.us = mul i32 %i.047.us, %J
; CHECK:   br label %for.body3.us

for.body3.us:                                     ; preds = %for.body.us, %if.end.us
  %j.040.us = phi i32 [ 0, %for.body.us ], [ %inc.us, %if.end.us ]
  %ret.139.us = phi i16 [ %ret.046.us, %for.body.us ], [ %ret.2.us, %if.end.us ]
  %prev.138.us = phi i32 [ %prev.045.us, %for.body.us ], [ %0, %if.end.us ]
  %tmp.137.us = phi i32 [ %tmp.044.us, %for.body.us ], [ %tmp.2.us, %if.end.us ]
  %add.us = add i32 %j.040.us, %mul.us
  %arrayidx.us = getelementptr inbounds i32, i32* %C, i32 %add.us
  %0 = load i32, i32* %arrayidx.us, align 4
  %add4.us = add nsw i32 %0, %tmp.137.us
  %cmp5.us = icmp sgt i32 %add4.us, %conv
  br i1 %cmp5.us, label %if.then.us, label %if.else.us
; CHECK: for.body3.us:
; CHECK:   %j.040.us = phi i32 [ 0, %for.body.us ]
; CHECK:   %ret.139.us = phi i16 [ %ret.046.us, %for.body.us ]
; CHECK:   %prev.138.us = phi i32 [ %prev.045.us, %for.body.us ]
; CHECK:   %tmp.137.us = phi i32 [ %tmp.044.us, %for.body.us ]
; CHECK:   %add.us = add i32 %j.040.us, %mul.us
; CHECK:   %arrayidx.us = getelementptr inbounds i32, i32* %C, i32 %i.047.us
; CHECK:   %0 = load i32, i32* %arrayidx.us, align 4
; CHECK:   %add4.us = add nsw i32 %0, %tmp.137.us
; CHECK:   %cmp5.us = icmp sgt i32 %add4.us, %conv
; CHECK:   br i1 %cmp5.us, label %if.then.us, label %if.else.us

if.else.us:                                       ; preds = %for.body3.us
  %cmp10.us = icmp sgt i32 %0, %prev.138.us
  %cond.us = zext i1 %cmp10.us to i32
  %conv1235.us = zext i16 %ret.139.us to i32
  %add13.us = add nuw nsw i32 %cond.us, %conv1235.us
  br label %if.end.us
; CHECK: if.else.us:
; CHECK:   %cmp10.us = icmp sgt i32 %0, %prev.138.us
; CHECK:   %cond.us = zext i1 %cmp10.us to i32
; CHECK:   %conv1235.us = zext i16 %ret.139.us to i32
; CHECK:   %add13.us = add nuw nsw i32 %cond.us, %conv1235.us
; CHECK:   br label %if.end.us

if.then.us:                                       ; preds = %for.body3.us
  %conv7.us = sext i16 %ret.139.us to i32
  %add8.us = add nsw i32 %conv7.us, 10
  br label %if.end.us
; CHECK: if.then.us:
; CHECK:   %conv7.us = sext i16 %ret.139.us to i32
; CHECK:   %add8.us = add nsw i32 %conv7.us, 10
; CHECK:   br label %if.end.us

if.end.us:                                        ; preds = %if.then.us, %if.else.us
  %tmp.2.us = phi i32 [ 0, %if.then.us ], [ %add4.us, %if.else.us ]
  %ret.2.in.us = phi i32 [ %add8.us, %if.then.us ], [ %add13.us, %if.else.us ]
  %ret.2.us = trunc i32 %ret.2.in.us to i16
  %inc.us = add nuw i32 %j.040.us, 1
  %exitcond = icmp ne i32 %inc.us, %J
  br i1 %exitcond, label %for.body3.us, label %for.cond1.for.inc15_crit_edge.us
; CHECK: if.end.us:
; CHECK:   %tmp.2.us = phi i32 [ 0, %if.then.us ], [ %add4.us, %if.else.us ]
; CHECK:   %ret.2.in.us = phi i32 [ %add8.us, %if.then.us ], [ %add13.us, %if.else.us ]
; CHECK:   %ret.2.us = trunc i32 %ret.2.in.us to i16
; CHECK:   %inc.us = add nuw i32 %j.040.us, 1
; CHECK:   %exitcond = icmp ne i32 %inc.us, %J
; CHECK:   br label %for.cond1.for.inc15_crit_edge.us

for.cond1.for.inc15_crit_edge.us:                 ; preds = %if.end.us
  %tmp.2.us.lcssa = phi i32 [ %tmp.2.us, %if.end.us ]
  %ret.2.us.lcssa = phi i16 [ %ret.2.us, %if.end.us ]
  %.lcssa = phi i32 [ %0, %if.end.us ]
  %inc16.us = add nuw i32 %i.047.us, 1
  %exitcond49 = icmp ne i32 %inc16.us, %J
  br i1 %exitcond49, label %for.body.us, label %for.end17.loopexit
; CHECK: for.cond1.for.inc15_crit_edge.us:
; CHECK:   %tmp.2.us.lcssa = phi i32 [ %tmp.2.us, %if.end.us ]
; CHECK:   %ret.2.us.lcssa = phi i16 [ %ret.2.us, %if.end.us ]
; CHECK:   %.lcssa = phi i32 [ %0, %if.end.us ]
; CHECK:   %inc16.us = add nuw i32 %i.047.us, 1
; CHECK:   %exitcond49 = icmp ne i32 %inc16.us, %flatten.tripcount
; CHECK:   br i1 %exitcond49, label %for.body.us, label %for.end17.loopexit

for.end17.loopexit:                               ; preds = %for.cond1.for.inc15_crit_edge.us
  %ret.2.us.lcssa.lcssa = phi i16 [ %ret.2.us.lcssa, %for.cond1.for.inc15_crit_edge.us ]
  br label %for.end17

for.end17:                                        ; preds = %for.end17.loopexit, %entry
  %ret.0.lcssa = phi i16 [ 0, %entry ], [ %ret.2.us.lcssa.lcssa, %for.end17.loopexit ]
  ret i16 %ret.0.lcssa
}

; CHECK-LABEL: test8
; Same as test1, but with different continue block order
; (uses icmp eq and loops on false)
define i32 @test8(i32 %val, i16* nocapture %A) {
entry:
  br label %for.body
; CHECK: entry:
; CHECK:   %flatten.tripcount = mul i32 20, 10
; CHECK:   br label %for.body

for.body:                                         ; preds = %entry, %for.inc6
  %i.018 = phi i32 [ 0, %entry ], [ %inc7, %for.inc6 ]
  %mul = mul nuw nsw i32 %i.018, 20
  br label %for.body3
; CHECK: for.body:
; CHECK:   %i.018 = phi i32 [ 0, %entry ], [ %inc7, %for.inc6 ]
; CHECK:   %mul = mul nuw nsw i32 %i.018, 20
; CHECK:   br label %for.body3

for.body3:                                        ; preds = %for.body, %for.body3
  %j.017 = phi i32 [ 0, %for.body ], [ %inc, %for.body3 ]
  %add = add nuw nsw i32 %j.017, %mul
  %arrayidx = getelementptr inbounds i16, i16* %A, i32 %add
  %0 = load i16, i16* %arrayidx, align 2
  %conv16 = zext i16 %0 to i32
  %add4 = add i32 %conv16, %val
  %conv5 = trunc i32 %add4 to i16
  store i16 %conv5, i16* %arrayidx, align 2
  %inc = add nuw nsw i32 %j.017, 1
  %exitcond = icmp eq i32 %inc, 20
  br i1 %exitcond, label %for.inc6, label %for.body3
; CHECK: for.body3:
; CHECK:   %j.017 = phi i32 [ 0, %for.body ]
; CHECK:   %add = add nuw nsw i32 %j.017, %mul
; CHECK:   %arrayidx = getelementptr inbounds i16, i16* %A, i32 %i.018
; CHECK:   %0 = load i16, i16* %arrayidx, align 2
; CHECK:   %conv16 = zext i16 %0 to i32
; CHECK:   %add4 = add i32 %conv16, %val
; CHECK:   %conv5 = trunc i32 %add4 to i16
; CHECK:   store i16 %conv5, i16* %arrayidx, align 2
; CHECK:   %inc = add nuw nsw i32 %j.017, 1
; CHECK:   %exitcond = icmp eq i32 %inc, 20
; CHECK:   br label %for.inc6

for.inc6:                                         ; preds = %for.body3
  %inc7 = add nuw nsw i32 %i.018, 1
  %exitcond19 = icmp eq i32 %inc7, 10
  br i1 %exitcond19, label %for.end8, label %for.body
; CHECK: for.inc6:
; CHECK:   %inc7 = add nuw nsw i32 %i.018, 1
; CHECK:   %exitcond19 = icmp eq i32 %inc7, %flatten.tripcount
; CHECK:   br i1 %exitcond19, label %for.end8, label %for.body

for.end8:                                         ; preds = %for.inc6
  ret i32 10
}


declare i32 @func(i32)

