; RUN: opt < %s -loop-unroll -S -unroll-count=4 | FileCheck %s
; Test phi update after partial unroll.

declare i1 @check() nounwind

; CHECK: @test
; CHECK: if.else:
; CHECK: if.then.loopexit
; CHECK: %sub5.lcssa = phi i32 [ %sub{{.*}}, %if.else{{.*}} ], [ %sub{{.*}}, %if.else{{.*}} ], [ %sub{{.*}}, %if.else{{.*}} ], [ %sub{{.*}}, %if.else{{.*}} ]
; CHECK: if.else.3
define void @test1(i32 %i, i32 %j) nounwind uwtable ssp {
entry:
  %cond1 = call zeroext i1 @check()
  br i1 %cond1, label %if.then, label %if.else.lr.ph

if.else.lr.ph:                                    ; preds = %entry
  br label %if.else

if.else:                                          ; preds = %if.else, %if.else.lr.ph
  %sub = phi i32 [ %i, %if.else.lr.ph ], [ %sub5, %if.else ]
  %sub5 = sub i32 %sub, %j
  %cond2 = call zeroext i1 @check()
  br i1 %cond2, label %if.then, label %if.else

if.then:                                          ; preds = %if.else, %entry
  %i.tr = phi i32 [ %i, %entry ], [ %sub5, %if.else ]
  ret void

}

; PR7318: assertion failure after doing a simple loop unroll
;
; CHECK-LABEL: @test2(
; CHECK: bb1.bb2_crit_edge:
; CHECK: %.lcssa = phi i32 [ %{{[2468]}}, %bb1{{.*}} ], [ %{{[2468]}}, %bb1{{.*}} ], [ %{{[2468]}}, %bb1{{.*}} ], [ %{{[2468]}}, %bb1{{.*}} ]
; CHECK: bb1.3:
define i32 @test2(i32* nocapture %p, i32 %n) nounwind readonly {
entry:
  %0 = icmp sgt i32 %n, 0                         ; <i1> [#uses=1]
  br i1 %0, label %bb.nph, label %bb2

bb.nph:                                           ; preds = %entry
  %tmp = zext i32 %n to i64                       ; <i64> [#uses=1]
  br label %bb

bb:                                               ; preds = %bb.nph, %bb1
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %bb1 ] ; <i64> [#uses=2]
  %s.01 = phi i32 [ 0, %bb.nph ], [ %2, %bb1 ]    ; <i32> [#uses=1]
  %scevgep = getelementptr i32, i32* %p, i64 %indvar   ; <i32*> [#uses=1]
  %1 = load i32, i32* %scevgep, align 1                ; <i32> [#uses=1]
  %2 = add nsw i32 %1, %s.01                      ; <i32> [#uses=2]
  br label %bb1

bb1:                                              ; preds = %bb
  %indvar.next = add i64 %indvar, 1               ; <i64> [#uses=2]
  %exitcond = icmp ne i64 %indvar.next, %tmp      ; <i1> [#uses=1]
  br i1 %exitcond, label %bb, label %bb1.bb2_crit_edge

bb1.bb2_crit_edge:                                ; preds = %bb1
  %.lcssa = phi i32 [ %2, %bb1 ]                  ; <i32> [#uses=1]
  br label %bb2

bb2:                                              ; preds = %bb1.bb2_crit_edge, %entry
  %s.0.lcssa = phi i32 [ %.lcssa, %bb1.bb2_crit_edge ], [ 0, %entry ] ; <i32> [#uses=1]
  ret i32 %s.0.lcssa
}

; Check phi update for loop with an early-exit.
;
; CHECK-LABEL: @test3(
; CHECK: return.loopexit:
; CHECK: %tmp7.i.lcssa = phi i32 [ %tmp7.i{{.*}}, %land.lhs.true{{.*}} ], [ %tmp7.i{{.*}}, %land.lhs.true{{.*}} ], [ %tmp7.i{{.*}}, %land.lhs.true{{.*}} ], [ %tmp7.i{{.*}}, %land.lhs.true{{.*}} ]
; CHECK: exit.3:
define i32 @test3() nounwind uwtable ssp align 2 {
entry:
  %cond1 = call zeroext i1 @check()
  br i1 %cond1, label %return, label %if.end

if.end:                                           ; preds = %entry
  br label %do.body

do.body:                                          ; preds = %do.cond, %if.end
  %cond2 = call zeroext i1 @check()
  br i1 %cond2, label %exit, label %do.cond

exit:                  ; preds = %do.body
  %tmp7.i = load i32, i32* undef, align 8
  br i1 undef, label %do.cond, label %land.lhs.true

land.lhs.true:                                    ; preds = %exit
  br i1 undef, label %return, label %do.cond

do.cond:                                          ; preds = %land.lhs.true, %exit, %do.body
  %cond3 = call zeroext i1 @check()
  br i1 %cond3, label %do.end, label %do.body

do.end:                                           ; preds = %do.cond
  br label %return

return:                                           ; preds = %do.end, %land.lhs.true, %entry
  %retval.0 = phi i32 [ 0, %do.end ], [ 0, %entry ], [ %tmp7.i, %land.lhs.true ]
  ret i32 %retval.0
}
