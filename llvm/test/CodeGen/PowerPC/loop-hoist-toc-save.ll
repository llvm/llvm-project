; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -hoist-const-stores -ppc-stack-ptr-caller-preserved < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -hoist-const-stores -ppc-stack-ptr-caller-preserved < %s | FileCheck %s -check-prefix=CHECKBE

; Test hoist out of single loop
define signext i32 @test1(i32 signext %lim, i32 (i32)* nocapture %Func) {
entry:
; CHECK-LABEL: test1
; CHECK: for.body.preheader
; CHECK: std 2, 24(1)
; CHECK: for.body
; CHECK-NOT: std 2, 24(1)
; CHECKBE-LABEL: test1
; CHECKBE: for.body.preheader
; CHECKBE: std 2, 40(1)
; CHECKBE: for.body
; CHECKBE-NOT: std 2, 40(1)

  %cmp6 = icmp sgt i32 %lim, 0
  br i1 %cmp6, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %Sum.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %Sum.0.lcssa

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.08 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %Sum.07 = phi i32 [ %add, %for.body ], [ 0, %for.body.preheader ]
  %call = tail call signext i32 %Func(i32 signext %i.08)
  %add = add nsw i32 %call, %Sum.07
  %inc = add nuw nsw i32 %i.08, 1
  %exitcond = icmp eq i32 %inc, %lim
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; Test hoist of nested loop goes to outter loop preheader
define signext i32 @test2(i32 signext %lim, i32 (i32)* nocapture %Func) {
entry:
; CHECK-LABEL: test2
; CHECK: for.body4.lr.ph.preheader
; CHECK: std 2, 24(1)
; CHECK: for.body4.lr.ph
; CHECK-NOT: std 2, 24(1)
; CHECKBE-LABEL: test2
; CHECKBE: for.body4.lr.ph.preheader
; CHECKBE: std 2, 40(1)
; CHECKBE: for.body4.lr.ph
; CHECKBE-NOT: std 2, 40(1)

  %cmp20 = icmp sgt i32 %lim, 0
  br i1 %cmp20, label %for.body4.lr.ph.preheader, label %for.cond.cleanup

for.body4.lr.ph.preheader:                        ; preds = %entry
  br label %for.body4.lr.ph

for.cond.cleanup:                                 ; preds = %for.cond.cleanup3, %entry
  %Sum.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.cond.cleanup3 ]
  ret i32 %Sum.0.lcssa

for.body4.lr.ph:                                  ; preds = %for.body4.lr.ph.preheader, %for.cond.cleanup3
  %j.022 = phi i32 [ %inc6, %for.cond.cleanup3 ], [ 0, %for.body4.lr.ph.preheader ]
  %Sum.021 = phi i32 [ %add, %for.cond.cleanup3 ], [ 0, %for.body4.lr.ph.preheader ]
  br label %for.body4

for.cond.cleanup3:                                ; preds = %for.body4
  %inc6 = add nuw nsw i32 %j.022, 1
  %exitcond24 = icmp eq i32 %inc6, %lim
  br i1 %exitcond24, label %for.cond.cleanup, label %for.body4.lr.ph

for.body4:                                        ; preds = %for.body4, %for.body4.lr.ph
  %i.019 = phi i32 [ %j.022, %for.body4.lr.ph ], [ %inc, %for.body4 ]
  %Sum.118 = phi i32 [ %Sum.021, %for.body4.lr.ph ], [ %add, %for.body4 ]
  %call = tail call signext i32 %Func(i32 signext %i.019)
  %add = add nsw i32 %call, %Sum.118
  %inc = add nuw nsw i32 %i.019, 1
  %exitcond = icmp eq i32 %inc, %lim
  br i1 %exitcond, label %for.cond.cleanup3, label %for.body4
}

; Test hoist out of if statement with low branch probability
; FIXME: we shouldn't hoist in such cases as it could increase the number
; of stores after hoisting.
define signext i32 @test3(i32 signext %lim, i32 (i32)* nocapture %Func) {
entry:
; CHECK-LABEL: test3
; CHECK: %for.body.lr.ph
; CHECK: std 2, 24(1)
; CHECK: %for.body
; CHECK-NOT: std 2, 24(1)
; CHECKBE-LABEL: test3
; CHECKBE: %for.body.lr.ph
; CHECKBE: std 2, 40(1)
; CHECKBE: %for.body
; CHECKBE-NOT: std 2, 40(1)

  %cmp13 = icmp sgt i32 %lim, 0
  br i1 %cmp13, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry
  %sub = add nsw i32 %lim, -1
  br label %for.body

for.cond.cleanup:                                 ; preds = %if.end, %entry
  %Sum.0.lcssa = phi i32 [ 0, %entry ], [ %add3, %if.end ]
  ret i32 %Sum.0.lcssa

for.body:                                         ; preds = %if.end, %for.body.lr.ph
  %i.015 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %if.end ]
  %Sum.014 = phi i32 [ 0, %for.body.lr.ph ], [ %add3, %if.end ]
  %cmp1 = icmp eq i32 %i.015, %sub
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %call = tail call signext i32 %Func(i32 signext %sub)
  %add = add nsw i32 %call, %Sum.014
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  %Sum.1 = phi i32 [ %add, %if.then ], [ %Sum.014, %for.body ]
  %call2 = tail call signext i32 @func(i32 signext %i.015)
  %add3 = add nsw i32 %call2, %Sum.1
  %inc = add nuw nsw i32 %i.015, 1
  %exitcond = icmp eq i32 %inc, %lim
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

declare signext i32 @func(i32 signext)
