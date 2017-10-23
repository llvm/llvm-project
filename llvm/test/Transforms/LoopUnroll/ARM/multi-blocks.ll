; RUN: opt -mtriple=thumbv8m.main -mcpu=cortex-m33 -loop-unroll -S < %s -o - | FileCheck %s
; RUN: opt -mtriple=thumbv7em -mcpu=cortex-m7 -loop-unroll -S < %s -o - | FileCheck %s

;CHECK-LABEL: test_three_blocks
;CHECK: for.body.epil:
;CHECK: if.then.epil:
;CHECK: for.inc.epil:
;CHECK: for.body:
;CHECK: if.then:
;CHECK: for.inc:
;CHECK: for.body.epil.1:
;CHECK: if.then.epil.1:
;CHECK: for.inc.epil.1:
;CHECK: for.body.epil.2:
;CHECK: if.then.epil.2:
;CHECK: for.inc.epil.2:
;CHECK: if.then.1:
;CHECK: for.inc.1:
;CHECK: if.then.2:
;CHECK: for.inc.2:
;CHECK: if.then.3:
;CHECK: for.inc.3:
define void @test_three_blocks(i32* nocapture %Output,
                               i32* nocapture readonly %Condition,
                               i32* nocapture readonly %Input,
                               i32 %MaxJ) {
entry:
  %cmp8 = icmp eq i32 %MaxJ, 0
  br i1 %cmp8, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.inc, %entry
  %temp.0.lcssa = phi i32 [ 0, %entry ], [ %temp.1, %for.inc ]
  store i32 %temp.0.lcssa, i32* %Output, align 4
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.inc
  %j.010 = phi i32 [ %inc, %for.inc ], [ 0, %for.body.preheader ]
  %temp.09 = phi i32 [ %temp.1, %for.inc ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %Condition, i32 %j.010
  %0 = load i32, i32* %arrayidx, align 4
  %tobool = icmp eq i32 %0, 0
  br i1 %tobool, label %for.inc, label %if.then

if.then:                                          ; preds = %for.body
  %arrayidx1 = getelementptr inbounds i32, i32* %Input, i32 %j.010
  %1 = load i32, i32* %arrayidx1, align 4
  %add = add i32 %1, %temp.09
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %temp.1 = phi i32 [ %add, %if.then ], [ %temp.09, %for.body ]
  %inc = add nuw i32 %j.010, 1
  %exitcond = icmp eq i32 %inc, %MaxJ
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

;CHECK-LABEL: test_two_exits
;CHECK: for.body:
;CHECK: if.end:
;CHECK: cleanup.loopexit:
;CHECK: cleanup:
;CHECK: for.body.1:
;CHECK: if.end.1:
;CHECK: for.body.2:
;CHECK: if.end.2:
;CHECK: for.body.3:
;CHECK: if.end.3:
define void @test_two_exits(i32* nocapture %Output,
                            i32* nocapture readonly %Condition,
                            i32* nocapture readonly %Input,
                            i32 %MaxJ) {
entry:
  %cmp14 = icmp eq i32 %MaxJ, 0
  br i1 %cmp14, label %cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %if.end
  %j.016 = phi i32 [ %inc, %if.end ], [ 0, %for.body.preheader ]
  %temp.015 = phi i32 [ %temp.0.add, %if.end ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %Input, i32 %j.016
  %0 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp ugt i32 %0, 65535
  br i1 %cmp1, label %cleanup, label %if.end

if.end:                                           ; preds = %for.body
  %arrayidx2 = getelementptr inbounds i32, i32* %Condition, i32 %j.016
  %1 = load i32, i32* %arrayidx2, align 4
  %tobool = icmp eq i32 %1, 0
  %add = select i1 %tobool, i32 0, i32 %0
  %temp.0.add = add i32 %add, %temp.015
  %inc = add nuw i32 %j.016, 1
  %cmp = icmp ult i32 %inc, %MaxJ
  br i1 %cmp, label %for.body, label %cleanup

cleanup:                                          ; preds = %if.end, %for.body, %entry
  %temp.0.lcssa = phi i32 [ 0, %entry ], [ %temp.015, %for.body ], [ %temp.0.add, %if.end ]
  store i32 %temp.0.lcssa, i32* %Output, align 4
  ret void
}

;CHECK-LABEL: test_three_exits
;CHECK-NOT: for.body.epil
;CHECK-NOT: if.end.epil
;CHECK-LABEL: for.body
;CHECK-LABEL: if.end
;CHECK-LABEL: if.end5
define void @test_three_exits(i32* nocapture %Output,
                              i32* nocapture readonly %Condition,
                              i32* nocapture readonly %Input,
                              i32 %MaxJ) {
entry:
  %cmp20 = icmp eq i32 %MaxJ, 0
  br i1 %cmp20, label %cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %if.end5
  %j.022 = phi i32 [ %inc, %if.end5 ], [ 0, %for.body.preheader ]
  %temp.021 = phi i32 [ %temp.0.add, %if.end5 ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %Condition, i32 %j.022
  %0 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp ugt i32 %0, 65535
  br i1 %cmp1, label %cleanup, label %if.end

if.end:                                           ; preds = %for.body
  %arrayidx2 = getelementptr inbounds i32, i32* %Input, i32 %j.022
  %1 = load i32, i32* %arrayidx2, align 4
  %cmp3 = icmp ugt i32 %1, 65535
  br i1 %cmp3, label %cleanup, label %if.end5

if.end5:                                          ; preds = %if.end
  %tobool = icmp eq i32 %0, 0
  %add = select i1 %tobool, i32 0, i32 %1
  %temp.0.add = add i32 %add, %temp.021
  %inc = add nuw i32 %j.022, 1
  %cmp = icmp ult i32 %inc, %MaxJ
  br i1 %cmp, label %for.body, label %cleanup

cleanup:                                          ; preds = %if.end5, %for.body, %if.end, %entry
  %temp.0.lcssa = phi i32 [ 0, %entry ], [ %temp.021, %if.end ], [ %temp.021, %for.body ], [ %temp.0.add, %if.end5 ]
  store i32 %temp.0.lcssa, i32* %Output, align 4
  ret void
}

;CHECK-LABEL: test_four_blocks
;CHECK: for.body.epil:
;CHECK: if.else.epil:
;CHECK: if.then.epil:
;CHECK: for.cond.cleanup:
;CHECK: for.body:
;CHECK: if.then:
;CHECK: for.inc:
;CHECK: for.body.epil.1:
;CHECK: if.else.epil.1:
;CHECK: if.then.epil.1:
;CHECK: for.inc.epil.1:
;CHECK: for.body.epil.2:
;CHECK: if.else.epil.2:
;CHECK: if.then.epil.2:
;CHECK: for.inc.epil.2:
;CHECK: if.else.1:
;CHECK: if.then.1:
;CHECK: for.inc.1:
;CHECK: if.else.2:
;CHECK: if.then.2:
;CHECK: for.inc.2:
;CHECK: if.else.3:
;CHECK: if.then.3:
;CHECK: for.inc.3:
define void @test_four_blocks(i32* nocapture %Output,
                              i32* nocapture readonly %Condition,
                              i32* nocapture readonly %Input,
                              i32 %MaxJ) {
entry:
  %cmp25 = icmp ugt i32 %MaxJ, 1
  br i1 %cmp25, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry
  %.pre = load i32, i32* %Input, align 4
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.inc, %entry
  %temp.0.lcssa = phi i32 [ 0, %entry ], [ %temp.1, %for.inc ]
  store i32 %temp.0.lcssa, i32* %Output, align 4
  ret void

for.body:                                         ; preds = %for.inc, %for.body.lr.ph
  %0 = phi i32 [ %.pre, %for.body.lr.ph ], [ %2, %for.inc ]
  %j.027 = phi i32 [ 1, %for.body.lr.ph ], [ %inc, %for.inc ]
  %temp.026 = phi i32 [ 0, %for.body.lr.ph ], [ %temp.1, %for.inc ]
  %arrayidx = getelementptr inbounds i32, i32* %Condition, i32 %j.027
  %1 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp ugt i32 %1, 65535
  %arrayidx2 = getelementptr inbounds i32, i32* %Input, i32 %j.027
  %2 = load i32, i32* %arrayidx2, align 4
  %cmp4 = icmp ugt i32 %2, %0
  br i1 %cmp1, label %if.then, label %if.else

if.then:                                          ; preds = %for.body
  %cond = zext i1 %cmp4 to i32
  %add = add i32 %temp.026, %cond
  br label %for.inc

if.else:                                          ; preds = %for.body
  %not.cmp4 = xor i1 %cmp4, true
  %sub = sext i1 %not.cmp4 to i32
  %sub10.sink = add i32 %j.027, %sub
  %arrayidx11 = getelementptr inbounds i32, i32* %Input, i32 %sub10.sink
  %3 = load i32, i32* %arrayidx11, align 4
  %sub13 = sub i32 %temp.026, %3
  br label %for.inc

for.inc:                                          ; preds = %if.then, %if.else
  %temp.1 = phi i32 [ %add, %if.then ], [ %sub13, %if.else ]
  %inc = add nuw i32 %j.027, 1
  %exitcond = icmp eq i32 %inc, %MaxJ
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

;CHECK-LABEL: test_five_blocks
;CHECK-NOT: for.body.epil:
;CHECK: for.body:
;CHECK: if.end:
;CHECK: if.else:
;CHECK: for.inc:
;CHECK-NOT: for.inc.1:
define void @test_five_blocks(i32* nocapture %Output,
                              i32* nocapture readonly %Condition,
                              i32* nocapture readonly %Input,
                              i32 %MaxJ) {
entry:
  %cmp24 = icmp ugt i32 %MaxJ, 1
  br i1 %cmp24, label %for.body.preheader, label %cleanup

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.inc
  %j.026 = phi i32 [ %inc, %for.inc ], [ 1, %for.body.preheader ]
  %temp.025 = phi i32 [ %temp.1, %for.inc ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %Input, i32 %j.026
  %0 = load i32, i32* %arrayidx, align 4
  %add = add i32 %0, %temp.025
  %cmp1 = icmp ugt i32 %add, 16777215
  br i1 %cmp1, label %cleanup, label %if.end

if.end:                                           ; preds = %for.body
  %arrayidx2 = getelementptr inbounds i32, i32* %Condition, i32 %j.026
  %1 = load i32, i32* %arrayidx2, align 4
  %cmp3 = icmp ugt i32 %1, 65535
  br i1 %cmp3, label %if.then4, label %if.else

if.then4:                                         ; preds = %if.end
  %sub = add i32 %j.026, -1
  %arrayidx6 = getelementptr inbounds i32, i32* %Input, i32 %sub
  %2 = load i32, i32* %arrayidx6, align 4
  %cmp7 = icmp ugt i32 %0, %2
  %cond = zext i1 %cmp7 to i32
  %add8 = add i32 %add, %cond
  br label %for.inc

if.else:                                          ; preds = %if.end
  %and = and i32 %add, %0
  br label %for.inc

for.inc:                                          ; preds = %if.then4, %if.else
  %temp.1 = phi i32 [ %add8, %if.then4 ], [ %and, %if.else ]
  %inc = add nuw i32 %j.026, 1
  %cmp = icmp ult i32 %inc, %MaxJ
  br i1 %cmp, label %for.body, label %cleanup

cleanup:                                          ; preds = %for.inc, %for.body, %entry
  %temp.2 = phi i32 [ 0, %entry ], [ %add, %for.body ], [ %temp.1, %for.inc ]
  store i32 %temp.2, i32* %Output, align 4
  ret void
}

;CHECK-LABEL: iterate_inc
;CHECK: while.body:
;CHECK: while.end:
;CHECK: while.body.1:
;CHECK: while.body.2:
;CHECK: while.body.3:
%struct.Node = type { %struct.Node*, i32 }
define void @iterate_inc(%struct.Node* %n, i32 %limit) {
entry:
  %tobool5 = icmp eq %struct.Node* %n, null
  br i1 %tobool5, label %while.end, label %land.rhs.preheader

land.rhs.preheader:                               ; preds = %entry
  br label %land.rhs

land.rhs:                                         ; preds = %land.rhs.preheader, %while.body
  %list.addr.06 = phi %struct.Node* [ %2, %while.body ], [ %n, %land.rhs.preheader ]
  %val = getelementptr inbounds %struct.Node, %struct.Node* %list.addr.06, i32 0, i32 1
  %0 = load i32, i32* %val, align 4
  %cmp = icmp slt i32 %0, %limit
  br i1 %cmp, label %while.body, label %while.end

while.body:                                       ; preds = %land.rhs
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* %val, align 4
  %1 = bitcast %struct.Node* %list.addr.06 to %struct.Node**
  %2 = load %struct.Node*, %struct.Node** %1, align 4
  %tobool = icmp eq %struct.Node* %2, null
  br i1 %tobool, label %while.end, label %land.rhs

while.end:                                        ; preds = %land.rhs, %while.body, %entry
  ret void
}
