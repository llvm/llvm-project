; RUN: opt -dfa-jump-threading -dfa-cost-threshold=25 -pass-remarks-missed='dfa-jump-threading' -pass-remarks-output=%t -disable-output %s
; RUN: FileCheck --input-file %t --check-prefix=REMARK %s
; RUN: opt -S -dfa-jump-threading %s | FileCheck %s

; This negative test case checks that the optimization doesn't trigger
; when the code size cost is too high.
define i32 @negative1(i32 %num) {
; REMARK: NotProfitable
; REMARK-NEXT: negative1
entry:
  br label %for.body

for.body:
  %count = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %state = phi i32 [ 1, %entry ], [ %state.next, %for.inc ]
  switch i32 %state, label %for.inc [
  i32 1, label %case1
  i32 2, label %case2
  ]

case1:
  br label %for.inc

case2:
  %cmp = icmp eq i32 %count, 50
  %sel = select i1 %cmp, i32 1, i32 2
  br label %for.inc

for.inc:
  %state.next = phi i32 [ %sel, %case2 ], [ 1, %for.body ], [ 2, %case1 ]
  %add1 = add i32 %num, %num
  %add2 = add i32 %add1, %add1
  %add3 = add i32 %add2, %add2
  %add4 = add i32 %add3, %add3
  %add5 = add i32 %add4, %add4
  %add6 = add i32 %add5, %add5
  %add7 = add i32 %add6, %add6
  %add8 = add i32 %add7, %add7
  %add9 = add i32 %add8, %add8
  %add10 = add i32 %add9, %add9
  %add11 = add i32 %add10, %add10
  %add12 = add i32 %add11, %add11
  %add13 = add i32 %add12, %add12
  %add14 = add i32 %add13, %add13
  %add15 = add i32 %add14, %add14
  %add16 = add i32 %add15, %add15
  %add17 = add i32 %add16, %add16
  %add18 = add i32 %add17, %add17
  %add19 = add i32 %add18, %add18
  %add20 = add i32 %add19, %add19
  %add21 = add i32 %add20, %add20
  %add22 = add i32 %add21, %add21
  %inc = add nsw i32 %count, 1
  %cmp.exit = icmp slt i32 %inc, %num
  br i1 %cmp.exit, label %for.body, label %for.end

for.end:
  ret i32 %add22
}

declare void @func()

define i32 @negative2(i32 %num) {
; REMARK: NonDuplicatableInst
; REMARK-NEXT: negative2
entry:
  br label %for.body

for.body:
  %count = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %state = phi i32 [ 1, %entry ], [ %state.next, %for.inc ]
  switch i32 %state, label %for.inc [
  i32 1, label %case1
  i32 2, label %case2
  ]

case1:
  br label %for.inc

case2:
  %cmp = icmp eq i32 %count, 50
  %sel = select i1 %cmp, i32 1, i32 2
  br label %for.inc

for.inc:
  %state.next = phi i32 [ %sel, %case2 ], [ 1, %for.body ], [ 2, %case1 ]
  call void @func() noduplicate
  %inc = add nsw i32 %count, 1
  %cmp.exit = icmp slt i32 %inc, %num
  br i1 %cmp.exit, label %for.body, label %for.end

for.end:
  ret i32 0
}

define i32 @negative3(i32 %num) {
; REMARK: ConvergentInst
; REMARK-NEXT: negative3
entry:
  br label %for.body

for.body:
  %count = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %state = phi i32 [ 1, %entry ], [ %state.next, %for.inc ]
  switch i32 %state, label %for.inc [
  i32 1, label %case1
  i32 2, label %case2
  ]

case1:
  br label %for.inc

case2:
  %cmp = icmp eq i32 %count, 50
  %sel = select i1 %cmp, i32 1, i32 2
  br label %for.inc

for.inc:
  %state.next = phi i32 [ %sel, %case2 ], [ 1, %for.body ], [ 2, %case1 ]
  call void @func() convergent
  %inc = add nsw i32 %count, 1
  %cmp.exit = icmp slt i32 %inc, %num
  br i1 %cmp.exit, label %for.body, label %for.end

for.end:
  ret i32 0
}

define i32 @negative4(i32 %num) {
; REMARK: SwitchNotPredictable
; REMARK-NEXT: negative4
entry:
  br label %for.body

for.body:
  %count = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %state = phi i32 [ 1, %entry ], [ %state.next, %for.inc ]
  switch i32 %state, label %for.inc [
  i32 1, label %case1
  i32 2, label %case2
  ]

case1:
  br label %for.inc

case2:
  %cmp = icmp eq i32 %count, 50
  %sel = select i1 %cmp, i32 1, i32 2
  br label %for.inc

for.inc:
  ; the switch variable is not predictable since the exit value for %case1
  ; is defined through a non-instruction (function argument).
  %state.next = phi i32 [ %sel, %case2 ], [ 1, %for.body ], [ %num, %case1 ]
  %inc = add nsw i32 %count, 1
  %cmp.exit = icmp slt i32 %inc, %num
  br i1 %cmp.exit, label %for.body, label %for.end

for.end:
  ret i32 0
}

; Do not optimize if marked minsize.
define i32 @negative5(i32 %num) minsize {
; CHECK-LABEL: @negative5(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[FOR_BODY:%.*]]
; CHECK:       for.body:
; CHECK-NEXT:    [[COUNT:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[INC:%.*]], [[FOR_INC:%.*]] ]
; CHECK-NEXT:    [[STATE:%.*]] = phi i32 [ 1, [[ENTRY]] ], [ [[STATE_NEXT:%.*]], [[FOR_INC]] ]
; CHECK-NEXT:    switch i32 [[STATE]], label [[FOR_INC]] [
; CHECK-NEXT:    i32 1, label [[CASE1:%.*]]
; CHECK-NEXT:    i32 2, label [[CASE2:%.*]]
; CHECK-NEXT:    ]
; CHECK:       case1:
; CHECK-NEXT:    br label [[FOR_INC]]
; CHECK:       case2:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[COUNT]], 50
; CHECK-NEXT:    [[SEL:%.*]] = select i1 [[CMP]], i32 1, i32 2
; CHECK-NEXT:    br label [[FOR_INC]]
; CHECK:       for.inc:
; CHECK-NEXT:    [[STATE_NEXT]] = phi i32 [ [[SEL]], [[CASE2]] ], [ 1, [[FOR_BODY]] ], [ 2, [[CASE1]] ]
; CHECK-NEXT:    [[INC]] = add nsw i32 [[COUNT]], 1
; CHECK-NEXT:    [[CMP_EXIT:%.*]] = icmp slt i32 [[INC]], [[NUM:%.*]]
; CHECK-NEXT:    br i1 [[CMP_EXIT]], label [[FOR_BODY]], label [[FOR_END:%.*]]
; CHECK:       for.end:
; CHECK-NEXT:    ret i32 0
;
entry:
  br label %for.body

for.body:
  %count = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %state = phi i32 [ 1, %entry ], [ %state.next, %for.inc ]
  switch i32 %state, label %for.inc [
  i32 1, label %case1
  i32 2, label %case2
  ]

case1:
  br label %for.inc

case2:
  %cmp = icmp eq i32 %count, 50
  %sel = select i1 %cmp, i32 1, i32 2
  br label %for.inc

for.inc:
  %state.next = phi i32 [ %sel, %case2 ], [ 1, %for.body ], [ 2, %case1 ]
  %inc = add nsw i32 %count, 1
  %cmp.exit = icmp slt i32 %inc, %num
  br i1 %cmp.exit, label %for.body, label %for.end

for.end:
  ret i32 0
}

declare i32 @arbitrary_function()

; Don't confuse %state.2 for the initial switch value.
define i32 @negative6(i32 %init) {
; REMARK: SwitchNotPredictable
; REMARK-NEXT: negative6
entry:
  %cmp = icmp eq i32 %init, 0
  br label %loop.2

loop.2:
  %state.2 = call i32 @arbitrary_function()
  br label %loop.3

loop.3:
  %state = phi i32 [ %state.2, %loop.2 ], [ 3, %case2 ]
  switch i32 %state, label %infloop.i [
    i32 2, label %case2
    i32 3, label %case3
    i32 4, label %case4
    i32 0, label %case0
    i32 1, label %case1
  ]

case2:
  br label %loop.3

case3:
  br i1 %cmp, label %loop.2.backedge, label %case4

case4:
  br label %loop.2.backedge

loop.2.backedge:
  br label %loop.2

case0:
  br label %exit

case1:
  br label %exit

infloop.i:
  br label %infloop.i

exit:
  ret i32 0
}
