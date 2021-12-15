; REQUIRES: asserts
; RUN: opt -S -dfa-jump-threading -debug-only=dfa-jump-threading -disable-output %s 2>&1 | FileCheck %s

; This test checks that the analysis identifies all threadable paths in a
; simple CFG. A threadable path includes a list of basic blocks, the exit
; state, and the block that determines the next state.
; < path of BBs that form a cycle > [ state, determinator ]
define i32 @test1(i32 %num) {
; CHECK: < for.body for.inc > [ 1, for.inc ]
; CHECK-NEXT: < for.body case1 for.inc > [ 2, for.inc ]
; CHECK-NEXT: < for.body case2 for.inc > [ 1, for.inc ]
; CHECK-NEXT: < for.body case2 si.unfold.false for.inc > [ 2, for.inc ]
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

; This test checks that the analysis finds threadable paths in a more
; complicated CFG. Here the FSM is represented as a nested loop, with
; fallthrough cases.
define i32 @test2(i32 %init) {
; CHECK: < loop.3 case2 > [ 3, loop.3 ]
; CHECK-NEXT: < loop.3 case2 loop.1.backedge loop.1 loop.2 > [ 1, loop.1 ]
; CHECK-NEXT: < loop.3 case2 loop.1.backedge si.unfold.false loop.1 loop.2 > [ 4, loop.1.backedge ]
; CHECK-NEXT: < loop.3 case3 loop.2.backedge loop.2 > [ 0, loop.2.backedge ]
; CHECK-NEXT: < loop.3 case3 case4 loop.2.backedge loop.2 > [ 3, loop.2.backedge ]
; CHECK-NEXT: < loop.3 case3 case4 loop.1.backedge loop.1 loop.2 > [ 1, loop.1 ]
; CHECK-NEXT: < loop.3 case3 case4 loop.1.backedge si.unfold.false loop.1 loop.2 > [ 2, loop.1.backedge ]
; CHECK-NEXT: < loop.3 case4 loop.2.backedge loop.2 > [ 3, loop.2.backedge ]
; CHECK-NEXT: < loop.3 case4 loop.1.backedge loop.1 loop.2 > [ 1, loop.1 ]
; CHECK-NEXT: < loop.3 case4 loop.1.backedge si.unfold.false loop.1 loop.2 > [ 2, loop.1.backedge ]
entry:
  %cmp = icmp eq i32 %init, 0
  %sel = select i1 %cmp, i32 0, i32 2
  br label %loop.1

loop.1:
  %state.1 = phi i32 [ %sel, %entry ], [ %state.1.be2, %loop.1.backedge ]
  br label %loop.2

loop.2:
  %state.2 = phi i32 [ %state.1, %loop.1 ], [ %state.2.be, %loop.2.backedge ]
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
  br i1 %cmp, label %loop.3, label %loop.1.backedge

case3:
  br i1 %cmp, label %loop.2.backedge, label %case4

case4:
  br i1 %cmp, label %loop.2.backedge, label %loop.1.backedge

loop.1.backedge:
  %state.1.be = phi i32 [ 2, %case4 ], [ 4, %case2 ]
  %state.1.be2 = select i1 %cmp, i32 1, i32 %state.1.be
  br label %loop.1

loop.2.backedge:
  %state.2.be = phi i32 [ 3, %case4 ], [ 0, %case3 ]
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

declare void @baz()

; Do not jump-thread those paths where the determinator basic block does not
; precede the basic block that defines the switch condition.
;
; Otherwise, it is possible that the state defined in the determinator block
; defines the state for the next iteration of the loop, rather than for the
; current one.
define i32 @wrong_bb_order() {
; CHECK-LABEL: DFA Jump threading: wrong_bb_order
; CHECK-NOT: < bb43 bb59 bb3 bb31 bb41 > [ 77, bb43 ]
; CHECK-NOT: < bb43 bb49 bb59 bb3 bb31 bb41 > [ 77, bb43 ]
bb:
  %i = alloca [420 x i8], align 1
  %i2 = getelementptr inbounds [420 x i8], [420 x i8]* %i, i64 0, i64 390
  br label %bb3

bb3:                                              ; preds = %bb59, %bb
  %i4 = phi i8* [ %i2, %bb ], [ %i60, %bb59 ]
  %i5 = phi i8 [ 77, %bb ], [ %i64, %bb59 ]
  %i6 = phi i32 [ 2, %bb ], [ %i63, %bb59 ]
  %i7 = phi i32 [ 26, %bb ], [ %i62, %bb59 ]
  %i8 = phi i32 [ 25, %bb ], [ %i61, %bb59 ]
  %i9 = icmp sgt i32 %i7, 2
  %i10 = select i1 %i9, i32 %i7, i32 2
  %i11 = add i32 %i8, 2
  %i12 = sub i32 %i11, %i10
  %i13 = mul nsw i32 %i12, 3
  %i14 = add nsw i32 %i13, %i6
  %i15 = sext i32 %i14 to i64
  %i16 = getelementptr inbounds i8, i8* %i4, i64 %i15
  %i17 = load i8, i8* %i16, align 1
  %i18 = icmp sgt i8 %i17, 0
  br i1 %i18, label %bb21, label %bb31

bb21:                                             ; preds = %bb3
  br i1 true, label %bb59, label %bb43

bb59:                                             ; preds = %bb49, %bb43, %bb31, %bb21
  %i60 = phi i8* [ %i44, %bb49 ], [ %i44, %bb43 ], [ %i34, %bb31 ], [ %i4, %bb21 ]
  %i61 = phi i32 [ %i45, %bb49 ], [ %i45, %bb43 ], [ %i33, %bb31 ], [ %i8, %bb21 ]
  %i62 = phi i32 [ %i47, %bb49 ], [ %i47, %bb43 ], [ %i32, %bb31 ], [ %i7, %bb21 ]
  %i63 = phi i32 [ %i48, %bb49 ], [ %i48, %bb43 ], [ 2, %bb31 ], [ %i6, %bb21 ]
  %i64 = phi i8 [ %i46, %bb49 ], [ %i46, %bb43 ], [ 77, %bb31 ], [ %i5, %bb21 ]
  %i65 = icmp sgt i32 %i62, 0
  br i1 %i65, label %bb3, label %bb66

bb31:                                             ; preds = %bb3
  %i32 = add nsw i32 %i7, -1
  %i33 = add nsw i32 %i8, -1
  %i34 = getelementptr inbounds i8, i8* %i4, i64 -15
  %i35 = icmp eq i8 %i5, 77
  br i1 %i35, label %bb59, label %bb41

bb41:                                             ; preds = %bb31
  tail call void @baz()
  br label %bb43

bb43:                                             ; preds = %bb41, %bb21
  %i44 = phi i8* [ %i34, %bb41 ], [ %i4, %bb21 ]
  %i45 = phi i32 [ %i33, %bb41 ], [ %i8, %bb21 ]
  %i46 = phi i8 [ 77, %bb41 ], [ %i5, %bb21 ]
  %i47 = phi i32 [ %i32, %bb41 ], [ %i7, %bb21 ]
  %i48 = phi i32 [ 2, %bb41 ], [ %i6, %bb21 ]
  tail call void @baz()
  switch i8 %i5, label %bb59 [
    i8 68, label %bb49
    i8 73, label %bb49
  ]

bb49:                                             ; preds = %bb43, %bb43
  tail call void @baz()
  br label %bb59

bb66:                                             ; preds = %bb59
  ret i32 0
}
