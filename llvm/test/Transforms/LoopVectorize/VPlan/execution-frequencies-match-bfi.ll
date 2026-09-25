; RUN: opt -passes='print<block-freq>' -disable-output %s 2>&1 \
; RUN:   | FileCheck --check-prefix=BFI %s
; RUN: opt -passes=loop-vectorize -force-vector-width=2 -force-vector-interleave=1 \
; RUN:     -vplan-print-after=recordExecutionFrequencies -disable-output %s 2>&1 \
; RUN:   | FileCheck --check-prefix=VPLAN %s

; Check that the execution frequencies VPlan records on the recipes of a block
; match the block frequencies BlockFrequencyInfo computes for the
; corresponding block of the original scalar loop.

define void @single_pred(ptr noalias %a, ptr noalias %b, ptr noalias %idx) {
; Execution frequency of each block of the loop
;
;   %loop      1000/1000 =   1
;   %if.then    250/1000 = 1/4
;   %latch     1000/1000 =   1
;
; BFI-LABEL: block-frequency-info: single_pred
; BFI-NEXT:   - entry: float = 1.0,
; BFI-NEXT:   - loop: float = 1000.0,
; BFI-NEXT:   - if.then: float = 250.0,
; BFI-NEXT:   - latch: float = 1000.0,
; BFI-NEXT:   - exit: float = 1.0,
;
; VPLAN-LABEL: VPlan for loop in 'single_pred'
; VPLAN:       loop:
; VPLAN-NEXT:    EMIT-SCALAR ir<%iv> = phi [ ir<0>, vector.ph ], [ ir<%iv.next>, latch ]{{$}}
; VPLAN-NEXT:    EMIT ir<%gep.idx> = getelementptr inbounds ir<%idx>, ir<%iv>{{$}}
; VPLAN-NEXT:    EMIT-SCALAR ir<%i> = load ir<%gep.idx>{{$}}
; VPLAN-NEXT:    EMIT ir<%gep.b> = getelementptr inbounds ir<%b>, ir<%iv>{{$}}
; VPLAN-NEXT:    EMIT store ir<%i>, ir<%gep.b>{{$}}
; VPLAN-NEXT:    EMIT ir<%c.0> = icmp sgt ir<%i>, ir<0>{{$}}
; VPLAN-NEXT:    EMIT branch-on-cond ir<%c.0> (!prof {1, 3}){{$}}
; VPLAN-NEXT:  Successor(s): if.then, latch
; VPLAN-EMPTY:
; VPLAN-NEXT:  if.then:
; VPLAN-NEXT:    EMIT ir<%gep.a> = getelementptr inbounds ir<%a>, ir<%iv> (!vplan.execution.frequency 2305843009213693952 (25%))
; VPLAN-NEXT:    EMIT store ir<%i>, ir<%gep.a> (!vplan.execution.frequency 2305843009213693952 (25%))
; VPLAN-NEXT:  Successor(s): latch
; VPLAN-EMPTY:
; VPLAN-NEXT:  latch:
; VPLAN-NEXT:    EMIT ir<%iv.next> = add ir<%iv>, ir<1>{{$}}
; VPLAN-NEXT:    EMIT ir<%ec> = icmp eq ir<%iv.next>, ir<1024>{{$}}
; VPLAN-NEXT:    EMIT branch-on-cond ir<%ec> (!prof {1, 999}){{$}}
; VPLAN-NEXT:  Successor(s): middle.block, loop
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %gep.idx = getelementptr inbounds i32, ptr %idx, i64 %iv
  %i = load i32, ptr %gep.idx, align 4
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %iv
  store i32 %i, ptr %gep.b, align 4
  %c.0 = icmp sgt i32 %i, 0
  br i1 %c.0, label %if.then, label %latch, !prof !0

if.then:
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  store i32 %i, ptr %gep.a, align 4
  br label %latch

latch:
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 1024
  br i1 %ec, label %exit, label %loop, !prof !3

exit:
  ret void
}

define void @single_pred_zero_weight(ptr noalias %a, ptr noalias %b, ptr noalias %idx) {
; %if.then is only entered via an edge with zero branch weight. BFI treats the
; edge as cold, not as never taken.
;
;   %loop       1 =        1
;   %if.then 2^-31 ~ 4.66e-10
;   %latch      1 =        1
;
; TODO: VPlan currently records a frequency of 0 for %if.then.
;
; BFI-LABEL: block-frequency-info: single_pred_zero_weight
; BFI-NEXT:   - entry: float = 1.0,
; BFI-NEXT:   - loop: float = 1000.0,
; BFI-NEXT:   - if.then: float = 0.00000046566,
; BFI-NEXT:   - latch: float = 1000.0,
; BFI-NEXT:   - exit: float = 1.0,
;
; VPLAN-LABEL: VPlan for loop in 'single_pred_zero_weight'
; VPLAN:       loop:
; VPLAN-NEXT:    EMIT-SCALAR ir<%iv> = phi [ ir<0>, vector.ph ], [ ir<%iv.next>, latch ]{{$}}
; VPLAN-NEXT:    EMIT ir<%gep.idx> = getelementptr inbounds ir<%idx>, ir<%iv>{{$}}
; VPLAN-NEXT:    EMIT-SCALAR ir<%i> = load ir<%gep.idx>{{$}}
; VPLAN-NEXT:    EMIT ir<%gep.b> = getelementptr inbounds ir<%b>, ir<%iv>{{$}}
; VPLAN-NEXT:    EMIT store ir<%i>, ir<%gep.b>{{$}}
; VPLAN-NEXT:    EMIT ir<%c.0> = icmp sgt ir<%i>, ir<0>{{$}}
; VPLAN-NEXT:    EMIT branch-on-cond ir<%c.0> (!prof {0, 1000}){{$}}
; VPLAN-NEXT:  Successor(s): if.then, latch
; VPLAN-EMPTY:
; VPLAN-NEXT:  if.then:
; VPLAN-NEXT:    EMIT ir<%gep.a> = getelementptr inbounds ir<%a>, ir<%iv>{{$}}
; VPLAN-NEXT:    EMIT store ir<%i>, ir<%gep.a>{{$}}
; VPLAN-NEXT:  Successor(s): latch
; VPLAN-EMPTY:
; VPLAN-NEXT:  latch:
; VPLAN-NEXT:    EMIT ir<%iv.next> = add ir<%iv>, ir<1>{{$}}
; VPLAN-NEXT:    EMIT ir<%ec> = icmp eq ir<%iv.next>, ir<1024>{{$}}
; VPLAN-NEXT:    EMIT branch-on-cond ir<%ec> (!prof {1, 999}){{$}}
; VPLAN-NEXT:  Successor(s): middle.block, loop
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %gep.idx = getelementptr inbounds i32, ptr %idx, i64 %iv
  %i = load i32, ptr %gep.idx, align 4
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %iv
  store i32 %i, ptr %gep.b, align 4
  %c.0 = icmp sgt i32 %i, 0
  br i1 %c.0, label %if.then, label %latch, !prof !13

if.then:
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  store i32 %i, ptr %gep.a, align 4
  br label %latch

latch:
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 1024
  br i1 %ec, label %exit, label %loop, !prof !3

exit:
  ret void
}

define void @single_pred_zero_weight_sibling(ptr noalias %a, ptr noalias %b, ptr noalias %idx) {
; The edge skipping %if.then has zero branch weight. BFI treats it as cold, not
; as never taken.
;
;   %loop            1 =          1
;   %if.then 1 - 2^-31 ~ 1 - 4.66e-10
;   %latch           1 =          1
;
; TODO: VPlan currently records %if.then as always executing.
;
; BFI-LABEL: block-frequency-info: single_pred_zero_weight_sibling
; BFI-NEXT:   - entry: float = 1.0,
; BFI-NEXT:   - loop: float = 1000.0, int = 18014398509481984
; BFI-NEXT:   - if.then: float = 1000.0, int = 18014398501093376
; BFI-NEXT:   - latch: float = 1000.0, int = 18014398509481984
; BFI-NEXT:   - exit: float = 1.0,
;
; VPLAN-LABEL: VPlan for loop in 'single_pred_zero_weight_sibling'
; VPLAN:       loop:
; VPLAN-NEXT:    EMIT-SCALAR ir<%iv> = phi [ ir<0>, vector.ph ], [ ir<%iv.next>, latch ]{{$}}
; VPLAN-NEXT:    EMIT ir<%gep.idx> = getelementptr inbounds ir<%idx>, ir<%iv>{{$}}
; VPLAN-NEXT:    EMIT-SCALAR ir<%i> = load ir<%gep.idx>{{$}}
; VPLAN-NEXT:    EMIT ir<%gep.b> = getelementptr inbounds ir<%b>, ir<%iv>{{$}}
; VPLAN-NEXT:    EMIT store ir<%i>, ir<%gep.b>{{$}}
; VPLAN-NEXT:    EMIT ir<%c.0> = icmp sgt ir<%i>, ir<0>{{$}}
; VPLAN-NEXT:    EMIT branch-on-cond ir<%c.0> (!prof {1000, 0}){{$}}
; VPLAN-NEXT:  Successor(s): if.then, latch
; VPLAN-EMPTY:
; VPLAN-NEXT:  if.then:
; VPLAN-NEXT:    EMIT ir<%gep.a> = getelementptr inbounds ir<%a>, ir<%iv>{{$}}
; VPLAN-NEXT:    EMIT store ir<%i>, ir<%gep.a>{{$}}
; VPLAN-NEXT:  Successor(s): latch
; VPLAN-EMPTY:
; VPLAN-NEXT:  latch:
; VPLAN-NEXT:    EMIT ir<%iv.next> = add ir<%iv>, ir<1>{{$}}
; VPLAN-NEXT:    EMIT ir<%ec> = icmp eq ir<%iv.next>, ir<1024>{{$}}
; VPLAN-NEXT:    EMIT branch-on-cond ir<%ec> (!prof {1, 999}){{$}}
; VPLAN-NEXT:  Successor(s): middle.block, loop
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %gep.idx = getelementptr inbounds i32, ptr %idx, i64 %iv
  %i = load i32, ptr %gep.idx, align 4
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %iv
  store i32 %i, ptr %gep.b, align 4
  %c.0 = icmp sgt i32 %i, 0
  br i1 %c.0, label %if.then, label %latch, !prof !14

if.then:
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  store i32 %i, ptr %gep.a, align 4
  br label %latch

latch:
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 1024
  br i1 %ec, label %exit, label %loop, !prof !3

exit:
  ret void
}

define void @two_preds(ptr noalias %a, ptr noalias %b, ptr noalias %c, ptr noalias %idx) {
; Execution frequency of each block of the loop. %merge is reached from both
; %then (1/4) and %else (3/4 * 1/3 = 1/4).
;
;   %loop      1000/1000 =   1
;   %then       250/1000 = 1/4
;   %else       750/1000 = 3/4
;   %merge      500/1000 = 1/2
;   %latch     1000/1000 =   1
;
; BFI-LABEL: block-frequency-info: two_preds
; BFI-NEXT:   - entry: float = 1.0,
; BFI-NEXT:   - loop: float = 1000.0,
; BFI-NEXT:   - then: float = 250.0,
; BFI-NEXT:   - else: float = 750.0,
; BFI-NEXT:   - merge: float = 500.0,
; BFI-NEXT:   - latch: float = 1000.0,
; BFI-NEXT:   - exit: float = 1.0,
;
; VPLAN-LABEL: VPlan for loop in 'two_preds'
; VPLAN:       loop:
; VPLAN-NEXT:    EMIT-SCALAR ir<%iv> = phi [ ir<0>, vector.ph ], [ ir<%iv.next>, latch ]{{$}}
; VPLAN-NEXT:    EMIT ir<%gep.idx> = getelementptr inbounds ir<%idx>, ir<%iv>{{$}}
; VPLAN-NEXT:    EMIT-SCALAR ir<%i> = load ir<%gep.idx>{{$}}
; VPLAN-NEXT:    EMIT ir<%c.0> = icmp sgt ir<%i>, ir<0>{{$}}
; VPLAN-NEXT:    EMIT branch-on-cond ir<%c.0> (!prof {1, 3}){{$}}
; VPLAN-NEXT:  Successor(s): then, else
; VPLAN-EMPTY:
; VPLAN-NEXT:  else:
; VPLAN-NEXT:    EMIT ir<%gep.c> = getelementptr inbounds ir<%c>, ir<%iv> (!vplan.execution.frequency 6917529027641081856 (75%))
; VPLAN-NEXT:    EMIT store ir<%i>, ir<%gep.c> (!vplan.execution.frequency 6917529027641081856 (75%))
; VPLAN-NEXT:    EMIT ir<%c.1> = icmp slt ir<%i>, ir<-100> (!vplan.execution.frequency 6917529027641081856 (75%))
; VPLAN-NEXT:    EMIT branch-on-cond ir<%c.1> (!prof {1, 2}, !vplan.execution.frequency 6917529027641081856 (75%))
; VPLAN-NEXT:  Successor(s): merge, latch
; VPLAN-EMPTY:
; VPLAN-NEXT:  then:
; VPLAN-NEXT:    EMIT ir<%gep.a> = getelementptr inbounds ir<%a>, ir<%iv> (!vplan.execution.frequency 2305843009213693952 (25%))
; VPLAN-NEXT:    EMIT store ir<%i>, ir<%gep.a> (!vplan.execution.frequency 2305843009213693952 (25%))
; VPLAN-NEXT:  Successor(s): merge
; VPLAN-EMPTY:
; VPLAN-NEXT:  merge:
; VPLAN-NEXT:    EMIT ir<%gep.b> = getelementptr inbounds ir<%b>, ir<%iv> (!vplan.execution.frequency 4611686019501129728 (50%))
; VPLAN-NEXT:    EMIT store ir<%i>, ir<%gep.b> (!vplan.execution.frequency 4611686019501129728 (50%))
; VPLAN-NEXT:  Successor(s): latch
; VPLAN-EMPTY:
; VPLAN-NEXT:  latch:
; VPLAN-NEXT:    EMIT ir<%iv.next> = add ir<%iv>, ir<1>{{$}}
; VPLAN-NEXT:    EMIT ir<%ec> = icmp eq ir<%iv.next>, ir<1024>{{$}}
; VPLAN-NEXT:    EMIT branch-on-cond ir<%ec> (!prof {1, 999}){{$}}
; VPLAN-NEXT:  Successor(s): middle.block, loop
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %gep.idx = getelementptr inbounds i32, ptr %idx, i64 %iv
  %i = load i32, ptr %gep.idx, align 4
  %c.0 = icmp sgt i32 %i, 0
  br i1 %c.0, label %then, label %else, !prof !0

then:
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  store i32 %i, ptr %gep.a, align 4
  br label %merge

else:
  %gep.c = getelementptr inbounds i32, ptr %c, i64 %iv
  store i32 %i, ptr %gep.c, align 4
  %c.1 = icmp slt i32 %i, -100
  br i1 %c.1, label %merge, label %latch, !prof !1

merge:
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %iv
  store i32 %i, ptr %gep.b, align 4
  br label %latch

latch:
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 1024
  br i1 %ec, label %exit, label %loop, !prof !3

exit:
  ret void
}

define void @nested_ifs(ptr noalias %a, ptr noalias %b, ptr noalias %idx) {
; Execution frequency of each block of the loop
;
;   %loop      1000/1000 =   1
;   %if.0       250/1000 = 1/4
;   %if.1       125/1000 = 1/8
;   %latch     1000/1000 =   1
;
; BFI-LABEL: block-frequency-info: nested_ifs
; BFI-NEXT:   - entry: float = 1.0,
; BFI-NEXT:   - loop: float = 1000.0,
; BFI-NEXT:   - if.0: float = 250.0,
; BFI-NEXT:   - if.1: float = 125.0,
; BFI-NEXT:   - latch: float = 1000.0,
; BFI-NEXT:   - exit: float = 1.0,
;
; VPLAN-LABEL: VPlan for loop in 'nested_ifs'
; VPLAN:       loop:
; VPLAN-NEXT:    EMIT-SCALAR ir<%iv> = phi [ ir<0>, vector.ph ], [ ir<%iv.next>, latch ]{{$}}
; VPLAN-NEXT:    EMIT ir<%gep.idx> = getelementptr inbounds ir<%idx>, ir<%iv>{{$}}
; VPLAN-NEXT:    EMIT-SCALAR ir<%i> = load ir<%gep.idx>{{$}}
; VPLAN-NEXT:    EMIT ir<%c.0> = icmp sgt ir<%i>, ir<0>{{$}}
; VPLAN-NEXT:    EMIT branch-on-cond ir<%c.0> (!prof {1, 3}){{$}}
; VPLAN-NEXT:  Successor(s): if.0, latch
; VPLAN-EMPTY:
; VPLAN-NEXT:  if.0:
; VPLAN-NEXT:    EMIT ir<%gep.b> = getelementptr inbounds ir<%b>, ir<%iv> (!vplan.execution.frequency 2305843009213693952 (25%))
; VPLAN-NEXT:    EMIT store ir<%i>, ir<%gep.b> (!vplan.execution.frequency 2305843009213693952 (25%))
; VPLAN-NEXT:    EMIT ir<%c.1> = icmp slt ir<%i>, ir<100> (!vplan.execution.frequency 2305843009213693952 (25%))
; VPLAN-NEXT:    EMIT branch-on-cond ir<%c.1> (!prof {1, 1}, !vplan.execution.frequency 2305843009213693952 (25%))
; VPLAN-NEXT:  Successor(s): if.1, latch
; VPLAN-EMPTY:
; VPLAN-NEXT:  if.1:
; VPLAN-NEXT:    EMIT ir<%gep.a> = getelementptr inbounds ir<%a>, ir<%iv> (!vplan.execution.frequency 1152921504606846976 (12.5%))
; VPLAN-NEXT:    EMIT store ir<%i>, ir<%gep.a> (!vplan.execution.frequency 1152921504606846976 (12.5%))
; VPLAN-NEXT:  Successor(s): latch
; VPLAN-EMPTY:
; VPLAN-NEXT:  latch:
; VPLAN-NEXT:    EMIT ir<%iv.next> = add ir<%iv>, ir<1>{{$}}
; VPLAN-NEXT:    EMIT ir<%ec> = icmp eq ir<%iv.next>, ir<1024>{{$}}
; VPLAN-NEXT:    EMIT branch-on-cond ir<%ec> (!prof {1, 999}){{$}}
; VPLAN-NEXT:  Successor(s): middle.block, loop
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %gep.idx = getelementptr inbounds i32, ptr %idx, i64 %iv
  %i = load i32, ptr %gep.idx, align 4
  %c.0 = icmp sgt i32 %i, 0
  br i1 %c.0, label %if.0, label %latch, !prof !0

if.0:
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %iv
  store i32 %i, ptr %gep.b, align 4
  %c.1 = icmp slt i32 %i, 100
  br i1 %c.1, label %if.1, label %latch, !prof !2

if.1:
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  store i32 %i, ptr %gep.a, align 4
  br label %latch

latch:
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 1024
  br i1 %ec, label %exit, label %loop, !prof !3

exit:
  ret void
}

define void @second_branch_without_weights(ptr noalias %a, ptr noalias %b, ptr noalias %idx) {
; %merge's branch has no weights, so BranchProbabilityInfo estimates them.
; %if.then.2's frequency is composed through that estimated edge, so it is
; marked estimated too.
;
;   %loop      1000/1000 =   1
;   %if.then.1  250/1000 = 1/4
;   %merge     1000/1000 =   1
;   %if.then.2  625/1000 = 5/8       (estimated)
;   %latch     1000/1000 =   1
;
; BFI-LABEL: block-frequency-info: second_branch_without_weights
; BFI-NEXT:   - entry: float = 1.0,
; BFI-NEXT:   - loop: float = 1000.0,
; BFI-NEXT:   - if.then.1: float = 250.0,
; BFI-NEXT:   - merge: float = 1000.0,
; BFI-NEXT:   - if.then.2: float = 625.0,
; BFI-NEXT:   - latch: float = 1000.0,
; BFI-NEXT:   - exit: float = 1.0,
;
; VPLAN-LABEL: VPlan for loop in 'second_branch_without_weights'
; VPLAN:       if.then.1:
; VPLAN-NEXT:    EMIT ir<%gep.a> = getelementptr inbounds ir<%a>, ir<%iv> (!vplan.execution.frequency 2305843009213693952 (25%))
; VPLAN-NEXT:    EMIT store ir<%i>, ir<%gep.a> (!vplan.execution.frequency 2305843009213693952 (25%))
; VPLAN-NEXT:  Successor(s): merge
; VPLAN-EMPTY:
; VPLAN-NEXT:  merge:
; VPLAN-NEXT:    EMIT branch-on-cond ir<%c.0> (!vplan.prof.estimated estimated {1342177280, 805306368}){{$}}
; VPLAN-NEXT:  Successor(s): if.then.2, latch
; VPLAN-EMPTY:
; VPLAN-NEXT:  if.then.2:
; VPLAN-NEXT:    EMIT ir<%gep.b> = getelementptr inbounds ir<%b>, ir<%iv> (!vplan.execution.frequency 5764607523034234880 (62.5%, estimated))
; VPLAN-NEXT:    EMIT store ir<%i>, ir<%gep.b> (!vplan.execution.frequency 5764607523034234880 (62.5%, estimated))
; VPLAN-NEXT:  Successor(s): latch
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %gep.idx = getelementptr inbounds i32, ptr %idx, i64 %iv
  %i = load i32, ptr %gep.idx, align 4
  %c.0 = icmp sgt i32 %i, 0
  br i1 %c.0, label %if.then.1, label %merge, !prof !0

if.then.1:
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  store i32 %i, ptr %gep.a, align 4
  br label %merge

merge:
  br i1 %c.0, label %if.then.2, label %latch

if.then.2:
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %iv
  store i32 %i, ptr %gep.b, align 4
  br label %latch

latch:
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 1024
  br i1 %ec, label %exit, label %loop, !prof !3

exit:
  ret void
}

define void @switch_common_dest(ptr noalias %a, ptr noalias %b, ptr noalias %c, ptr noalias %idx) {
; Execution frequency of each block of the loop. %if.then is reached from 2 of
; the switch's cases (125 + 250 = 375) and %default via the default edge (500).
;
;   %loop      1000/1000 =   1
;   %if.then    375/1000 = 3/8
;   %other      125/1000 = 1/8
;   %default    500/1000 = 1/2
;   %latch     1000/1000 =   1
;
; BFI-LABEL: block-frequency-info: switch_common_dest
; BFI-NEXT:   - entry: float = 1.0,
; BFI-NEXT:   - loop: float = 1000.0,
; BFI-NEXT:   - default: float = 500.0,
; BFI-NEXT:   - if.then: float = 375.0,
; BFI-NEXT:   - other: float = 125.0,
; BFI-NEXT:   - latch: float = 1000.0,
; BFI-NEXT:   - exit: float = 1.0,
;
; VPLAN-LABEL: VPlan for loop in 'switch_common_dest'
; VPLAN:       loop:
; VPLAN-NEXT:    EMIT-SCALAR ir<%iv> = phi [ ir<0>, vector.ph ], [ ir<%iv.next>, latch ]{{$}}
; VPLAN-NEXT:    EMIT ir<%gep.idx> = getelementptr inbounds ir<%idx>, ir<%iv>{{$}}
; VPLAN-NEXT:    EMIT-SCALAR ir<%l> = load ir<%gep.idx>{{$}}
; VPLAN-NEXT:    EMIT switch ir<%l>, ir<0>, ir<1>, ir<2> (!prof {500, 125, 250, 125}){{$}}
; VPLAN-NEXT:  Successor(s): default, if.then, if.then, other
; VPLAN-EMPTY:
; VPLAN-NEXT:  other:
; VPLAN-NEXT:    EMIT ir<%gep.b> = getelementptr inbounds ir<%b>, ir<%iv> (!vplan.execution.frequency 1152921504606846976 (12.5%))
; VPLAN-NEXT:    EMIT store ir<2>, ir<%gep.b> (!vplan.execution.frequency 1152921504606846976 (12.5%))
; VPLAN-NEXT:  Successor(s): latch
; VPLAN-EMPTY:
; VPLAN-NEXT:  if.then:
; VPLAN-NEXT:    EMIT ir<%gep.a> = getelementptr inbounds ir<%a>, ir<%iv> (!vplan.execution.frequency 3458764513820540928 (37.5%))
; VPLAN-NEXT:    EMIT store ir<1>, ir<%gep.a> (!vplan.execution.frequency 3458764513820540928 (37.5%))
; VPLAN-NEXT:  Successor(s): latch
; VPLAN-EMPTY:
; VPLAN-NEXT:  default:
; VPLAN-NEXT:    EMIT ir<%gep.c> = getelementptr inbounds ir<%c>, ir<%iv> (!vplan.execution.frequency 4611686018427387904 (50%))
; VPLAN-NEXT:    EMIT store ir<0>, ir<%gep.c> (!vplan.execution.frequency 4611686018427387904 (50%))
; VPLAN-NEXT:  Successor(s): latch
; VPLAN-EMPTY:
; VPLAN-NEXT:  latch:
; VPLAN-NEXT:    EMIT ir<%iv.next> = add ir<%iv>, ir<1>{{$}}
; VPLAN-NEXT:    EMIT ir<%ec> = icmp eq ir<%iv.next>, ir<1024>{{$}}
; VPLAN-NEXT:    EMIT branch-on-cond ir<%ec> (!prof {1, 999}){{$}}
; VPLAN-NEXT:  Successor(s): middle.block, loop
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %gep.idx = getelementptr inbounds i8, ptr %idx, i64 %iv
  %l = load i8, ptr %gep.idx, align 1
  switch i8 %l, label %default [
    i8 0, label %if.then
    i8 1, label %if.then
    i8 2, label %other
  ], !prof !4

default:
  %gep.c = getelementptr inbounds i8, ptr %c, i64 %iv
  store i8 0, ptr %gep.c, align 1
  br label %latch

if.then:
  %gep.a = getelementptr inbounds i8, ptr %a, i64 %iv
  store i8 1, ptr %gep.a, align 1
  br label %latch

other:
  %gep.b = getelementptr inbounds i8, ptr %b, i64 %iv
  store i8 2, ptr %gep.b, align 1
  br label %latch

latch:
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 1024
  br i1 %ec, label %exit, label %loop, !prof !3

exit:
  ret void
}

define void @switch_common_dest_weight_sum_not_a_power_of_two(ptr noalias %a, ptr noalias %b, ptr noalias %idx) {
; %if.then is reached from both of the switch's cases, %default via the default
; edge. The weights do not sum to a power of two, so converting each edge to a
; probability on its own does not divide evenly.
;
;   %loop      1000/1000 =   1
;   %if.then    667/1000 = 2/3
;   %default    333/1000 = 1/3
;   %latch     1000/1000 =   1
;
; BFI-LABEL: block-frequency-info: switch_common_dest_weight_sum_not_a_power_of_two
; BFI-NEXT:   - entry: float = 1.0,
; BFI-NEXT:   - loop: float = 1000.0,
; BFI-NEXT:   - default: float = 333.33,
; BFI-NEXT:   - if.then: float = 666.67,
; BFI-NEXT:   - latch: float = 1000.0,
; BFI-NEXT:   - exit: float = 1.0,
;
; VPLAN-LABEL: VPlan for loop in 'switch_common_dest_weight_sum_not_a_power_of_two'
; VPLAN:       loop:
; VPLAN-NEXT:    EMIT-SCALAR ir<%iv> = phi [ ir<0>, vector.ph ], [ ir<%iv.next>, latch ]{{$}}
; VPLAN-NEXT:    EMIT ir<%gep.idx> = getelementptr inbounds ir<%idx>, ir<%iv>{{$}}
; VPLAN-NEXT:    EMIT-SCALAR ir<%l> = load ir<%gep.idx>{{$}}
; VPLAN-NEXT:    EMIT switch ir<%l>, ir<0>, ir<1> (!prof {1, 1, 1}){{$}}
; VPLAN-NEXT:  Successor(s): default, if.then, if.then
; VPLAN-EMPTY:
; VPLAN-NEXT:  if.then:
; VPLAN-NEXT:    EMIT ir<%gep.a> = getelementptr inbounds ir<%a>, ir<%iv> (!vplan.execution.frequency 6148914689804861440 (66.67%))
; VPLAN-NEXT:    EMIT store ir<1>, ir<%gep.a> (!vplan.execution.frequency 6148914689804861440 (66.67%))
; VPLAN-NEXT:  Successor(s): latch
; VPLAN-EMPTY:
; VPLAN-NEXT:  default:
; VPLAN-NEXT:    EMIT ir<%gep.b> = getelementptr inbounds ir<%b>, ir<%iv> (!vplan.execution.frequency 3074457347049914368 (33.33%))
; VPLAN-NEXT:    EMIT store ir<0>, ir<%gep.b> (!vplan.execution.frequency 3074457347049914368 (33.33%))
; VPLAN-NEXT:  Successor(s): latch
; VPLAN-EMPTY:
; VPLAN-NEXT:  latch:
; VPLAN-NEXT:    EMIT ir<%iv.next> = add ir<%iv>, ir<1>{{$}}
; VPLAN-NEXT:    EMIT ir<%ec> = icmp eq ir<%iv.next>, ir<1024>{{$}}
; VPLAN-NEXT:    EMIT branch-on-cond ir<%ec> (!prof {1, 999}){{$}}
; VPLAN-NEXT:  Successor(s): middle.block, loop
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %gep.idx = getelementptr inbounds i8, ptr %idx, i64 %iv
  %l = load i8, ptr %gep.idx, align 1
  switch i8 %l, label %default [
    i8 0, label %if.then
    i8 1, label %if.then
  ], !prof !5

default:
  %gep.b = getelementptr inbounds i8, ptr %b, i64 %iv
  store i8 0, ptr %gep.b, align 1
  br label %latch

if.then:
  %gep.a = getelementptr inbounds i8, ptr %a, i64 %iv
  store i8 1, ptr %gep.a, align 1
  br label %latch

latch:
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 1024
  br i1 %ec, label %exit, label %loop, !prof !3

exit:
  ret void
}

define void @switch_common_dest_almost_always_taken(ptr noalias %a, ptr noalias %b, ptr noalias %idx) {
; %if.then is reached from both of the switch's cases, each taken with a
; probability just under 1/2, so together they are taken almost always.
;
;   %loop                        1 =   1
;   %if.then    4294967294/4294967295 ~   1
;   %default             1/4294967295 ~   0
;   %latch                       1 =   1
;
; BFI-LABEL: block-frequency-info: switch_common_dest_almost_always_taken
; BFI-NEXT:   - entry: float = 1.0,
; BFI-NEXT:   - loop: float = 1000.0,
; BFI-NEXT:   - default: float = 0.00000046566,
; BFI-NEXT:   - if.then: float = 1000.0,
; BFI-NEXT:   - latch: float = 1000.0,
; BFI-NEXT:   - exit: float = 1.0,
;
; VPLAN-LABEL: VPlan for loop in 'switch_common_dest_almost_always_taken'
; VPLAN:       loop:
; VPLAN-NEXT:    EMIT-SCALAR ir<%iv> = phi [ ir<0>, vector.ph ], [ ir<%iv.next>, latch ]{{$}}
; VPLAN-NEXT:    EMIT ir<%gep.idx> = getelementptr inbounds ir<%idx>, ir<%iv>{{$}}
; VPLAN-NEXT:    EMIT-SCALAR ir<%l> = load ir<%gep.idx>{{$}}
; VPLAN-NEXT:    EMIT switch ir<%l>, ir<0>, ir<1> (!prof {1, 2147483647, 2147483647}){{$}}
; VPLAN-NEXT:  Successor(s): default, if.then, if.then
; VPLAN-EMPTY:
; VPLAN-NEXT:  if.then:
; VPLAN-NEXT:    EMIT ir<%gep.a> = getelementptr inbounds ir<%a>, ir<%iv> (!vplan.execution.frequency 9223372032559808512 (100%))
; VPLAN-NEXT:    EMIT store ir<1>, ir<%gep.a> (!vplan.execution.frequency 9223372032559808512 (100%))
; VPLAN-NEXT:  Successor(s): latch
; VPLAN-EMPTY:
; VPLAN-NEXT:  default:
; VPLAN-NEXT:    EMIT ir<%gep.b> = getelementptr inbounds ir<%b>, ir<%iv> (!vplan.execution.frequency 4294967296 (4.657E-8%))
; VPLAN-NEXT:    EMIT store ir<0>, ir<%gep.b> (!vplan.execution.frequency 4294967296 (4.657E-8%))
; VPLAN-NEXT:  Successor(s): latch
; VPLAN-EMPTY:
; VPLAN-NEXT:  latch:
; VPLAN-NEXT:    EMIT ir<%iv.next> = add ir<%iv>, ir<1>{{$}}
; VPLAN-NEXT:    EMIT ir<%ec> = icmp eq ir<%iv.next>, ir<1024>{{$}}
; VPLAN-NEXT:    EMIT branch-on-cond ir<%ec> (!prof {1, 999}){{$}}
; VPLAN-NEXT:  Successor(s): middle.block, loop
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %gep.idx = getelementptr inbounds i8, ptr %idx, i64 %iv
  %l = load i8, ptr %gep.idx, align 1
  switch i8 %l, label %default [
    i8 0, label %if.then
    i8 1, label %if.then
  ], !prof !6

default:
  %gep.b = getelementptr inbounds i8, ptr %b, i64 %iv
  store i8 0, ptr %gep.b, align 1
  br label %latch

if.then:
  %gep.a = getelementptr inbounds i8, ptr %a, i64 %iv
  store i8 1, ptr %gep.a, align 1
  br label %latch

latch:
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 1024
  br i1 %ec, label %exit, label %loop, !prof !3

exit:
  ret void
}

define void @switch_common_dest_almost_never_taken(ptr noalias %a, ptr noalias %b, ptr noalias %idx) {
; %mid is almost never executed, and %if.then is reached from 4 of the switch's
; cases in %mid. Both are far below BranchProbability's 2^-31 resolution, so
; they must be represented as block frequencies to stay distinguishable from
; zero.
;
;   %loop                       1 =   1
;   %mid             1/2147483648 ~   0
;   %if.then    4/5 * 1/2147483648 ~   0
;   %latch                      1 =   1
;
; BFI-LABEL: block-frequency-info: switch_common_dest_almost_never_taken
; BFI-NEXT:   - entry: float = 1.0,
; BFI-NEXT:   - loop: float = 1000.0,
; BFI-NEXT:   - mid: float = 0.00000046566,
; BFI-NEXT:   - if.then: float = 0.00000037253,
; BFI-NEXT:   - latch: float = 1000.0,
; BFI-NEXT:   - exit: float = 1.0,
;
; VPLAN-LABEL: VPlan for loop in 'switch_common_dest_almost_never_taken'
; VPLAN:       loop:
; VPLAN-NEXT:    EMIT-SCALAR ir<%iv> = phi [ ir<0>, vector.ph ], [ ir<%iv.next>, latch ]{{$}}
; VPLAN-NEXT:    EMIT ir<%gep.idx> = getelementptr inbounds ir<%idx>, ir<%iv>{{$}}
; VPLAN-NEXT:    EMIT-SCALAR ir<%l> = load ir<%gep.idx>{{$}}
; VPLAN-NEXT:    EMIT ir<%c> = icmp sgt ir<%l>, ir<0>{{$}}
; VPLAN-NEXT:    EMIT branch-on-cond ir<%c> (!prof {1, 2147483647}){{$}}
; VPLAN-NEXT:  Successor(s): mid, latch
; VPLAN-EMPTY:
; VPLAN-NEXT:  mid:
; VPLAN-NEXT:    EMIT ir<%gep.b> = getelementptr inbounds ir<%b>, ir<%iv> (!vplan.execution.frequency 4294967296 (4.657E-8%))
; VPLAN-NEXT:    EMIT store ir<0>, ir<%gep.b> (!vplan.execution.frequency 4294967296 (4.657E-8%))
; VPLAN-NEXT:    EMIT switch ir<%l>, ir<1>, ir<2>, ir<3>, ir<4> (!prof {1, 1, 1, 1, 1}, !vplan.execution.frequency 4294967296 (4.657E-8%))
; VPLAN-NEXT:  Successor(s): latch, if.then, if.then, if.then, if.then
; VPLAN-EMPTY:
; VPLAN-NEXT:  if.then:
; VPLAN-NEXT:    EMIT ir<%gep.a> = getelementptr inbounds ir<%a>, ir<%iv> (!vplan.execution.frequency 3435973836 (3.725E-8%))
; VPLAN-NEXT:    EMIT store ir<1>, ir<%gep.a> (!vplan.execution.frequency 3435973836 (3.725E-8%))
; VPLAN-NEXT:  Successor(s): latch
; VPLAN-EMPTY:
; VPLAN-NEXT:  latch:
; VPLAN-NEXT:    EMIT ir<%iv.next> = add ir<%iv>, ir<1>{{$}}
; VPLAN-NEXT:    EMIT ir<%ec> = icmp eq ir<%iv.next>, ir<1024>{{$}}
; VPLAN-NEXT:    EMIT branch-on-cond ir<%ec> (!prof {1, 999}){{$}}
; VPLAN-NEXT:  Successor(s): middle.block, loop
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %gep.idx = getelementptr inbounds i8, ptr %idx, i64 %iv
  %l = load i8, ptr %gep.idx, align 1
  %c = icmp sgt i8 %l, 0
  br i1 %c, label %mid, label %latch, !prof !7

mid:
  %gep.b = getelementptr inbounds i8, ptr %b, i64 %iv
  store i8 0, ptr %gep.b, align 1
  switch i8 %l, label %latch [
    i8 1, label %if.then
    i8 2, label %if.then
    i8 3, label %if.then
    i8 4, label %if.then
  ], !prof !8

if.then:
  %gep.a = getelementptr inbounds i8, ptr %a, i64 %iv
  store i8 1, ptr %gep.a, align 1
  br label %latch

latch:
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 1024
  br i1 %ec, label %exit, label %loop, !prof !3

exit:
  ret void
}

define void @switch_common_dest_many_edges_almost_never_taken(ptr noalias %a, ptr noalias %b, ptr noalias %idx) {
; Same as @switch_common_dest_almost_never_taken, but with more parallel edges
; to %if.then. Their weights must be summed per successor before the edge
; probability is computed, so that it is rounded once rather than once per edge.
;
;   %loop                       1 =   1
;   %mid             1/2147483648 ~   0
;   %if.then    8/9 * 1/2147483648 ~   0
;   %latch                      1 =   1
;
; BFI-LABEL: block-frequency-info: switch_common_dest_many_edges_almost_never_taken
; BFI-NEXT:   - entry: float = 1.0,
; BFI-NEXT:   - loop: float = 1000.0,
; BFI-NEXT:   - mid: float = 0.00000046566,
; BFI-NEXT:   - if.then: float = 0.00000041392,
; BFI-NEXT:   - latch: float = 1000.0,
; BFI-NEXT:   - exit: float = 1.0,
;
; VPLAN-LABEL: VPlan for loop in 'switch_common_dest_many_edges_almost_never_taken'
; VPLAN:       loop:
; VPLAN-NEXT:    EMIT-SCALAR ir<%iv> = phi [ ir<0>, vector.ph ], [ ir<%iv.next>, latch ]{{$}}
; VPLAN-NEXT:    EMIT ir<%gep.idx> = getelementptr inbounds ir<%idx>, ir<%iv>{{$}}
; VPLAN-NEXT:    EMIT-SCALAR ir<%l> = load ir<%gep.idx>{{$}}
; VPLAN-NEXT:    EMIT ir<%c> = icmp sgt ir<%l>, ir<0>{{$}}
; VPLAN-NEXT:    EMIT branch-on-cond ir<%c> (!prof {1, 2147483647}){{$}}
; VPLAN-NEXT:  Successor(s): mid, latch
; VPLAN-EMPTY:
; VPLAN-NEXT:  mid:
; VPLAN-NEXT:    EMIT ir<%gep.b> = getelementptr inbounds ir<%b>, ir<%iv> (!vplan.execution.frequency 4294967296 (4.657E-8%))
; VPLAN-NEXT:    EMIT store ir<0>, ir<%gep.b> (!vplan.execution.frequency 4294967296 (4.657E-8%))
; VPLAN-NEXT:    EMIT switch ir<%l>, ir<1>, ir<2>, ir<3>, ir<4>, ir<5>, ir<6>, ir<7>, ir<8> (!prof {1, 1, 1, 1, 1, 1, 1, 1, 1}, !vplan.execution.frequency 4294967296 (4.657E-8%))
; VPLAN-NEXT:  Successor(s): latch, if.then, if.then, if.then, if.then, if.then, if.then, if.then, if.then
; VPLAN-EMPTY:
; VPLAN-NEXT:  if.then:
; VPLAN-NEXT:    EMIT ir<%gep.a> = getelementptr inbounds ir<%a>, ir<%iv> (!vplan.execution.frequency 3817748708 (4.139E-8%))
; VPLAN-NEXT:    EMIT store ir<1>, ir<%gep.a> (!vplan.execution.frequency 3817748708 (4.139E-8%))
; VPLAN-NEXT:  Successor(s): latch
; VPLAN-EMPTY:
; VPLAN-NEXT:  latch:
; VPLAN-NEXT:    EMIT ir<%iv.next> = add ir<%iv>, ir<1>{{$}}
; VPLAN-NEXT:    EMIT ir<%ec> = icmp eq ir<%iv.next>, ir<1024>{{$}}
; VPLAN-NEXT:    EMIT branch-on-cond ir<%ec> (!prof {1, 999}){{$}}
; VPLAN-NEXT:  Successor(s): middle.block, loop
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %gep.idx = getelementptr inbounds i8, ptr %idx, i64 %iv
  %l = load i8, ptr %gep.idx, align 1
  %c = icmp sgt i8 %l, 0
  br i1 %c, label %mid, label %latch, !prof !7

mid:
  %gep.b = getelementptr inbounds i8, ptr %b, i64 %iv
  store i8 0, ptr %gep.b, align 1
  switch i8 %l, label %latch [
    i8 1, label %if.then
    i8 2, label %if.then
    i8 3, label %if.then
    i8 4, label %if.then
    i8 5, label %if.then
    i8 6, label %if.then
    i8 7, label %if.then
    i8 8, label %if.then
  ], !prof !9

if.then:
  %gep.a = getelementptr inbounds i8, ptr %a, i64 %iv
  store i8 1, ptr %gep.a, align 1
  br label %latch

latch:
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 1024
  br i1 %ec, label %exit, label %loop, !prof !3

exit:
  ret void
}

define void @switch_common_dest_weight_sum_exceeds_32_bits(ptr noalias %a, ptr noalias %idx) {
; The weights of the five parallel edges to %if.then sum to more than 2^32, so
; scaling the frequency by them must not lose enough precision to round
; %if.then's frequency up to the one of an always executing block; that would
; drop the recorded frequency, because such a frequency needs no annotation.
;
;   %loop                              1 =   1
;   %if.then     21474836475/21474836511 ~   1
;   %latch                             1 =   1
;
; BFI-LABEL: block-frequency-info: switch_common_dest_weight_sum_exceeds_32_bits
; BFI-NEXT:   - entry: float = 1.0,
; BFI-NEXT:   - loop: float = 1000.0,
; BFI-NEXT:   - if.then: float = 1000.0,
; BFI-NEXT:   - latch: float = 1000.0,
; BFI-NEXT:   - exit: float = 1.0,
;
; VPLAN-LABEL: VPlan for loop in 'switch_common_dest_weight_sum_exceeds_32_bits'
; VPLAN:       loop:
; VPLAN-NEXT:    EMIT-SCALAR ir<%iv> = phi [ ir<0>, vector.ph ], [ ir<%iv.next>, latch ]{{$}}
; VPLAN-NEXT:    EMIT ir<%gep.idx> = getelementptr inbounds ir<%idx>, ir<%iv>{{$}}
; VPLAN-NEXT:    EMIT-SCALAR ir<%l> = load ir<%gep.idx>{{$}}
; VPLAN-NEXT:    EMIT switch ir<%l>, ir<0>, ir<1>, ir<2>, ir<3>, ir<4> (!prof {36, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295}){{$}}
; VPLAN-NEXT:  Successor(s): latch, if.then, if.then, if.then, if.then, if.then
; VPLAN-EMPTY:
; VPLAN-NEXT:  if.then:
; VPLAN-NEXT:    EMIT ir<%gep.a> = getelementptr inbounds ir<%a>, ir<%iv> (!vplan.execution.frequency 9223372023969873920 (100%))
; VPLAN-NEXT:    EMIT store ir<1>, ir<%gep.a> (!vplan.execution.frequency 9223372023969873920 (100%))
; VPLAN-NEXT:  Successor(s): latch
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %gep.idx = getelementptr inbounds i32, ptr %idx, i64 %iv
  %l = load i32, ptr %gep.idx, align 4
  switch i32 %l, label %latch [
    i32 0, label %if.then
    i32 1, label %if.then
    i32 2, label %if.then
    i32 3, label %if.then
    i32 4, label %if.then
  ], !prof !10

if.then:
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  store i32 1, ptr %gep.a, align 4
  br label %latch

latch:
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 1024
  br i1 %ec, label %exit, label %loop, !prof !3

exit:
  ret void
}

define void @switch_many_edges_to_latch(ptr noalias %a, ptr noalias %idx) {
; %if.then is reached only via the switch's default edge, while 32 parallel
; edges go to %latch. BlockFrequencyInfo rounds once per edge, so the
; cross-check's tolerance has to account for the number of edges, not just the
; number of blocks.
;
;   %loop                1 =      1
;   %if.then     1000/1224 ~ 0.8170
;   %latch               1 =      1
;
; BFI-LABEL: block-frequency-info: switch_many_edges_to_latch
; BFI-NEXT:   - entry: float = 1.0,
; BFI-NEXT:   - loop: float = 1000.0,
; BFI-NEXT:   - if.then: float = 816.99,
; BFI-NEXT:   - latch: float = 1000.0,
; BFI-NEXT:   - exit: float = 1.0,
;
; VPLAN-LABEL: VPlan for loop in 'switch_many_edges_to_latch'
; VPLAN:       if.then:
; VPLAN-NEXT:    EMIT ir<%gep.a> = getelementptr inbounds ir<%a>, ir<%iv> (!vplan.execution.frequency 7535434672457646080 (81.7%))
; VPLAN-NEXT:    EMIT store ir<1>, ir<%gep.a> (!vplan.execution.frequency 7535434672457646080 (81.7%))
; VPLAN-NEXT:  Successor(s): latch
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %gep.idx = getelementptr inbounds i32, ptr %idx, i64 %iv
  %l = load i32, ptr %gep.idx, align 4
  switch i32 %l, label %if.then [
    i32 0, label %latch
    i32 1, label %latch
    i32 2, label %latch
    i32 3, label %latch
    i32 4, label %latch
    i32 5, label %latch
    i32 6, label %latch
    i32 7, label %latch
    i32 8, label %latch
    i32 9, label %latch
    i32 10, label %latch
    i32 11, label %latch
    i32 12, label %latch
    i32 13, label %latch
    i32 14, label %latch
    i32 15, label %latch
    i32 16, label %latch
    i32 17, label %latch
    i32 18, label %latch
    i32 19, label %latch
    i32 20, label %latch
    i32 21, label %latch
    i32 22, label %latch
    i32 23, label %latch
    i32 24, label %latch
    i32 25, label %latch
    i32 26, label %latch
    i32 27, label %latch
    i32 28, label %latch
    i32 29, label %latch
    i32 30, label %latch
    i32 31, label %latch
  ], !prof !11

if.then:
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  store i32 1, ptr %gep.a, align 4
  br label %latch

latch:
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 1024
  br i1 %ec, label %exit, label %loop, !prof !3

exit:
  ret void
}

!10 = !{!"branch_weights", i32 36, i32 4294967295, i32 4294967295, i32 4294967295, i32 4294967295, i32 4294967295}
!11 = !{!"branch_weights", i32 1000, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7}

define void @switch_weights_clamped_at_both_ends(ptr noalias %a, ptr noalias %b, ptr noalias %idx) {
; The weights sum to more than 2^32.
;
;   %loop                            1 =        1
;   %if.then     8589934593/8589934595 ~        1
;   %default              2/8589934595 ~ 2.33e-10
;   %latch                           1 =        1
;
; BFI-LABEL: block-frequency-info: switch_weights_clamped_at_both_ends
; BFI-NEXT:   - entry: float = 1.0,
; BFI-NEXT:   - loop: float = 1000.0,
; BFI-NEXT:   - default: float = 0.00000046566,
; BFI-NEXT:   - if.then: float = 1000.0,
; BFI-NEXT:   - latch: float = 1000.0,
; BFI-NEXT:   - exit: float = 1.0,
;
; VPLAN-LABEL: VPlan for loop in 'switch_weights_clamped_at_both_ends'
; VPLAN:       if.then:
; VPLAN-NEXT:    EMIT ir<%gep.a> = getelementptr inbounds ir<%a>, ir<%iv> (!vplan.execution.frequency 9223372032559808512 (100%))
; VPLAN-NEXT:    EMIT store ir<1>, ir<%gep.a> (!vplan.execution.frequency 9223372032559808512 (100%))
; VPLAN-NEXT:  Successor(s): latch
; VPLAN-EMPTY:
; VPLAN-NEXT:  default:
; VPLAN-NEXT:    EMIT ir<%gep.b> = getelementptr inbounds ir<%b>, ir<%iv> (!vplan.execution.frequency 4294967296 (4.657E-8%))
; VPLAN-NEXT:    EMIT store ir<0>, ir<%gep.b> (!vplan.execution.frequency 4294967296 (4.657E-8%))
; VPLAN-NEXT:  Successor(s): latch
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %gep.idx = getelementptr inbounds i32, ptr %idx, i64 %iv
  %l = load i32, ptr %gep.idx, align 4
  switch i32 %l, label %default [
    i32 0, label %if.then
    i32 1, label %if.then
    i32 2, label %if.then
  ], !prof !12

default:
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %iv
  store i32 0, ptr %gep.b, align 4
  br label %latch

if.then:
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  store i32 1, ptr %gep.a, align 4
  br label %latch

latch:
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 1024
  br i1 %ec, label %exit, label %loop, !prof !3

exit:
  ret void
}

define void @nested_blocks_almost_never_entered(ptr noalias %a, ptr noalias %idx) {
; Each nested block is entered with probability 2^-31
;
;   %loop          1 =        1
;   %if.then.1 2^-31 ~ 4.66e-10
;   %if.then.2 2^-62 ~ 2.17e-19
;   %if.then.3 2^-63 ~ 1.08e-19  (clamped up from 2^-93)
;   %latch         1 =        1
;
; BFI-LABEL: block-frequency-info: nested_blocks_almost_never_entered
; BFI-NEXT:   - entry: float = 1.0,
; BFI-NEXT:   - loop: float = 1000.0,
; BFI-NEXT:   - if.then.1: float = 0.00000046566,
; BFI-NEXT:   - if.then.2: float = 0.00000000000000021684,
; BFI-NEXT:   - if.then.3: float = 0.00000000000000005421,
; BFI-NEXT:   - latch: float = 1000.0,
; BFI-NEXT:   - exit: float = 1.0,
;
; VPLAN-LABEL: VPlan for loop in 'nested_blocks_almost_never_entered'
; VPLAN:       if.then.1:
; VPLAN-NEXT:    EMIT ir<%c.1> = icmp sgt ir<%l>, ir<1> (!vplan.execution.frequency 4294967296 (4.657E-8%))
; VPLAN-NEXT:    EMIT branch-on-cond ir<%c.1> (!prof {1, 2147483647}, !vplan.execution.frequency 4294967296 (4.657E-8%))
; VPLAN-NEXT:  Successor(s): if.then.2, latch
; VPLAN-EMPTY:
; VPLAN-NEXT:  if.then.2:
; VPLAN-NEXT:    EMIT ir<%c.2> = icmp sgt ir<%l>, ir<2> (!vplan.execution.frequency 2 (2.168E-17%))
; VPLAN-NEXT:    EMIT branch-on-cond ir<%c.2> (!prof {1, 2147483647}, !vplan.execution.frequency 2 (2.168E-17%))
; VPLAN-NEXT:  Successor(s): if.then.3, latch
; VPLAN-EMPTY:
; VPLAN-NEXT:  if.then.3:
; VPLAN-NEXT:    EMIT ir<%gep.a> = getelementptr inbounds ir<%a>, ir<%iv> (!vplan.execution.frequency 1 (1.084E-17%))
; VPLAN-NEXT:    EMIT store ir<1>, ir<%gep.a> (!vplan.execution.frequency 1 (1.084E-17%))
; VPLAN-NEXT:  Successor(s): latch
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %gep.idx = getelementptr inbounds i32, ptr %idx, i64 %iv
  %l = load i32, ptr %gep.idx, align 4
  %c.0 = icmp sgt i32 %l, 0
  br i1 %c.0, label %if.then.1, label %latch, !prof !7

if.then.1:
  %c.1 = icmp sgt i32 %l, 1
  br i1 %c.1, label %if.then.2, label %latch, !prof !7

if.then.2:
  %c.2 = icmp sgt i32 %l, 2
  br i1 %c.2, label %if.then.3, label %latch, !prof !7

if.then.3:
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  store i32 1, ptr %gep.a, align 4
  br label %latch

latch:
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 1024
  br i1 %ec, label %exit, label %loop, !prof !3

exit:
  ret void
}

define void @switch_join_always(ptr noalias %a, ptr noalias %idx) {
; %join executes on every path through the loop, so it always executes, even
; though none of the switch's probabilities (5/7, 1/7, 1/7) is exact.
;
;   %loop     1 =   1
;   %case.1 1/7 ~ 0.143
;   %case.2 1/7 ~ 0.143
;   %join     1 =   1
;   %latch    1 =   1
;
; TODO: %join and %latch are currently recorded as executing slightly less
; often, as the rounded probabilities do not add up to 1.
;
; BFI-LABEL: block-frequency-info: switch_join_always
; BFI-NEXT:   - entry: float = 1.0,
; BFI-NEXT:   - loop: float = 1000.0,
; BFI-NEXT:   - case.1: float = 142.86,
; BFI-NEXT:   - case.2: float = 142.86,
; BFI-NEXT:   - join: float = 1000.0,
; BFI-NEXT:   - latch: float = 1000.0,
; BFI-NEXT:   - exit: float = 1.0,
;
; VPLAN-LABEL: VPlan for loop in 'switch_join_always'
; VPLAN:       case.2:
; VPLAN-NEXT:    EMIT store ir<2>, ir<%gep.a> (!vplan.execution.frequency 1317624575466405888 (14.29%))
; VPLAN-NEXT:  Successor(s): join
; VPLAN-EMPTY:
; VPLAN-NEXT:  case.1:
; VPLAN-NEXT:    EMIT store ir<1>, ir<%gep.a> (!vplan.execution.frequency 1317624575466405888 (14.29%))
; VPLAN-NEXT:  Successor(s): join
; VPLAN-EMPTY:
; VPLAN-NEXT:  join:
; VPLAN-NEXT:    EMIT ir<%add> = add ir<%i>, ir<10> (!vplan.execution.frequency 9223372032559808512 (100%))
; VPLAN-NEXT:    EMIT store ir<%add>, ir<%gep.a> (!vplan.execution.frequency 9223372032559808512 (100%))
; VPLAN-NEXT:  Successor(s): latch
; VPLAN-EMPTY:
; VPLAN-NEXT:  latch:
; VPLAN-NEXT:    EMIT ir<%iv.next> = add ir<%iv>, ir<1> (!vplan.execution.frequency 9223372032559808512 (100%))
; VPLAN-NEXT:    EMIT ir<%ec> = icmp eq ir<%iv.next>, ir<1024> (!vplan.execution.frequency 9223372032559808512 (100%))
; VPLAN-NEXT:    EMIT branch-on-cond ir<%ec> (!prof {1, 999}, !vplan.execution.frequency 9223372032559808512 (100%))
; VPLAN-NEXT:  Successor(s): middle.block, loop
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %gep.idx = getelementptr inbounds i32, ptr %idx, i64 %iv
  %i = load i32, ptr %gep.idx, align 4
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  switch i32 %i, label %join [
    i32 1, label %case.1
    i32 2, label %case.2
  ], !prof !15

case.1:
  store i32 1, ptr %gep.a, align 4
  br label %join

case.2:
  store i32 2, ptr %gep.a, align 4
  br label %join

join:
  %add = add i32 %i, 10
  store i32 %add, ptr %gep.a, align 4
  br label %latch

latch:
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 1024
  br i1 %ec, label %exit, label %loop, !prof !3

exit:
  ret void
}

define void @all_zero_weights(ptr noalias %a, ptr noalias %idx) {
; The branch in %if.then has all-zero weights. BranchProbabilityInfo takes both
; of its edges as equally likely.
;
;   %loop         1 =   1
;   %if.then    1/2 = 1/2
;   %if.then.2  1/4 = 1/4
;   %join       3/4 = 3/4
;   %if.then.3  3/8 = 3/8
;   %latch        1 =   1
;
; TODO: VPlan currently treats all-zero weights as unknown, so it records no
; frequency for %if.then.2 and the blocks it reaches.
;
; BFI-LABEL: block-frequency-info: all_zero_weights
; BFI-NEXT:   - entry: float = 1.0,
; BFI-NEXT:   - loop: float = 1000.0,
; BFI-NEXT:   - if.then: float = 500.0,
; BFI-NEXT:   - if.then.2: float = 250.0,
; BFI-NEXT:   - join: float = 750.0,
; BFI-NEXT:   - if.then.3: float = 375.0,
; BFI-NEXT:   - latch: float = 1000.0,
; BFI-NEXT:   - exit: float = 1.0,
;
; VPLAN-LABEL: VPlan for loop in 'all_zero_weights'
; VPLAN:       if.then:
; VPLAN-NEXT:    EMIT store ir<1>, ir<%gep.a> (!vplan.execution.frequency 4611686018427387904 (50%))
; VPLAN-NEXT:    EMIT ir<%c.1> = icmp sgt ir<%i>, ir<10> (!vplan.execution.frequency 4611686018427387904 (50%))
; VPLAN-NEXT:    EMIT branch-on-cond ir<%c.1> (!prof {0, 0}, !vplan.execution.frequency 4611686018427387904 (50%))
; VPLAN-NEXT:  Successor(s): if.then.2, latch
; VPLAN-EMPTY:
; VPLAN-NEXT:  if.then.2:
; VPLAN-NEXT:    EMIT store ir<2>, ir<%gep.a>{{$}}
; VPLAN-NEXT:  Successor(s): join
; VPLAN-EMPTY:
; VPLAN-NEXT:  join:
; VPLAN-NEXT:    EMIT ir<%c.2> = icmp sgt ir<%i>, ir<20>{{$}}
; VPLAN-NEXT:    EMIT branch-on-cond ir<%c.2> (!prof {1, 1}){{$}}
; VPLAN-NEXT:  Successor(s): if.then.3, latch
; VPLAN-EMPTY:
; VPLAN-NEXT:  if.then.3:
; VPLAN-NEXT:    EMIT store ir<3>, ir<%gep.a>{{$}}
; VPLAN-NEXT:  Successor(s): latch
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %gep.idx = getelementptr inbounds i32, ptr %idx, i64 %iv
  %i = load i32, ptr %gep.idx, align 4
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  %c.0 = icmp sgt i32 %i, 0
  br i1 %c.0, label %if.then, label %join, !prof !2

if.then:
  store i32 1, ptr %gep.a, align 4
  %c.1 = icmp sgt i32 %i, 10
  br i1 %c.1, label %if.then.2, label %latch, !prof !16

if.then.2:
  store i32 2, ptr %gep.a, align 4
  br label %join

join:
  %c.2 = icmp sgt i32 %i, 20
  br i1 %c.2, label %if.then.3, label %latch, !prof !2

if.then.3:
  store i32 3, ptr %gep.a, align 4
  br label %latch

latch:
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 1024
  br i1 %ec, label %exit, label %loop, !prof !3

exit:
  ret void
}

define void @estimated_through_weighted_branch(ptr noalias %a, ptr noalias %idx) {
;
;   %loop            1 =    1
;   %outer.then    5/8 =  5/8   (estimated)
;   %inner.then   5/32 = 5/32   (estimated)
;   %latch           1 =    1
;
; BFI-LABEL: block-frequency-info: estimated_through_weighted_branch
; BFI-NEXT:   - entry: float = 1.0,
; BFI-NEXT:   - loop: float = 1000.0,
; BFI-NEXT:   - outer.then: float = 625.0,
; BFI-NEXT:   - inner.then: float = 156.25,
; BFI-NEXT:   - latch: float = 1000.0,
; BFI-NEXT:   - exit: float = 1.0,
;
; VPLAN-LABEL: VPlan for loop in 'estimated_through_weighted_branch'
; VPLAN:       outer.then:
; VPLAN-NEXT:    EMIT store ir<1>, ir<%gep.a> (!vplan.execution.frequency 5764607523034234880 (62.5%, estimated))
; VPLAN-NEXT:    EMIT ir<%c.1> = icmp sgt ir<%i>, ir<10> (!vplan.execution.frequency 5764607523034234880 (62.5%, estimated))
; VPLAN-NEXT:    EMIT branch-on-cond ir<%c.1> (!prof {1, 3}, !vplan.execution.frequency 5764607523034234880 (62.5%, estimated))
; VPLAN-NEXT:  Successor(s): inner.then, latch
; VPLAN-EMPTY:
; VPLAN-NEXT:  inner.then:
; VPLAN-NEXT:    EMIT store ir<2>, ir<%gep.a> (!vplan.execution.frequency 1441151880758558720 (15.63%, estimated))
; VPLAN-NEXT:  Successor(s): latch
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %gep.idx = getelementptr inbounds i32, ptr %idx, i64 %iv
  %i = load i32, ptr %gep.idx, align 4
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  %c.0 = icmp sgt i32 %i, 0
  br i1 %c.0, label %outer.then, label %latch

outer.then:
  store i32 1, ptr %gep.a, align 4
  %c.1 = icmp sgt i32 %i, 10
  br i1 %c.1, label %inner.then, label %latch, !prof !0

inner.then:
  store i32 2, ptr %gep.a, align 4
  br label %latch

latch:
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 1024
  br i1 %ec, label %exit, label %loop, !prof !3

exit:
  ret void
}

define void @rarely_executed_chain(ptr noalias %a, ptr noalias %idx) {
;aaaa
;   %loop     1 =        1
;   %if.a 2^-31 ~ 4.66e-10
;   %if.b 2^-62 ~ 2.17e-19
;   %if.c 2^-63 ~ 1.08e-19
;   %if.d 2^-64 ~ 5.42e-20
;   %if.e 2^-64 ~ 5.42e-20  (clamped up from 2^-65)
;   %latch    1 =        1
;
; TODO: %if.a and the blocks it reaches are currently recorded as never
; executing.
;
; BFI-LABEL: block-frequency-info: rarely_executed_chain
; BFI-NEXT:   - entry: float = 1.0,
; BFI-NEXT:   - loop: float = 1000.0,
; BFI-NEXT:   - if.a: float = 0.00000046566,
; BFI-NEXT:   - if.b: float = 0.00000000000000021684,
; BFI-NEXT:   - if.c: float = 0.00000000000000010842,
; BFI-NEXT:   - if.d: float = 0.00000000000000005421,
; BFI-NEXT:   - if.e: float = 0.00000000000000005421,
; BFI-NEXT:   - latch: float = 1000.0,
; BFI-NEXT:   - exit: float = 1.0,
;
; VPLAN-LABEL: VPlan for loop in 'rarely_executed_chain'
; VPLAN:       if.a:
; VPLAN-NEXT:    EMIT ir<%c.1> = icmp sgt ir<%i>, ir<10>{{$}}
; VPLAN-NEXT:    EMIT branch-on-cond ir<%c.1> (!prof {1000, 0}){{$}}
; VPLAN-NEXT:  Successor(s): latch, if.b
; VPLAN-EMPTY:
; VPLAN-NEXT:  if.b:
; VPLAN-NEXT:    EMIT ir<%c.2> = icmp sgt ir<%i>, ir<20>{{$}}
; VPLAN-NEXT:    EMIT branch-on-cond ir<%c.2> (!prof {1, 1}){{$}}
; VPLAN-NEXT:  Successor(s): if.c, latch
; VPLAN-EMPTY:
; VPLAN-NEXT:  if.c:
; VPLAN-NEXT:    EMIT ir<%c.3> = icmp sgt ir<%i>, ir<30>{{$}}
; VPLAN-NEXT:    EMIT branch-on-cond ir<%c.3> (!prof {1, 1}){{$}}
; VPLAN-NEXT:  Successor(s): latch, if.d
; VPLAN-EMPTY:
; VPLAN-NEXT:  if.d:
; VPLAN-NEXT:    EMIT store ir<1>, ir<%gep.a>{{$}}
; VPLAN-NEXT:    EMIT ir<%c.4> = icmp sgt ir<%i>, ir<40>{{$}}
; VPLAN-NEXT:    EMIT branch-on-cond ir<%c.4> (!prof {1, 1}){{$}}
; VPLAN-NEXT:  Successor(s): if.e, latch
; VPLAN-EMPTY:
; VPLAN-NEXT:  if.e:
; VPLAN-NEXT:    EMIT store ir<2>, ir<%gep.a>{{$}}
; VPLAN-NEXT:  Successor(s): latch
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %gep.idx = getelementptr inbounds i32, ptr %idx, i64 %iv
  %i = load i32, ptr %gep.idx, align 4
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  %c.0 = icmp sgt i32 %i, 0
  br i1 %c.0, label %latch, label %if.a, !prof !14

if.a:
  %c.1 = icmp sgt i32 %i, 10
  br i1 %c.1, label %latch, label %if.b, !prof !14

if.b:
  %c.2 = icmp sgt i32 %i, 20
  br i1 %c.2, label %if.c, label %latch, !prof !2

if.c:
  %c.3 = icmp sgt i32 %i, 30
  br i1 %c.3, label %latch, label %if.d, !prof !2

if.d:
  store i32 1, ptr %gep.a, align 4
  %c.4 = icmp sgt i32 %i, 40
  br i1 %c.4, label %if.e, label %latch, !prof !2

if.e:
  store i32 2, ptr %gep.a, align 4
  br label %latch

latch:
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 1024
  br i1 %ec, label %exit, label %loop, !prof !3

exit:
  ret void
}

define void @nested_zero_weight_siblings(ptr noalias %a, ptr noalias %idx) {
; Nested branches whose skipping edges have zero weight. Each is treated as
; cold.
;
;   %loop               1 =             1
;   %then.0  1 -   2^-31  ~ 1 - 4.66e-10
;   %then.1 (1 - 2^-31)^2 ~ 1 - 9.31e-10
;   %then.2 (1 - 2^-31)^3 ~ 1 - 1.40e-9
;   %latch              1 =             1
;
; TODO: %then.0, %then.1 and %then.2 are currently recorded as always executing.
;
; BFI-LABEL: block-frequency-info: nested_zero_weight_siblings
; BFI-NEXT:   - entry: float = 1.0,
; BFI-NEXT:   - loop: float = 1000.0, int = 18014398509481984
; BFI-NEXT:   - then.0: float = 1000.0, int = 18014398501093376
; BFI-NEXT:   - then.1: float = 1000.0, int = 18014398492704768
; BFI-NEXT:   - then.2: float = 1000.0, int = 18014398484316160
; BFI-NEXT:   - latch: float = 1000.0, int = 18014398509481984
; BFI-NEXT:   - exit: float = 1.0,
;
; VPLAN-LABEL: VPlan for loop in 'nested_zero_weight_siblings'
; VPLAN:       then.0:
; VPLAN-NEXT:    EMIT store ir<0>, ir<%gep.a>{{$}}
; VPLAN-NEXT:    EMIT ir<%c.1> = icmp sgt ir<%i>, ir<10>{{$}}
; VPLAN-NEXT:    EMIT branch-on-cond ir<%c.1> (!prof {1000, 0}){{$}}
; VPLAN-NEXT:  Successor(s): then.1, latch
; VPLAN-EMPTY:
; VPLAN-NEXT:  then.1:
; VPLAN-NEXT:    EMIT store ir<1>, ir<%gep.a>{{$}}
; VPLAN-NEXT:    EMIT ir<%c.2> = icmp sgt ir<%i>, ir<20>{{$}}
; VPLAN-NEXT:    EMIT branch-on-cond ir<%c.2> (!prof {1000, 0}){{$}}
; VPLAN-NEXT:  Successor(s): then.2, latch
; VPLAN-EMPTY:
; VPLAN-NEXT:  then.2:
; VPLAN-NEXT:    EMIT store ir<2>, ir<%gep.a>{{$}}
; VPLAN-NEXT:  Successor(s): latch
; VPLAN-EMPTY:
; VPLAN-NEXT:  latch:
; VPLAN-NEXT:    EMIT ir<%iv.next> = add ir<%iv>, ir<1>{{$}}
; VPLAN-NEXT:    EMIT ir<%ec> = icmp eq ir<%iv.next>, ir<1024>{{$}}
; VPLAN-NEXT:    EMIT branch-on-cond ir<%ec> (!prof {1, 999}){{$}}
; VPLAN-NEXT:  Successor(s): middle.block, loop
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %gep.idx = getelementptr inbounds i32, ptr %idx, i64 %iv
  %i = load i32, ptr %gep.idx, align 4
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  %c.0 = icmp sgt i32 %i, 0
  br i1 %c.0, label %then.0, label %latch, !prof !14

then.0:
  store i32 0, ptr %gep.a, align 4
  %c.1 = icmp sgt i32 %i, 10
  br i1 %c.1, label %then.1, label %latch, !prof !14

then.1:
  store i32 1, ptr %gep.a, align 4
  %c.2 = icmp sgt i32 %i, 20
  br i1 %c.2, label %then.2, label %latch, !prof !14

then.2:
  store i32 2, ptr %gep.a, align 4
  br label %latch

latch:
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 1024
  br i1 %ec, label %exit, label %loop, !prof !3

exit:
  ret void
}

!0 = !{!"branch_weights", i32 1, i32 3}
!1 = !{!"branch_weights", i32 1, i32 2}
!2 = !{!"branch_weights", i32 1, i32 1}
!3 = !{!"branch_weights", i32 1, i32 999}
!4 = !{!"branch_weights", i32 500, i32 125, i32 250, i32 125}
!5 = !{!"branch_weights", i32 1, i32 1, i32 1}
!6 = !{!"branch_weights", i32 1, i32 2147483647, i32 2147483647}
!7 = !{!"branch_weights", i32 1, i32 2147483647}
!8 = !{!"branch_weights", i32 1, i32 1, i32 1, i32 1, i32 1}
!9 = !{!"branch_weights", i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1}
!12 = !{!"branch_weights", i32 2, i32 2863311531, i32 2863311531, i32 2863311531}
!13 = !{!"branch_weights", i32 0, i32 1000}
!14 = !{!"branch_weights", i32 1000, i32 0}
!15 = !{!"branch_weights", i32 5, i32 1, i32 1}
!16 = !{!"branch_weights", i32 0, i32 0}
