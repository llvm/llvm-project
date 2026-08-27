; RUN: opt -passes='print<block-freq>' -disable-output %s 2>&1 \
; RUN:   | FileCheck --check-prefix=BFI %s
; RUN: opt -passes=loop-vectorize -force-vector-width=2 -force-vector-interleave=1 \
; RUN:     -vplan-print-after=introduceMasksAndLinearize -disable-output %s 2>&1 \
; RUN:   | FileCheck --check-prefix=VPLAN %s

; Check that the execution probabilities VPlan records on the masked recipes of
; a block match the block frequencies BlockFrequencyInfo computes for the
; corresponding block of the original scalar loop.

define void @single_pred(ptr noalias %a, ptr noalias %b, ptr noalias %idx) {
; Execution probability of each block of the loop
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
; VPLAN:         vector.body:
; VPLAN-NEXT:      ir<%iv> = WIDEN-INDUCTION ir<0>, ir<1>, vp<%{{.+}}>
; VPLAN-NEXT:      EMIT ir<%gep.idx> = getelementptr inbounds ir<%idx>, ir<%iv>
; VPLAN-NEXT:      EMIT-SCALAR ir<%i> = load ir<%gep.idx>
; VPLAN-NEXT:      EMIT ir<%gep.b> = getelementptr inbounds ir<%b>, ir<%iv>
; VPLAN-NEXT:      EMIT store ir<%i>, ir<%gep.b>{{$}}
; VPLAN-NEXT:      EMIT ir<%c.0> = icmp sgt ir<%i>, ir<0>
; VPLAN-NEXT:    Successor(s): if.then
; VPLAN-EMPTY:
; VPLAN-NEXT:    if.then:
; VPLAN-NEXT:      EMIT ir<%gep.a> = getelementptr inbounds ir<%a>, ir<%iv>
; VPLAN-NEXT:      EMIT store ir<%i>, ir<%gep.a>, ir<%c.0> (!vplan.execution.probability 25%)
; VPLAN-NEXT:    Successor(s): latch
; VPLAN-EMPTY:
; VPLAN-NEXT:    latch:
; VPLAN-NEXT:      EMIT ir<%iv.next> = add ir<%iv>, ir<1>
; VPLAN-NEXT:      EMIT ir<%ec> = icmp eq ir<%iv.next>, ir<1024>
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

define void @two_preds(ptr noalias %a, ptr noalias %b, ptr noalias %c, ptr noalias %idx) {
; Execution probability of each block of the loop. %merge is reached from both
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
; VPLAN:         vector.body:
; VPLAN-NEXT:      ir<%iv> = WIDEN-INDUCTION ir<0>, ir<1>, vp<%{{.+}}>
; VPLAN-NEXT:      EMIT ir<%gep.idx> = getelementptr inbounds ir<%idx>, ir<%iv>
; VPLAN-NEXT:      EMIT-SCALAR ir<%i> = load ir<%gep.idx>
; VPLAN-NEXT:      EMIT ir<%c.0> = icmp sgt ir<%i>, ir<0>
; VPLAN-NEXT:    Successor(s): else
; VPLAN-EMPTY:
; VPLAN-NEXT:    else:
; VPLAN-NEXT:      EMIT vp<[[NOT_C0:%.+]]> = not ir<%c.0>
; VPLAN-NEXT:      EMIT ir<%gep.c> = getelementptr inbounds ir<%c>, ir<%iv>
; VPLAN-NEXT:      EMIT store ir<%i>, ir<%gep.c>, vp<[[NOT_C0]]> (!vplan.execution.probability 75%)
; VPLAN-NEXT:      EMIT ir<%c.1> = icmp slt ir<%i>, ir<-100>, vp<[[NOT_C0]]> (!vplan.execution.probability 75%)
; VPLAN-NEXT:    Successor(s): then
; VPLAN-EMPTY:
; VPLAN-NEXT:    then:
; VPLAN-NEXT:      EMIT ir<%gep.a> = getelementptr inbounds ir<%a>, ir<%iv>
; VPLAN-NEXT:      EMIT store ir<%i>, ir<%gep.a>, ir<%c.0> (!vplan.execution.probability 25%)
; VPLAN-NEXT:    Successor(s): merge
; VPLAN-EMPTY:
; VPLAN-NEXT:    merge:
; VPLAN-NEXT:      EMIT vp<[[AND:%.+]]> = logical-and vp<[[NOT_C0]]>, ir<%c.1>
; VPLAN-NEXT:      EMIT vp<[[MASK:%.+]]> = or vp<[[AND]]>, ir<%c.0>
; VPLAN-NEXT:      EMIT ir<%gep.b> = getelementptr inbounds ir<%b>, ir<%iv>
; VPLAN-NEXT:      EMIT store ir<%i>, ir<%gep.b>, vp<[[MASK]]> (!vplan.execution.probability 50%)
; VPLAN-NEXT:    Successor(s): latch
; VPLAN-EMPTY:
; VPLAN-NEXT:    latch:
; VPLAN-NEXT:      EMIT ir<%iv.next> = add ir<%iv>, ir<1>
; VPLAN-NEXT:      EMIT ir<%ec> = icmp eq ir<%iv.next>, ir<1024>
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
; Execution probability of each block of the loop
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
; VPLAN:         vector.body:
; VPLAN-NEXT:      ir<%iv> = WIDEN-INDUCTION ir<0>, ir<1>, vp<%{{.+}}>
; VPLAN-NEXT:      EMIT ir<%gep.idx> = getelementptr inbounds ir<%idx>, ir<%iv>
; VPLAN-NEXT:      EMIT-SCALAR ir<%i> = load ir<%gep.idx>
; VPLAN-NEXT:      EMIT ir<%c.0> = icmp sgt ir<%i>, ir<0>
; VPLAN-NEXT:    Successor(s): if.0
; VPLAN-EMPTY:
; VPLAN-NEXT:    if.0:
; VPLAN-NEXT:      EMIT ir<%gep.b> = getelementptr inbounds ir<%b>, ir<%iv>
; VPLAN-NEXT:      EMIT store ir<%i>, ir<%gep.b>, ir<%c.0> (!vplan.execution.probability 25%)
; VPLAN-NEXT:      EMIT ir<%c.1> = icmp slt ir<%i>, ir<100>, ir<%c.0> (!vplan.execution.probability 25%)
; VPLAN-NEXT:    Successor(s): if.1
; VPLAN-EMPTY:
; VPLAN-NEXT:    if.1:
; VPLAN-NEXT:      EMIT vp<[[MASK:%.+]]> = logical-and ir<%c.0>, ir<%c.1>
; VPLAN-NEXT:      EMIT ir<%gep.a> = getelementptr inbounds ir<%a>, ir<%iv>
; VPLAN-NEXT:      EMIT store ir<%i>, ir<%gep.a>, vp<[[MASK]]> (!vplan.execution.probability 12.5%)
; VPLAN-NEXT:    Successor(s): latch
; VPLAN-EMPTY:
; VPLAN-NEXT:    latch:
; VPLAN-NEXT:      EMIT ir<%iv.next> = add ir<%iv>, ir<1>
; VPLAN-NEXT:      EMIT ir<%ec> = icmp eq ir<%iv.next>, ir<1024>
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

define void @switch_common_dest(ptr noalias %a, ptr noalias %b, ptr noalias %c, ptr noalias %idx) {
; Execution probability of each block of the loop. %if.then is reached from 2 of
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
; VPLAN:         vector.body:
; VPLAN-NEXT:      ir<%iv> = WIDEN-INDUCTION ir<0>, ir<1>, vp<%{{.+}}>
; VPLAN-NEXT:      EMIT ir<%gep.idx> = getelementptr inbounds ir<%idx>, ir<%iv>
; VPLAN-NEXT:      EMIT-SCALAR ir<%l> = load ir<%gep.idx>
; VPLAN-NEXT:    Successor(s): other
; VPLAN-EMPTY:
; VPLAN-NEXT:    other:
; VPLAN-NEXT:      EMIT vp<[[C0:%.+]]> = icmp eq ir<%l>, ir<0>
; VPLAN-NEXT:      EMIT vp<[[C1:%.+]]> = icmp eq ir<%l>, ir<1>
; VPLAN-NEXT:      EMIT vp<[[C2:%.+]]> = icmp eq ir<%l>, ir<2>
; VPLAN-NEXT:      EMIT vp<[[C0_OR_C1:%.+]]> = or vp<[[C0]]>, vp<[[C1]]>
; VPLAN-NEXT:      EMIT vp<[[ANY:%.+]]> = or vp<[[C0_OR_C1]]>, vp<[[C2]]>
; VPLAN-NEXT:      EMIT vp<[[DEFAULT:%.+]]> = not vp<[[ANY]]>
; VPLAN-NEXT:      EMIT ir<%gep.b> = getelementptr inbounds ir<%b>, ir<%iv>
; VPLAN-NEXT:      EMIT store ir<2>, ir<%gep.b>, vp<[[C2]]> (!vplan.execution.probability 12.5%)
; VPLAN-NEXT:    Successor(s): if.then
; VPLAN-EMPTY:
; VPLAN-NEXT:    if.then:
; VPLAN-NEXT:      EMIT ir<%gep.a> = getelementptr inbounds ir<%a>, ir<%iv>
; VPLAN-NEXT:      EMIT store ir<1>, ir<%gep.a>, vp<[[C0_OR_C1]]> (!vplan.execution.probability 37.5%)
; VPLAN-NEXT:    Successor(s): default
; VPLAN-EMPTY:
; VPLAN-NEXT:    default:
; VPLAN-NEXT:      EMIT ir<%gep.c> = getelementptr inbounds ir<%c>, ir<%iv>
; VPLAN-NEXT:      EMIT store ir<0>, ir<%gep.c>, vp<[[DEFAULT]]> (!vplan.execution.probability 50%)
; VPLAN-NEXT:    Successor(s): latch
; VPLAN-EMPTY:
; VPLAN-NEXT:    latch:
; VPLAN-NEXT:      EMIT ir<%iv.next> = add ir<%iv>, ir<1>
; VPLAN-NEXT:      EMIT ir<%ec> = icmp eq ir<%iv.next>, ir<1024>
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
; VPLAN:         vector.body:
; VPLAN-NEXT:      ir<%iv> = WIDEN-INDUCTION ir<0>, ir<1>, vp<%{{.+}}>
; VPLAN-NEXT:      EMIT ir<%gep.idx> = getelementptr inbounds ir<%idx>, ir<%iv>
; VPLAN-NEXT:      EMIT-SCALAR ir<%l> = load ir<%gep.idx>
; VPLAN-NEXT:    Successor(s): if.then
; VPLAN-EMPTY:
; VPLAN-NEXT:    if.then:
; VPLAN-NEXT:      EMIT vp<[[C0:%.+]]> = icmp eq ir<%l>, ir<0>
; VPLAN-NEXT:      EMIT vp<[[C1:%.+]]> = icmp eq ir<%l>, ir<1>
; VPLAN-NEXT:      EMIT vp<[[C0_OR_C1:%.+]]> = or vp<[[C0]]>, vp<[[C1]]>
; VPLAN-NEXT:      EMIT vp<[[DEFAULT:%.+]]> = not vp<[[C0_OR_C1]]>
; VPLAN-NEXT:      EMIT ir<%gep.a> = getelementptr inbounds ir<%a>, ir<%iv>
; VPLAN-NEXT:      EMIT store ir<1>, ir<%gep.a>, vp<[[C0_OR_C1]]> (!vplan.execution.probability 66.67%)
; VPLAN-NEXT:    Successor(s): default
; VPLAN-EMPTY:
; VPLAN-NEXT:    default:
; VPLAN-NEXT:      EMIT ir<%gep.b> = getelementptr inbounds ir<%b>, ir<%iv>
; VPLAN-NEXT:      EMIT store ir<0>, ir<%gep.b>, vp<[[DEFAULT]]> (!vplan.execution.probability 33.33%)
; VPLAN-NEXT:    Successor(s): latch
; VPLAN-EMPTY:
; VPLAN-NEXT:    latch:
; VPLAN-NEXT:      EMIT ir<%iv.next> = add ir<%iv>, ir<1>
; VPLAN-NEXT:      EMIT ir<%ec> = icmp eq ir<%iv.next>, ir<1024>
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
; VPLAN:         vector.body:
; VPLAN-NEXT:      ir<%iv> = WIDEN-INDUCTION ir<0>, ir<1>, vp<%{{.+}}>
; VPLAN-NEXT:      EMIT ir<%gep.idx> = getelementptr inbounds ir<%idx>, ir<%iv>
; VPLAN-NEXT:      EMIT-SCALAR ir<%l> = load ir<%gep.idx>
; VPLAN-NEXT:    Successor(s): if.then
; VPLAN-EMPTY:
; VPLAN-NEXT:    if.then:
; VPLAN-NEXT:      EMIT vp<[[C0:%.+]]> = icmp eq ir<%l>, ir<0>
; VPLAN-NEXT:      EMIT vp<[[C1:%.+]]> = icmp eq ir<%l>, ir<1>
; VPLAN-NEXT:      EMIT vp<[[C0_OR_C1:%.+]]> = or vp<[[C0]]>, vp<[[C1]]>
; VPLAN-NEXT:      EMIT vp<[[DEFAULT:%.+]]> = not vp<[[C0_OR_C1]]>
; VPLAN-NEXT:      EMIT ir<%gep.a> = getelementptr inbounds ir<%a>, ir<%iv>
; VPLAN-NEXT:      EMIT store ir<1>, ir<%gep.a>, vp<[[C0_OR_C1]]> (!vplan.execution.probability 100%)
; VPLAN-NEXT:    Successor(s): default
; VPLAN-EMPTY:
; VPLAN-NEXT:    default:
; VPLAN-NEXT:      EMIT ir<%gep.b> = getelementptr inbounds ir<%b>, ir<%iv>
; VPLAN-NEXT:      EMIT store ir<0>, ir<%gep.b>, vp<[[DEFAULT]]> (!vplan.execution.probability 4.657e-08%)
; VPLAN-NEXT:    Successor(s): latch
; VPLAN-EMPTY:
; VPLAN-NEXT:    latch:
; VPLAN-NEXT:      EMIT ir<%iv.next> = add ir<%iv>, ir<1>
; VPLAN-NEXT:      EMIT ir<%ec> = icmp eq ir<%iv.next>, ir<1024>
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
; cases in %mid. Both probabilities are far below BranchProbability's 2^-31
; resolution, so they must be represented as block frequencies to stay
; distinguishable from zero.
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
; VPLAN:         vector.body:
; VPLAN-NEXT:      ir<%iv> = WIDEN-INDUCTION ir<0>, ir<1>, vp<%{{.+}}>
; VPLAN-NEXT:      EMIT ir<%gep.idx> = getelementptr inbounds ir<%idx>, ir<%iv>
; VPLAN-NEXT:      EMIT-SCALAR ir<%l> = load ir<%gep.idx>
; VPLAN-NEXT:      EMIT ir<%c> = icmp sgt ir<%l>, ir<0>
; VPLAN-NEXT:    Successor(s): mid
; VPLAN-EMPTY:
; VPLAN-NEXT:    mid:
; VPLAN-NEXT:      EMIT ir<%gep.b> = getelementptr inbounds ir<%b>, ir<%iv>
; VPLAN-NEXT:      EMIT store ir<0>, ir<%gep.b>, ir<%c> (!vplan.execution.probability 4.657e-08%)
; VPLAN-NEXT:    Successor(s): if.then
; VPLAN-EMPTY:
; VPLAN-NEXT:    if.then:
; VPLAN-NEXT:      EMIT vp<[[C1:%.+]]> = icmp eq ir<%l>, ir<1>
; VPLAN-NEXT:      EMIT vp<[[C2:%.+]]> = icmp eq ir<%l>, ir<2>
; VPLAN-NEXT:      EMIT vp<[[C3:%.+]]> = icmp eq ir<%l>, ir<3>
; VPLAN-NEXT:      EMIT vp<[[C4:%.+]]> = icmp eq ir<%l>, ir<4>
; VPLAN-NEXT:      EMIT vp<[[OR_0:%.+]]> = or vp<[[C1]]>, vp<[[C2]]>
; VPLAN-NEXT:      EMIT vp<[[OR_1:%.+]]> = or vp<[[OR_0]]>, vp<[[C3]]>
; VPLAN-NEXT:      EMIT vp<[[ANY:%.+]]> = or vp<[[OR_1]]>, vp<[[C4]]>
; VPLAN-NEXT:      EMIT vp<[[MASK:%.+]]> = logical-and ir<%c>, vp<[[ANY]]>
; VPLAN-NEXT:      EMIT vp<[[NOT_MASK:%.+]]> = not vp<[[MASK]]>
; VPLAN-NEXT:      EMIT vp<[[DEFAULT:%.+]]> = logical-and ir<%c>, vp<[[NOT_MASK]]>
; VPLAN-NEXT:      EMIT ir<%gep.a> = getelementptr inbounds ir<%a>, ir<%iv>
; VPLAN-NEXT:      EMIT store ir<1>, ir<%gep.a>, vp<[[MASK]]> (!vplan.execution.probability 4.657e-08%)
; VPLAN-NEXT:    Successor(s): latch
; VPLAN-EMPTY:
; VPLAN-NEXT:    latch:
; VPLAN-NEXT:      EMIT ir<%iv.next> = add ir<%iv>, ir<1>
; VPLAN-NEXT:      EMIT ir<%ec> = icmp eq ir<%iv.next>, ir<1024>
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
; VPLAN:         vector.body:
; VPLAN-NEXT:      ir<%iv> = WIDEN-INDUCTION ir<0>, ir<1>, vp<%{{.+}}>
; VPLAN-NEXT:      EMIT ir<%gep.idx> = getelementptr inbounds ir<%idx>, ir<%iv>
; VPLAN-NEXT:      EMIT-SCALAR ir<%l> = load ir<%gep.idx>
; VPLAN-NEXT:      EMIT ir<%c> = icmp sgt ir<%l>, ir<0>
; VPLAN-NEXT:    Successor(s): mid
; VPLAN-EMPTY:
; VPLAN-NEXT:    mid:
; VPLAN-NEXT:      EMIT ir<%gep.b> = getelementptr inbounds ir<%b>, ir<%iv>
; VPLAN-NEXT:      EMIT store ir<0>, ir<%gep.b>, ir<%c> (!vplan.execution.probability 4.657e-08%)
; VPLAN-NEXT:    Successor(s): if.then
; VPLAN-EMPTY:
; VPLAN-NEXT:    if.then:
; VPLAN:           EMIT vp<[[MASK:%.+]]> = logical-and ir<%c>, vp<%{{.+}}>
; VPLAN-NEXT:      EMIT vp<[[NOT_MASK:%.+]]> = not vp<[[MASK]]>
; VPLAN-NEXT:      EMIT vp<[[DEFAULT:%.+]]> = logical-and ir<%c>, vp<[[NOT_MASK]]>
; VPLAN-NEXT:      EMIT ir<%gep.a> = getelementptr inbounds ir<%a>, ir<%iv>
; VPLAN-NEXT:      EMIT store ir<1>, ir<%gep.a>, vp<[[MASK]]> (!vplan.execution.probability 4.657e-08%)
; VPLAN-NEXT:    Successor(s): latch
; VPLAN-EMPTY:
; VPLAN-NEXT:    latch:
; VPLAN-NEXT:      EMIT ir<%iv.next> = add ir<%iv>, ir<1>
; VPLAN-NEXT:      EMIT ir<%ec> = icmp eq ir<%iv.next>, ir<1024>
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
; %if.then's probability up to one; that would drop the recorded probability,
; because probability one needs no annotation.
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
; VPLAN:         vector.body:
; VPLAN-NEXT:      ir<%iv> = WIDEN-INDUCTION ir<0>, ir<1>, vp<%{{.+}}>
; VPLAN-NEXT:      EMIT ir<%gep.idx> = getelementptr inbounds ir<%idx>, ir<%iv>
; VPLAN-NEXT:      EMIT-SCALAR ir<%l> = load ir<%gep.idx>
; VPLAN-NEXT:    Successor(s): if.then
; VPLAN-EMPTY:
; VPLAN-NEXT:    if.then:
; VPLAN-NEXT:      EMIT vp<[[C0:%.+]]> = icmp eq ir<%l>, ir<0>
; VPLAN-NEXT:      EMIT vp<[[C1:%.+]]> = icmp eq ir<%l>, ir<1>
; VPLAN-NEXT:      EMIT vp<[[C2:%.+]]> = icmp eq ir<%l>, ir<2>
; VPLAN-NEXT:      EMIT vp<[[C3:%.+]]> = icmp eq ir<%l>, ir<3>
; VPLAN-NEXT:      EMIT vp<[[C4:%.+]]> = icmp eq ir<%l>, ir<4>
; VPLAN-NEXT:      EMIT vp<[[OR0:%.+]]> = or vp<[[C0]]>, vp<[[C1]]>
; VPLAN-NEXT:      EMIT vp<[[OR1:%.+]]> = or vp<[[OR0]]>, vp<[[C2]]>
; VPLAN-NEXT:      EMIT vp<[[OR2:%.+]]> = or vp<[[OR1]]>, vp<[[C3]]>
; VPLAN-NEXT:      EMIT vp<[[MASK:%.+]]> = or vp<[[OR2]]>, vp<[[C4]]>
; VPLAN-NEXT:      EMIT vp<{{.+}}> = not vp<[[MASK]]>
; VPLAN-NEXT:      EMIT ir<%gep.a> = getelementptr inbounds ir<%a>, ir<%iv>
; VPLAN-NEXT:      EMIT store ir<1>, ir<%gep.a>, vp<[[MASK]]> (!vplan.execution.probability 100%)
; VPLAN-NEXT:    Successor(s): latch
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
; VPLAN:         if.then:
; VPLAN:           EMIT store ir<1>, ir<%gep.a>, vp<{{.+}}> (!vplan.execution.probability 81.7%)
; VPLAN-NEXT:    Successor(s): latch
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
