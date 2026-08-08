; RUN: opt -passes='print<block-freq>' -disable-output %s 2>&1 \
; RUN:   | FileCheck --check-prefix=BFI %s
; RUN: opt -passes=loop-vectorize -force-vector-width=2 -force-vector-interleave=1 \
; RUN:     -vplan-print-after=introduceMasksAndLinearize -disable-output %s 2>&1 \
; RUN:   | FileCheck --check-prefix=VPLAN %s

; Check that the branch weights VPlan puts on the masked recipes of a block
; describe the same execution probability BlockFrequencyInfo computes for the
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
; VPLAN-NEXT:      EMIT store ir<%i>, ir<%gep.a>, ir<%c.0> (!prof {1, 3})
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
; VPLAN-NEXT:      EMIT store ir<%i>, ir<%gep.c>, vp<[[NOT_C0]]> (!prof {3, 1})
; VPLAN-NEXT:      EMIT ir<%c.1> = icmp slt ir<%i>, ir<-100>, vp<[[NOT_C0]]> (!prof {3, 1})
; VPLAN-NEXT:    Successor(s): then
; VPLAN-EMPTY:
; VPLAN-NEXT:    then:
; VPLAN-NEXT:      EMIT ir<%gep.a> = getelementptr inbounds ir<%a>, ir<%iv>
; VPLAN-NEXT:      EMIT store ir<%i>, ir<%gep.a>, ir<%c.0> (!prof {1, 3})
; VPLAN-NEXT:    Successor(s): merge
; VPLAN-EMPTY:
; VPLAN-NEXT:    merge:
; VPLAN-NEXT:      EMIT vp<[[AND:%.+]]> = logical-and vp<[[NOT_C0]]>, ir<%c.1>
; VPLAN-NEXT:      EMIT vp<[[MASK:%.+]]> = or vp<[[AND]]>, ir<%c.0>
; VPLAN-NEXT:      EMIT ir<%gep.b> = getelementptr inbounds ir<%b>, ir<%iv>
; VPLAN-NEXT:      EMIT store ir<%i>, ir<%gep.b>, vp<[[MASK]]> (!prof {1, 1})
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
; VPLAN-NEXT:      EMIT store ir<%i>, ir<%gep.b>, ir<%c.0> (!prof {1, 3})
; VPLAN-NEXT:      EMIT ir<%c.1> = icmp slt ir<%i>, ir<100>, ir<%c.0> (!prof {1, 3})
; VPLAN-NEXT:    Successor(s): if.1
; VPLAN-EMPTY:
; VPLAN-NEXT:    if.1:
; VPLAN-NEXT:      EMIT vp<[[MASK:%.+]]> = logical-and ir<%c.0>, ir<%c.1>
; VPLAN-NEXT:      EMIT ir<%gep.a> = getelementptr inbounds ir<%a>, ir<%iv>
; VPLAN-NEXT:      EMIT store ir<%i>, ir<%gep.a>, vp<[[MASK]]> (!prof {1, 7})
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
; VPLAN-NEXT:      EMIT store ir<2>, ir<%gep.b>, vp<[[C2]]> (!prof {1, 7})
; VPLAN-NEXT:    Successor(s): if.then
; VPLAN-EMPTY:
; VPLAN-NEXT:    if.then:
; VPLAN-NEXT:      EMIT ir<%gep.a> = getelementptr inbounds ir<%a>, ir<%iv>
; VPLAN-NEXT:      EMIT store ir<1>, ir<%gep.a>, vp<[[C0_OR_C1]]> (!prof {3, 5})
; VPLAN-NEXT:    Successor(s): default
; VPLAN-EMPTY:
; VPLAN-NEXT:    default:
; VPLAN-NEXT:      EMIT ir<%gep.c> = getelementptr inbounds ir<%c>, ir<%iv>
; VPLAN-NEXT:      EMIT store ir<0>, ir<%gep.c>, vp<[[DEFAULT]]> (!prof {1, 1})
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

!0 = !{!"branch_weights", i32 1, i32 3}
!1 = !{!"branch_weights", i32 1, i32 2}
!2 = !{!"branch_weights", i32 1, i32 1}
!3 = !{!"branch_weights", i32 1, i32 999}
!4 = !{!"branch_weights", i32 500, i32 125, i32 250, i32 125}
