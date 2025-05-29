; REQUIRES: asserts
; RUN: opt -p loop-vectorize -force-vector-width=2 -force-vector-interleave=1 -debug -disable-output %s 2>&1 | FileCheck %s

define void @switch4_default_common_dest_with_case(ptr %start, ptr %end) {
; CHECK:      VPlan 'Final VPlan for VF={2},UF={1}' {
; CHECK-NEXT: Live-in ir<[[VF:.+]]> = VF
; CHECK-NEXT: Live-in ir<[[VFxUF:.+]]> = VF * UF
; CHECK-NEXT: Live-in ir<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT: ir<%0> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<entry>:
; CHECK-NEXT: Successor(s): ir-bb<scalar.ph>, ir-bb<vector.ph>
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<vector.ph>:
; CHECK-NEXT:   IR %n.mod.vf = urem i64 %0, 2
; CHECK-NEXT:   IR %n.vec = sub i64 %0, %n.mod.vf
; CHECK-NEXT:   vp<[[END:%.+]]> = DERIVED-IV ir<%start> + ir<%n.vec> * ir<1>
; CHECK-NEXT: Successor(s): vector.body
; CHECK-EMPTY:
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = phi [ ir<0>, ir-bb<vector.ph> ], [ vp<[[CAN_IV_NEXT:%.+]]>, default.2 ]
; CHECK-NEXT:   vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>, ir<2>
; CHECK-NEXT:   EMIT vp<[[PTR:%.+]]> = ptradd ir<%start>, vp<[[STEPS]]>
; CHECK-NEXT:   vp<[[WIDE_PTR:%.+]]> = vector-pointer vp<[[PTR]]>
; CHECK-NEXT:   WIDEN ir<%l> = load vp<[[WIDE_PTR]]>
; CHECK-NEXT:   EMIT vp<[[C1:%.+]]> = icmp eq ir<%l>, ir<-12>
; CHECK-NEXT:   EMIT vp<[[C2:%.+]]> = icmp eq ir<%l>, ir<13>
; CHECK-NEXT:   EMIT vp<[[OR_CASES:%.+]]> = or vp<[[C1]]>, vp<[[C2]]>
; CHECK-NEXT:   EMIT vp<[[DEFAULT_MASK:%.+]]> = not vp<[[OR_CASES]]>
; CHECK-NEXT: Successor(s): pred.store
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.store: {
; CHECK-NEXT:   pred.store.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<[[C2]]>
; CHECK-NEXT:   Successor(s): pred.store.if, pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.if:
; CHECK-NEXT:     REPLICATE store ir<0>, vp<[[PTR]]>
; CHECK-NEXT:   Successor(s): pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.continue:
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): if.then.2.0
; CHECK-EMPTY:
; CHECK-NEXT: if.then.2.0:
; CHECK-NEXT: Successor(s): pred.store
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.store: {
; CHECK-NEXT:   pred.store.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<[[C1]]>
; CHECK-NEXT:   Successor(s): pred.store.if, pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:     pred.store.if:
; CHECK-NEXT:     REPLICATE store ir<42>, vp<[[PTR]]>
; CHECK-NEXT:   Successor(s): pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.continue:
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): if.then.1.1
; CHECK-EMPTY:
; CHECK-NEXT: if.then.1.1:
; CHECK-NEXT: Successor(s): pred.store
; CHECK-EMPTY:
; CHECK-NEXT: <xVFxUF> pred.store: {
; CHECK-NEXT:   pred.store.entry:
; CHECK-NEXT:     BRANCH-ON-MASK vp<[[DEFAULT_MASK]]>
; CHECK-NEXT:   Successor(s): pred.store.if, pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.if:
; CHECK-NEXT:     REPLICATE store ir<2>, vp<[[PTR]]>
; CHECK-NEXT:   Successor(s): pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:   pred.store.continue:
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): default.2
; CHECK-EMPTY:
; CHECK-NEXT: default.2:
; CHECK-NEXT:   EMIT vp<[[CAN_IV_NEXT]]> = add nuw vp<[[CAN_IV]]>, ir<[[VFxUF]]>
; CHECK-NEXT:   EMIT branch-on-count vp<[[CAN_IV_NEXT]]>, ir<[[VTC]]>
; CHECK-NEXT: Successor(s): middle.block, vector.body
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT vp<[[MIDDLE_CMP:%.+]]> = icmp eq ir<%0>, ir<[[VTC]]>
; CHECK-NEXT:   EMIT branch-on-cond vp<[[MIDDLE_CMP]]>
; CHECK-NEXT: Successor(s): ir-bb<exit>, ir-bb<scalar.ph>
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<exit>:
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<scalar.ph>:
; CHECK-NEXT:   EMIT vp<[[RESUME:%.+]]> = resume-phi vp<[[END]]>, ir<%start>
; CHECK-NEXT: Successor(s): ir-bb<loop.header>
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<loop.header>:
; CHECK-NEXT:   IR   %ptr.iv = phi ptr [ %start, %scalar.ph ], [ %ptr.iv.next, %loop.latch ] (extra operand: vp<[[RESUME]]> from ir-bb<scalar.ph>)
; CHECK-NEXT:   IR   %l = load i8, ptr %ptr.iv, align 1
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %loop.header

loop.header:
  %ptr.iv = phi ptr [ %start, %entry ], [ %ptr.iv.next, %loop.latch ]
  %l = load i8, ptr %ptr.iv, align 1
  switch i8 %l, label %default [
  i8 -12, label %if.then.1
  i8 13, label %if.then.2
  i8 0, label %default
  ]

if.then.1:
  store i8 42, ptr %ptr.iv, align 1
  br label %loop.latch

if.then.2:
  store i8 0, ptr %ptr.iv, align 1
  br label %loop.latch

default:
  store i8 2, ptr %ptr.iv, align 1
  br label %loop.latch

loop.latch:
  %ptr.iv.next = getelementptr inbounds i8, ptr %ptr.iv, i64 1
  %ec = icmp eq ptr %ptr.iv.next, %end
  br i1 %ec, label %exit, label %loop.header

exit:
  ret void
}
