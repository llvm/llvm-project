; REQUIRES: asserts
; RUN: opt < %s -passes=loop-vectorize,instcombine -force-vector-width=4 -force-vector-interleave=1 -debug-only=loop-vectorize -disable-output -print-after=instcombine 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"

; CHECK-LABEL: more_than_one_use
;
; PR30627. Check that a compare instruction with more than one use is not
; recognized as uniform and is vectorized.
;
; CHECK-NOT: Found uniform instruction: %cond = icmp slt i64 %i.next, %n
; CHECK:     vector.body
; CHECK:       %[[I:.+]] = add nuw nsw <4 x i64> %vec.ind, <i64 1, i64 1, i64 1, i64 1>
; CHECK:       icmp slt <4 x i64> %[[I]], %broadcast.splat
; CHECK:       br i1 {{.*}}, label %middle.block, label %vector.body
;
define i32 @more_than_one_use(ptr %a, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %r = phi i32 [ %tmp3, %for.body ], [ 0, %entry ]
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  %tmp0 = select i1 %cond, i64 %i.next, i64 0
  %tmp1 = getelementptr inbounds i32, ptr %a, i64 %tmp0
  %tmp2 = load i32, ptr %tmp1, align 8
  %tmp3 = add i32 %r, %tmp2
  br i1 %cond, label %for.body, label %for.end

for.end:
  %tmp4 = phi i32 [ %tmp3, %for.body ]
  ret i32 %tmp4
}

; Check for crash exposed by D76992.
; CHECK-LABEL: 'test'
; CHECK:      VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; CHECK-NEXT: Live-in vp<[[BTC:%.+]]> = backedge-taken count
; CHECK-NEXT: Live-in ir<14> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; CHECK-NEXT:   WIDEN-INDUCTION %iv = phi 0, %iv.next, ir<1>
; CHECK-NEXT:   EMIT vp<[[COND:%.+]]> = icmp ule ir<%iv> vp<[[BTC]]>
; CHECK-NEXT:   WIDEN ir<%cond0> = icmp ult ir<%iv>, ir<13>
; CHECK-NEXT:   WIDEN-SELECT ir<%s> = select ir<%cond0>, ir<10>, ir<20>
; CHECK-NEXT: Successor(s): pred.store
; CHECK-EMPTY:
; CHECK-NEXT:  <xVFxUF> pred.store: {
; CHECK-NEXT:    pred.store.entry:
; CHECK-NEXT:      BRANCH-ON-MASK vp<[[COND]]>
; CHECK-NEXT:    Successor(s): pred.store.if, pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:    pred.store.if:
; CHECK-NEXT:      vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:      REPLICATE ir<%gep> = getelementptr inbounds ir<%ptr>, vp<[[STEPS]]>
; CHECK-NEXT:      REPLICATE store ir<%s>, ir<%gep>
; CHECK-NEXT:    Successor(s): pred.store.continue
; CHECK-EMPTY:
; CHECK-NEXT:    pred.store.continue:
; CHECK-NEXT:    No successors
; CHECK-NEXT:  }
; CHECK-NEXT:  Successor(s): loop.0
; CHECK-EMPTY:
; CHECK-NEXT: loop.0:
; CHECK-NEXT:   EMIT vp<[[CAN_IV_NEXT:%.+]]> = VF * UF + vp<[[CAN_IV]]>
; CHECK-NEXT:   EMIT branch-on-count vp<[[CAN_IV_NEXT]]> vp<[[VEC_TC]]>
; CHECK-NEXT: No successor
; CHECK-NEXT: }
define void @test(ptr %ptr) {
entry:
  br label %loop

loop:                       ; preds = %loop, %entry
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %cond0 = icmp ult i64 %iv, 13
  %s = select i1 %cond0, i32 10, i32 20
  %gep = getelementptr inbounds i32, ptr %ptr, i64 %iv
  store i32 %s, ptr %gep
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 14
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}
