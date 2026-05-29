; RUN: opt -passes=loop-vectorize -force-vector-width=4 -force-vector-interleave=1 \
; RUN:     -vplan-print-after=simplifyRecipes -disable-output %s 2>&1 \
; RUN:   | FileCheck --strict-whitespace %s
; REQUIRES: asserts

; Tests that simplifyRecipes performs constant folding on VPlan0 (the initial
; VPlan, before VF specific transforms).

%rec8 = type { i16 }

@a = global [1 x %rec8] zeroinitializer
@b = global [2 x ptr] zeroinitializer

; %_tmp2 = getelementptr @a, 0, 0 folds to @a, so the store stores @a directly.
define void @f1() {
; CHECK-LABEL: VPlan for loop in 'f1' after VPlanTransforms::simplifyRecipes
; CHECK-NEXT:  VPlan ' for UF>=1' {
; CHECK-NEXT:  Live-in vp<[[VF:%[0-9]+]]> = VF
; CHECK-NEXT:  Live-in ir<2> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<bb1>:
; CHECK-NEXT:  Successor(s): scalar.ph, vector.ph
; CHECK-EMPTY:
; CHECK-NEXT:  vector.ph:
; CHECK-NEXT:  Successor(s): bb2
; CHECK-EMPTY:
; CHECK-NEXT:  bb2:
; CHECK-NEXT:    ir<%c.1.0> = WIDEN-INDUCTION nsw ir<0>, ir<1>, vp<[[VF]]>
; CHECK-NEXT:    EMIT-SCALAR ir<%_tmp1> = zext ir<0> to i64
; CHECK-NEXT:    EMIT ir<%_tmp2> = getelementptr ir<@a>, ir<0>, ir<0>
; CHECK-NEXT:    EMIT-SCALAR ir<%_tmp6> = sext ir<%c.1.0> to i64
; CHECK-NEXT:    EMIT ir<%_tmp7> = getelementptr ir<@b>, ir<0>, ir<%_tmp6>
; CHECK-NEXT:    EMIT store ir<@a>, ir<%_tmp7>
; CHECK-NEXT:    EMIT ir<%_tmp9> = add nsw ir<%c.1.0>, ir<1>
; CHECK-NEXT:    EMIT ir<%_tmp11> = icmp sge ir<%_tmp9>, ir<2>
; CHECK-NEXT:    EMIT vp<{{.+}}> = not ir<%_tmp11>
; CHECK-NEXT:    EMIT branch-on-cond ir<%_tmp11>
; CHECK-NEXT:  Successor(s): middle.block, bb2
;
bb1:
  br label %bb2

bb2:
  %c.1.0 = phi i16 [ 0, %bb1 ], [ %_tmp9, %bb2 ]
  %_tmp1 = zext i16 0 to i64
  %_tmp2 = getelementptr [1 x %rec8], ptr @a, i16 0, i64 %_tmp1
  %_tmp6 = sext i16 %c.1.0 to i64
  %_tmp7 = getelementptr [2 x ptr], ptr @b, i16 0, i64 %_tmp6
  store ptr %_tmp2, ptr %_tmp7
  %_tmp9 = add nsw i16 %c.1.0, 1
  %_tmp11 = icmp slt i16 %_tmp9, 2
  br i1 %_tmp11, label %bb2, label %bb3

bb3:
  ret void
}

; Test case for https://github.com/llvm/llvm-project/issues/131359.
; %or = or %cmp, true folds to true, simplifying the dependent select and
; branch-on-cond to use %c.1 directly.
define void @redundant_or_1(ptr %dst, i1 %c.0, i1 %c.1) {
; CHECK-LABEL: VPlan for loop in 'redundant_or_1' after VPlanTransforms::simplifyRecipes
; CHECK-NEXT:  VPlan ' for UF>=1' {
; CHECK-NEXT:  Live-in vp<[[VF:%[0-9]+]]> = VF
; CHECK-NEXT:  Live-in ir<3> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<entry>:
; CHECK-NEXT:  Successor(s): scalar.ph, vector.ph
; CHECK-EMPTY:
; CHECK-NEXT:  vector.ph:
; CHECK-NEXT:  Successor(s): loop.header
; CHECK-EMPTY:
; CHECK-NEXT:  loop.header:
; CHECK-NEXT:    ir<%iv> = WIDEN-INDUCTION nuw nsw ir<0>, ir<1>, vp<[[VF]]>
; CHECK-NEXT:    EMIT branch-on-cond ir<%c.0>
; CHECK-NEXT:  Successor(s): loop.latch, then.1
; CHECK-EMPTY:
; CHECK-NEXT:  then.1:
; CHECK-NEXT:    EMIT ir<%cmp> = icmp eq ir<%iv>, ir<2>
; CHECK-NEXT:    EMIT ir<%or> = or ir<%cmp>, ir<true>
; CHECK-NEXT:    EMIT ir<%cond> = select ir<true>, ir<%c.1>, ir<false>
; CHECK-NEXT:    EMIT branch-on-cond ir<%c.1>
; CHECK-NEXT:  Successor(s): then.2, loop.latch
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  br i1 %c.0, label %loop.latch, label %then.1

then.1:
  %cmp = icmp eq i32 %iv, 2
  %or = or i1 %cmp, true
  %cond = select i1 %or, i1 %c.1, i1 false
  br i1 %cond, label %then.2, label %loop.latch

then.2:
  %gep = getelementptr inbounds i32, ptr %dst, i32 %iv
  store i32 0, ptr %gep, align 4
  br label %loop.latch

loop.latch:
  %iv.next = add nuw nsw i32 %iv, 1
  %ec = icmp eq i32 %iv.next, 3
  br i1 %ec, label %exit, label %loop.header

exit:
  ret void
}

; %or = or true, %cmp folds to true (commuted form of redundant_or_1).
define void @redundant_or_2(ptr %dst, i1 %c.0, i1 %c.1) {
; CHECK-LABEL: VPlan for loop in 'redundant_or_2' after VPlanTransforms::simplifyRecipes
; CHECK-NEXT:  VPlan ' for UF>=1' {
; CHECK-NEXT:  Live-in vp<[[VF:%[0-9]+]]> = VF
; CHECK-NEXT:  Live-in ir<3> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<entry>:
; CHECK-NEXT:  Successor(s): scalar.ph, vector.ph
; CHECK-EMPTY:
; CHECK-NEXT:  vector.ph:
; CHECK-NEXT:  Successor(s): loop.header
; CHECK-EMPTY:
; CHECK-NEXT:  loop.header:
; CHECK-NEXT:    ir<%iv> = WIDEN-INDUCTION nuw nsw ir<0>, ir<1>, vp<[[VF]]>
; CHECK-NEXT:    EMIT branch-on-cond ir<%c.0>
; CHECK-NEXT:  Successor(s): loop.latch, then.1
; CHECK-EMPTY:
; CHECK-NEXT:  then.1:
; CHECK-NEXT:    EMIT ir<%cmp> = icmp eq ir<%iv>, ir<2>
; CHECK-NEXT:    EMIT ir<%or> = or ir<true>, ir<%cmp>
; CHECK-NEXT:    EMIT ir<%cond> = select ir<true>, ir<%c.1>, ir<false>
; CHECK-NEXT:    EMIT branch-on-cond ir<%c.1>
; CHECK-NEXT:  Successor(s): then.2, loop.latch
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  br i1 %c.0, label %loop.latch, label %then.1

then.1:
  %cmp = icmp eq i32 %iv, 2
  %or = or i1 true, %cmp
  %cond = select i1 %or, i1 %c.1, i1 false
  br i1 %cond, label %then.2, label %loop.latch

then.2:
  %gep = getelementptr inbounds i32, ptr %dst, i32 %iv
  store i32 0, ptr %gep, align 4
  br label %loop.latch

loop.latch:
  %iv.next = add nuw nsw i32 %iv, 1
  %ec = icmp eq i32 %iv.next, 3
  br i1 %ec, label %exit, label %loop.header

exit:
  ret void
}

; %or = or %cmp, false folds to %cmp, propagated into the dependent select.
define void @redundant_and_1(ptr %dst, i1 %c.0, i1 %c.1) {
; CHECK-LABEL: VPlan for loop in 'redundant_and_1' after VPlanTransforms::simplifyRecipes
; CHECK-NEXT:  VPlan ' for UF>=1' {
; CHECK-NEXT:  Live-in vp<[[VF:%[0-9]+]]> = VF
; CHECK-NEXT:  Live-in ir<3> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<entry>:
; CHECK-NEXT:  Successor(s): scalar.ph, vector.ph
; CHECK-EMPTY:
; CHECK-NEXT:  vector.ph:
; CHECK-NEXT:  Successor(s): loop.header
; CHECK-EMPTY:
; CHECK-NEXT:  loop.header:
; CHECK-NEXT:    ir<%iv> = WIDEN-INDUCTION nuw nsw ir<0>, ir<1>, vp<[[VF]]>
; CHECK-NEXT:    EMIT branch-on-cond ir<%c.0>
; CHECK-NEXT:  Successor(s): loop.latch, then.1
; CHECK-EMPTY:
; CHECK-NEXT:  then.1:
; CHECK-NEXT:    EMIT ir<%cmp> = icmp eq ir<%iv>, ir<2>
; CHECK-NEXT:    EMIT ir<%or> = or ir<%cmp>, ir<false>
; CHECK-NEXT:    EMIT ir<%cond> = select ir<%cmp>, ir<%c.1>, ir<false>
; CHECK-NEXT:    EMIT branch-on-cond ir<%cond>
; CHECK-NEXT:  Successor(s): then.2, loop.latch
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  br i1 %c.0, label %loop.latch, label %then.1

then.1:
  %cmp = icmp eq i32 %iv, 2
  %or = or i1 %cmp, false
  %cond = select i1 %or, i1 %c.1, i1 false
  br i1 %cond, label %then.2, label %loop.latch

then.2:
  %gep = getelementptr inbounds i32, ptr %dst, i32 %iv
  store i32 0, ptr %gep, align 4
  br label %loop.latch

loop.latch:
  %iv.next = add nuw nsw i32 %iv, 1
  %ec = icmp eq i32 %iv.next, 3
  br i1 %ec, label %exit, label %loop.header, !llvm.loop !1

exit:
  ret void
}

; %or = and false, %cmp folds to false, simplifying the dependent select and
; branch-on-cond to false directly.
define void @redundant_and_2(ptr %dst, i1 %c.0, i1 %c.1) {
; CHECK-LABEL: VPlan for loop in 'redundant_and_2' after VPlanTransforms::simplifyRecipes
; CHECK-NEXT:  VPlan ' for UF>=1' {
; CHECK-NEXT:  Live-in vp<[[VF:%[0-9]+]]> = VF
; CHECK-NEXT:  Live-in ir<3> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<entry>:
; CHECK-NEXT:  Successor(s): scalar.ph, vector.ph
; CHECK-EMPTY:
; CHECK-NEXT:  vector.ph:
; CHECK-NEXT:  Successor(s): loop.header
; CHECK-EMPTY:
; CHECK-NEXT:  loop.header:
; CHECK-NEXT:    ir<%iv> = WIDEN-INDUCTION nuw nsw ir<0>, ir<1>, vp<[[VF]]>
; CHECK-NEXT:    EMIT branch-on-cond ir<%c.0>
; CHECK-NEXT:  Successor(s): loop.latch, then.1
; CHECK-EMPTY:
; CHECK-NEXT:  then.1:
; CHECK-NEXT:    EMIT ir<%cmp> = icmp eq ir<%iv>, ir<2>
; CHECK-NEXT:    EMIT ir<%or> = and ir<false>, ir<%cmp>
; CHECK-NEXT:    EMIT ir<%cond> = select ir<false>, ir<%c.1>, ir<false>
; CHECK-NEXT:    EMIT branch-on-cond ir<false>
; CHECK-NEXT:  Successor(s): then.2, loop.latch
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  br i1 %c.0, label %loop.latch, label %then.1

then.1:
  %cmp = icmp eq i32 %iv, 2
  %or = and i1 false, %cmp
  %cond = select i1 %or, i1 %c.1, i1 false
  br i1 %cond, label %then.2, label %loop.latch

then.2:
  %gep = getelementptr inbounds i32, ptr %dst, i32 %iv
  store i32 0, ptr %gep, align 4
  br label %loop.latch

loop.latch:
  %iv.next = add nuw nsw i32 %iv, 1
  %ec = icmp eq i32 %iv.next, 3
  br i1 %ec, label %exit, label %loop.header

exit:
  ret void
}

!1 = distinct !{!1, !2, !3, !4}
!2 = !{!"llvm.loop.vectorize.width", i32 4}
!3 = !{!"llvm.loop.vectorize.enable", i1 true}
!4 = !{!"llvm.loop.vectorize.predicate.enable", i1 true}
