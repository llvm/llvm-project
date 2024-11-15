; REQUIRES: asserts

; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN: -force-tail-folding-style=data-with-evl \
; RUN: -prefer-predicate-over-epilogue=predicate-dont-vectorize \
; RUN: -mtriple=riscv64 -mattr=+v -riscv-v-vector-bits-max=128 -disable-output < %s 2>&1 | FileCheck --check-prefixes=IF-EVL,CHECK %s

; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN: -force-tail-folding-style=none \
; RUN: -prefer-predicate-over-epilogue=predicate-else-scalar-epilogue \
; RUN: -mtriple=riscv64 -mattr=+v -riscv-v-vector-bits-max=128 -disable-output < %s 2>&1 | FileCheck --check-prefixes=NO-VP,CHECK %s

define void @foo(ptr noalias %a, ptr noalias %b, ptr noalias %c, i64 %N) {
; IF-EVL: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF={1}' {
; IF-EVL-NEXT: Live-in vp<[[VFUF:%[0-9]+]]> = VF * UF
; IF-EVL-NEXT: Live-in vp<[[VTC:%[0-9]+]]> = vector-trip-count
; IF-EVL-NEXT: Live-in ir<%N> = original trip-count
; IF-EVL-EMPTY:
; IF-EVL:      vector.ph:
; IF-EVL-NEXT: Successor(s): vector loop
; IF-EVL-EMPTY:
; IF-EVL-NEXT: <x1> vector loop: {
; IF-EVL-NEXT:  vector.body:
; IF-EVL-NEXT:    EMIT vp<[[IV:%[0-9]+]]> = CANONICAL-INDUCTION
; IF-EVL-NEXT:    EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<[[EVL_PHI:%[0-9]+]]> = phi ir<0>, vp<[[IV_NEXT:%[0-9]+]]>
; IF-EVL-NEXT:    EMIT vp<[[AVL:%.+]]> = sub ir<%N>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:    EMIT vp<[[EVL:%.+]]> = EXPLICIT-VECTOR-LENGTH vp<[[AVL]]>
; IF-EVL-NEXT:    vp<[[ST:%[0-9]+]]> = SCALAR-STEPS vp<[[EVL_PHI]]>, ir<1>
; IF-EVL-NEXT:    CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds ir<%b>, vp<[[ST]]>
; IF-EVL-NEXT:    vp<[[PTR1:%[0-9]+]]> = vector-pointer ir<[[GEP1]]>
; IF-EVL-NEXT:    WIDEN ir<[[LD1:%.+]]> = vp.load vp<[[PTR1]]>, vp<[[EVL]]>
; IF-EVL-NEXT:    CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%c>, vp<[[ST]]>
; IF-EVL-NEXT:    vp<[[PTR2:%[0-9]+]]> = vector-pointer ir<[[GEP2]]>
; IF-EVL-NEXT:    WIDEN ir<[[LD2:%.+]]> = vp.load vp<[[PTR2]]>, vp<[[EVL]]>
; IF-EVL-NEXT:    WIDEN ir<[[ADD:%.+]]> = vp.add nsw ir<[[LD2]]>, ir<[[LD1]]>, vp<[[EVL]]>
; IF-EVL-NEXT:    CLONE ir<[[GEP3:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; IF-EVL-NEXT:    vp<[[PTR3:%[0-9]+]]> = vector-pointer ir<[[GEP3]]>
; IF-EVL-NEXT:    WIDEN vp.store vp<[[PTR3]]>, ir<[[ADD]]>, vp<[[EVL]]>
; IF-EVL-NEXT:    SCALAR-CAST vp<[[CAST:%[0-9]+]]> = zext vp<[[EVL]]> to i64
; IF-EVL-NEXT:    EMIT vp<[[IV_NEXT]]> = add vp<[[CAST]]>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:    EMIT vp<[[IV_NEXT_EXIT:%[0-9]+]]> = add vp<[[IV]]>, vp<[[VFUF]]>
; IF-EVL-NEXT:    EMIT branch-on-count  vp<[[IV_NEXT_EXIT]]>, vp<[[VTC]]>
; IF-EVL-NEXT:  No successors
; IF-EVL-NEXT: }

; NO-VP: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF>=1' {
; NO-VP-NEXT: Live-in vp<[[VFUF:%[0-9]+]]> = VF * UF
; NO-VP-NEXT: Live-in vp<[[VTC:%[0-9]+]]> = vector-trip-count
; NO-VP-NEXT: Live-in ir<%N> = original trip-count
; NO-VP-EMPTY:
; NO-VP:      vector.ph:
; NO-VP-NEXT: Successor(s): vector loop
; NO-VP-EMPTY:
; NO-VP-NEXT: <x1> vector loop: {
; NO-VP-NEXT:  vector.body:
; NO-VP-NEXT:    EMIT vp<[[IV:%[0-9]+]]> = CANONICAL-INDUCTION
; NO-VP-NEXT:    vp<[[ST:%[0-9]+]]>    = SCALAR-STEPS vp<[[IV]]>, ir<1>
; NO-VP-NEXT:    CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds ir<%b>, vp<[[ST]]>
; NO-VP-NEXT:    vp<[[PTR1:%[0-9]+]]> = vector-pointer ir<[[GEP1]]>
; NO-VP-NEXT:    WIDEN ir<[[LD1:%.+]]> = load vp<[[PTR1]]>
; NO-VP-NEXT:    CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%c>, vp<[[ST]]>
; NO-VP-NEXT:    vp<[[PTR2:%[0-9]+]]> = vector-pointer ir<[[GEP2]]>
; NO-VP-NEXT:    WIDEN ir<[[LD2:%.+]]> = load vp<[[PTR2]]>
; NO-VP-NEXT:    WIDEN ir<[[ADD:%.+]]> = add nsw ir<[[LD2]]>, ir<[[LD1]]>
; NO-VP-NEXT:    CLONE ir<[[GEP3:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; NO-VP-NEXT:    vp<[[PTR3:%[0-9]+]]> = vector-pointer ir<[[GEP3]]>
; NO-VP-NEXT:    WIDEN store vp<[[PTR3]]>, ir<[[ADD]]>
; NO-VP-NEXT:    EMIT vp<[[IV_NEXT:%[0-9]+]]> = add nuw vp<[[IV]]>, vp<[[VFUF]]>
; NO-VP-NEXT:    EMIT branch-on-count  vp<[[IV_NEXT]]>, vp<[[VTC]]>
; NO-VP-NEXT:  No successors
; NO-VP-NEXT: }

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %c, i64 %iv
  %1 = load i32, ptr %arrayidx2, align 4
  %add = add nsw i32 %1, %0
  %arrayidx4 = getelementptr inbounds i32, ptr %a, i64 %iv
  store i32 %add, ptr %arrayidx4, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void
}

define void @safe_dep(ptr %p) {
; CHECK: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VFUF:%[0-9]+]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VTC:%[0-9]+]]> = vector-trip-count
; CHECK-NEXT: Live-in ir<512> = original trip-count
; CHECK-EMPTY:
; CHECK:      vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:  vector.body:
; CHECK-NEXT:    EMIT vp<[[IV:%[0-9]+]]> = CANONICAL-INDUCTION
; CHECK-NEXT:    vp<[[ST:%[0-9]+]]>    = SCALAR-STEPS vp<[[IV]]>, ir<1>
; CHECK-NEXT:    CLONE ir<[[GEP1:%.+]]> = getelementptr ir<%p>, vp<[[ST]]>
; CHECK-NEXT:    vp<[[PTR1:%[0-9]+]]> = vector-pointer ir<[[GEP1]]>
; CHECK-NEXT:    WIDEN ir<[[V:%.+]]> = load vp<[[PTR1]]>
; CHECK-NEXT:    CLONE ir<[[OFFSET:.+]]> = add vp<[[ST]]>, ir<100>
; CHECK-NEXT:    CLONE ir<[[GEP2:%.+]]> = getelementptr ir<%p>, ir<[[OFFSET]]>
; CHECK-NEXT:    vp<[[PTR2:%[0-9]+]]> = vector-pointer ir<[[GEP2]]>
; CHECK-NEXT:    WIDEN store vp<[[PTR2]]>, ir<[[V]]>
; CHECK-NEXT:    EMIT vp<[[IV_NEXT:%[0-9]+]]> = add nuw vp<[[IV]]>, vp<[[VFUF]]>
; CHECK-NEXT:    EMIT branch-on-count  vp<[[IV_NEXT]]>, vp<[[VTC]]>
; CHECK-NEXT:  No successors
; CHECK-NEXT: }

entry:
  br label %loop

loop:
  %iv = phi i64 [0, %entry], [%iv.next, %loop]
  %a1 = getelementptr i64, ptr %p, i64 %iv
  %v = load i64, ptr %a1, align 32
  %offset = add i64 %iv, 100
  %a2 = getelementptr i64, ptr %p, i64 %offset
  store i64 %v, ptr %a2, align 32
  %iv.next = add i64 %iv, 1
  %cmp = icmp ne i64 %iv, 511
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

