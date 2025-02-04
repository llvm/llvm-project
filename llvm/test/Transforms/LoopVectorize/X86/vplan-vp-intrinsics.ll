; REQUIRES: asserts

; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize -force-vector-width=4 \
; RUN: -force-tail-folding-style=data-with-evl \
; RUN: -prefer-predicate-over-epilogue=predicate-dont-vectorize \
; RUN: -mtriple=x86_64 -mattr=+avx512f -disable-output < %s 2>&1 | FileCheck --check-prefix=IF-EVL %s

; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize -force-vector-width=4 \
; RUN: -force-tail-folding-style=none \
; RUN: -prefer-predicate-over-epilogue=predicate-else-scalar-epilogue \
; RUN: -mtriple=x86_64 -mattr=+avx512f -disable-output < %s 2>&1 | FileCheck --check-prefix=NO-VP %s

define void @foo(ptr noalias %a, ptr noalias %b, ptr noalias %c, i64 %N) {
; IF-EVL: VPlan 'Initial VPlan for VF={4},UF>=1' {
; IF-EVL-NEXT: Live-in vp<[[VFUF:%[0-9]+]]> = VF * UF
; IF-EVL-NEXT: Live-in vp<[[VTC:%[0-9]+]]> = vector-trip-count
; IF-EVL-NEXT: Live-in vp<[[BETC:%[0-9]+]]> = backedge-taken count
; IF-EVL-NEXT: Live-in ir<%N> = original trip-count
; IF-EVL-EMPTY:
; IF-EVL:      vector.ph:
; IF-EVL-NEXT: Successor(s): vector loop
; IF-EVL-EMPTY:
; IF-EVL-NEXT: <x1> vector loop: {
; IF-EVL-NEXT:  vector.body:
; IF-EVL-NEXT:    EMIT vp<[[IV:%[0-9]+]]> = CANONICAL-INDUCTION
; IF-EVL-NEXT:    vp<[[ST:%[0-9]+]]>    = SCALAR-STEPS vp<[[IV]]>, ir<1>
; IF-EVL-NEXT:    EMIT vp<[[VIV:%[0-9]+]]> = WIDEN-CANONICAL-INDUCTION vp<[[IV]]>
; IF-EVL-NEXT:    EMIT vp<[[MASK:%[0-9]+]]> = icmp ule vp<[[VIV]]>, vp<[[BETC]]>
; IF-EVL-NEXT:    CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds ir<%b>, vp<[[ST]]>
; IF-EVL-NEXT:    vp<[[PTR1:%[0-9]+]]> = vector-pointer ir<[[GEP1]]>
; IF-EVL-NEXT:    WIDEN ir<[[LD1:%.+]]> = load vp<[[PTR1]]>, vp<[[MASK]]>
; IF-EVL-NEXT:    CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%c>, vp<[[ST]]>
; IF-EVL-NEXT:    vp<[[PTR2:%[0-9]+]]> = vector-pointer ir<[[GEP2]]>
; IF-EVL-NEXT:    WIDEN ir<[[LD2:%.+]]> = load vp<[[PTR2]]>, vp<[[MASK]]>
; IF-EVL-NEXT:    WIDEN ir<[[ADD:%.+]]> = add nsw ir<[[LD2]]>, ir<[[LD1]]>
; IF-EVL-NEXT:    CLONE ir<[[GEP3:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; IF-EVL-NEXT:    vp<[[PTR3:%[0-9]+]]> = vector-pointer ir<[[GEP3]]>
; IF-EVL-NEXT:    WIDEN store vp<[[PTR3]]>, ir<[[ADD]]>, vp<[[MASK]]>
; IF-EVL-NEXT:    EMIT vp<[[IV_NEXT:%.+]]> = add vp<[[IV]]>, vp<[[VFUF]]>
; IF-EVL-NEXT:    EMIT branch-on-count  vp<[[IV_NEXT]]>, vp<[[VTC]]>
; IF-EVL-NEXT:  No successors
; IF-EVL-NEXT: }

; NO-VP: VPlan 'Initial VPlan for VF={4},UF>=1' {
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
; NO-VP-NEXT:    EMIT vp<[[IV_NEXT:%.+]]> = add nuw vp<[[IV]]>, vp<[[VFUF]]>
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
