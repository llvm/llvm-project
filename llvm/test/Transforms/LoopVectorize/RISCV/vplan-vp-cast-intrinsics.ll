; REQUIRES: asserts

; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN: -force-tail-folding-style=data-with-evl \
; RUN: -prefer-predicate-over-epilogue=predicate-dont-vectorize \
; RUN: -mtriple=riscv64 -mattr=+v -riscv-v-vector-bits-max=128 -disable-output < %s 2>&1 | FileCheck --check-prefix=IF-EVL %s

; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN: -force-tail-folding-style=none \
; RUN: -prefer-predicate-over-epilogue=predicate-else-scalar-epilogue \
; RUN: -mtriple=riscv64 -mattr=+v -riscv-v-vector-bits-max=128 -disable-output < %s 2>&1 | FileCheck --check-prefix=NO-VP %s

define void @vp_sext(ptr noalias %a, ptr noalias %b, i64 %N) {
; IF-EVL: VPlan 'Final VPlan for VF={vscale x 1,vscale x 2},UF={1}' {
; IF-EVL-NEXT: Live-in vp<%0> = VF * UF
; IF-EVL-NEXT: Live-in vp<%1> = vector-trip-count
; IF-EVL-NEXT: Live-in ir<%N> = original trip-count

; IF-EVL: vector.ph:
; IF-EVL-NEXT: Successor(s): vector loop

; IF-EVL: <x1> vector loop: {
; IF-EVL-NEXT:  vector.body:
; IF-EVL-NEXT:    EMIT vp<%3> = CANONICAL-INDUCTION ir<0>, vp<%12>
; IF-EVL-NEXT:    EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<%4> = phi ir<0>, vp<%11>
; IF-EVL-NEXT:    EMIT vp<%5> = EXPLICIT-VECTOR-LENGTH vp<%4>, ir<%N>
; IF-EVL-NEXT:    vp<%6> = SCALAR-STEPS vp<%4>, ir<1>
; IF-EVL-NEXT:    CLONE ir<%arrayidx> = getelementptr inbounds ir<%b>, vp<%6>
; IF-EVL-NEXT:    vp<%7> = vector-pointer ir<%arrayidx>
; IF-EVL-NEXT:    WIDEN ir<%0> = vp.load vp<%7>, vp<%5>
; IF-EVL-NEXT:    WIDEN-VP vp<%8> = vp.sext  ir<%0>, vp<%5> to i64
; IF-EVL-NEXT:    CLONE ir<%arrayidx4> = getelementptr inbounds ir<%a>, vp<%6>
; IF-EVL-NEXT:    vp<%9> = vector-pointer ir<%arrayidx4>
; IF-EVL-NEXT:    WIDEN vp.store vp<%9>, vp<%8>, vp<%5>
; IF-EVL-NEXT:    SCALAR-CAST vp<%10> = zext vp<%5> to i64
; IF-EVL-NEXT:    EMIT vp<%11> = add vp<%10>, vp<%4>
; IF-EVL-NEXT:    EMIT vp<%12> = add vp<%3>, vp<%0>
; IF-EVL-NEXT:    EMIT branch-on-count vp<%12>, vp<%1>
; IF-EVL-NEXT:  No successors
; IF-EVL-NEXT: }

; NO-VP: VPlan 'Final VPlan for VF={vscale x 1,vscale x 2},UF>=1' {
; NO-VP-NEXT: Live-in vp<%0> = VF * UF
; NO-VP-NEXT: Live-in vp<%1> = vector-trip-count
; NO-VP-NEXT: Live-in ir<%N> = original trip-count
 
; NO-VP: vector.ph:
; NO-VP-NEXT: Successor(s): vector loop

; NO-VP: <x1> vector loop: {
; NO-VP-NEXT:   vector.body:
; NO-VP-NEXT:     EMIT vp<%2> = CANONICAL-INDUCTION ir<0>, vp<%6>
; NO-VP-NEXT:     vp<%3> = SCALAR-STEPS vp<%2>, ir<1>
; NO-VP-NEXT:     CLONE ir<%arrayidx> = getelementptr inbounds ir<%b>, vp<%3>
; NO-VP-NEXT:     vp<%4> = vector-pointer ir<%arrayidx>
; NO-VP-NEXT:     WIDEN ir<%0> = load vp<%4>
; NO-VP-NEXT:     WIDEN-CAST ir<%conv2> = sext  ir<%0> to i64
; NO-VP-NEXT:     CLONE ir<%arrayidx4> = getelementptr inbounds ir<%a>, vp<%3>
; NO-VP-NEXT:     vp<%5> = vector-pointer ir<%arrayidx4>
; NO-VP-NEXT:     WIDEN store vp<%5>, ir<%conv2>
; NO-VP-NEXT:     EMIT vp<%6> = add nuw vp<%2>, vp<%0>
; NO-VP-NEXT:     EMIT branch-on-count vp<%6>, vp<%1>
; NO-VP-NEXT:   No successors
; NO-VP-NEXT: }

entry:
  %cmp8 = icmp sgt i64 %N, 0
  br i1 %cmp8, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %conv2 = sext i32 %0 to i64
  %arrayidx4 = getelementptr inbounds i64, ptr %a, i64 %indvars.iv
  store i64 %conv2, ptr %arrayidx4, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @vp_zext(ptr noalias %a, ptr noalias %b, i64 %N) {
; IF-EVL: VPlan 'Final VPlan for VF={vscale x 1,vscale x 2},UF={1}' {
; IF-EVL-NEXT: Live-in vp<%0> = VF * UF
; IF-EVL-NEXT: Live-in vp<%1> = vector-trip-count
; IF-EVL-NEXT: Live-in ir<%N> = original trip-count

; IF-EVL: vector.ph:
; IF-EVL-NEXT: Successor(s): vector loop

; IF-EVL: <x1> vector loop: {
; IF-EVL-NEXT:   vector.body:
; IF-EVL-NEXT:     EMIT vp<%3> = CANONICAL-INDUCTION ir<0>, vp<%12>
; IF-EVL-NEXT:     EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<%4> = phi ir<0>, vp<%11>
; IF-EVL-NEXT:     EMIT vp<%5> = EXPLICIT-VECTOR-LENGTH vp<%4>, ir<%N>
; IF-EVL-NEXT:     vp<%6> = SCALAR-STEPS vp<%4>, ir<1>
; IF-EVL-NEXT:     CLONE ir<%arrayidx> = getelementptr inbounds ir<%b>, vp<%6>
; IF-EVL-NEXT:     vp<%7> = vector-pointer ir<%arrayidx>
; IF-EVL-NEXT:     WIDEN ir<%0> = vp.load vp<%7>, vp<%5>
; IF-EVL-NEXT:     WIDEN-VP vp<%8> = vp.zext  ir<%0>, vp<%5> to i64
; IF-EVL-NEXT:     CLONE ir<%arrayidx4> = getelementptr inbounds ir<%a>, vp<%6>
; IF-EVL-NEXT:     vp<%9> = vector-pointer ir<%arrayidx4>
; IF-EVL-NEXT:     WIDEN vp.store vp<%9>, vp<%8>, vp<%5>
; IF-EVL-NEXT:     SCALAR-CAST vp<%10> = zext vp<%5> to i64
; IF-EVL-NEXT:     EMIT vp<%11> = add vp<%10>, vp<%4>
; IF-EVL-NEXT:     EMIT vp<%12> = add vp<%3>, vp<%0>
; IF-EVL-NEXT:     EMIT branch-on-count vp<%12>, vp<%1>
; IF-EVL-NEXT:   No successors
; IF-EVL-NEXT: }

; NO-VP: VPlan 'Final VPlan for VF={vscale x 1,vscale x 2},UF>=1' {
; NO-VP-NEXT: Live-in vp<%0> = VF * UF
; NO-VP-NEXT: Live-in vp<%1> = vector-trip-count
; NO-VP-NEXT: Live-in ir<%N> = original trip-count

; NO-VP: vector.ph:
; NO-VP-NEXT: Successor(s): vector loop

; NO-VP: <x1> vector loop: {
; NO-VP-NEXT:   vector.body:
; NO-VP-NEXT:     EMIT vp<%2> = CANONICAL-INDUCTION ir<0>, vp<%6>
; NO-VP-NEXT:     vp<%3> = SCALAR-STEPS vp<%2>, ir<1>
; NO-VP-NEXT:     CLONE ir<%arrayidx> = getelementptr inbounds ir<%b>, vp<%3>
; NO-VP-NEXT:     vp<%4> = vector-pointer ir<%arrayidx>
; NO-VP-NEXT:     WIDEN ir<%0> = load vp<%4>
; NO-VP-NEXT:     WIDEN-CAST ir<%conv2> = zext  ir<%0> to i64
; NO-VP-NEXT:     CLONE ir<%arrayidx4> = getelementptr inbounds ir<%a>, vp<%3>
; NO-VP-NEXT:     vp<%5> = vector-pointer ir<%arrayidx4>
; NO-VP-NEXT:     WIDEN store vp<%5>, ir<%conv2>
; NO-VP-NEXT:     EMIT vp<%6> = add nuw vp<%2>, vp<%0>
; NO-VP-NEXT:     EMIT branch-on-count vp<%6>, vp<%1>
; NO-VP-NEXT:   No successors
; NO-VP-NEXT: }

entry:
  %cmp8 = icmp sgt i64 %N, 0
  br i1 %cmp8, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %conv2 = zext i32 %0 to i64
  %arrayidx4 = getelementptr inbounds i64, ptr %a, i64 %indvars.iv
  store i64 %conv2, ptr %arrayidx4, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @vp_truncate(ptr noalias %a, ptr noalias %b, i64 %N) {
; IF-EVL : VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF={1}' {
; IF-EVL-NEXT : Live-in vp<%0> = VF * UF
; IF-EVL-NEXT : Live-in vp<%1> = vector-trip-count
; IF-EVL-NEXT : Live-in ir<%N> = original trip-count

; IF-EVL : vector.ph:
; IF-EVL-NEXT : Successor(s): vector loop

; IF-EVL : <x1> vector loop: {
; IF-EVL-NEXT :   vector.body:
; IF-EVL-NEXT :     EMIT vp<%3> = CANONICAL-INDUCTION ir<0>, vp<%12>
; IF-EVL-NEXT :     EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<%4> = phi ir<0>, vp<%11>
; IF-EVL-NEXT :     EMIT vp<%5> = EXPLICIT-VECTOR-LENGTH vp<%4>, ir<%N>
; IF-EVL-NEXT :     vp<%6> = SCALAR-STEPS vp<%4>, ir<1>
; IF-EVL-NEXT :     CLONE ir<%arrayidx> = getelementptr inbounds ir<%b>, vp<%6>
; IF-EVL-NEXT :     vp<%7> = vector-pointer ir<%arrayidx>
; IF-EVL-NEXT :     WIDEN ir<%0> = vp.load vp<%7>, vp<%5>
; IF-EVL-NEXT :     WIDEN-VP vp<%8> = vp.trunc  ir<%0>, vp<%5> to i16
; IF-EVL-NEXT :     CLONE ir<%arrayidx4> = getelementptr inbounds ir<%a>, vp<%6>
; IF-EVL-NEXT :     vp<%9> = vector-pointer ir<%arrayidx4>
; IF-EVL-NEXT :     WIDEN vp.store vp<%9>, vp<%8>, vp<%5>
; IF-EVL-NEXT :     SCALAR-CAST vp<%10> = zext vp<%5> to i64
; IF-EVL-NEXT :     EMIT vp<%11> = add vp<%10>, vp<%4>
; IF-EVL-NEXT :     EMIT vp<%12> = add vp<%3>, vp<%0>
; IF-EVL-NEXT :     EMIT branch-on-count vp<%12>, vp<%1>
; IF-EVL-NEXT :   No successors
; IF-EVL-NEXT : }

; NO-VP: Final VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF>=1' {
; NO-VP-NEXT: Live-in vp<%0> = VF * UF
; NO-VP-NEXT: Live-in vp<%1> = vector-trip-count
; NO-VP-NEXT: Live-in ir<%N> = original trip-count

; NO-VP: vector.ph:
; NO-VP-NEXT: Successor(s): vector loop

; NO-VP: <x1> vector loop: {
; NO-VP:   vector.body:
; NO-VP-NEXT:     EMIT vp<%2> = CANONICAL-INDUCTION ir<0>, vp<%6>
; NO-VP-NEXT:     vp<%3> = SCALAR-STEPS vp<%2>, ir<1>
; NO-VP-NEXT:     CLONE ir<%arrayidx> = getelementptr inbounds ir<%b>, vp<%3>
; NO-VP-NEXT:     vp<%4> = vector-pointer ir<%arrayidx>
; NO-VP-NEXT:     WIDEN ir<%0> = load vp<%4>
; NO-VP-NEXT:     WIDEN-CAST ir<%conv2> = trunc  ir<%0> to i16
; NO-VP-NEXT:     CLONE ir<%arrayidx4> = getelementptr inbounds ir<%a>, vp<%3>
; NO-VP-NEXT:     vp<%5> = vector-pointer ir<%arrayidx4>
; NO-VP-NEXT:     WIDEN store vp<%5>, ir<%conv2>
; NO-VP-NEXT:     EMIT vp<%6> = add nuw vp<%2>, vp<%0>
; NO-VP-NEXT:     EMIT branch-on-count vp<%6>, vp<%1>
; NO-VP-NEXT:   No successors
; NO-VP-NEXT: }

entry:
  %cmp8 = icmp sgt i64 %N, 0
  br i1 %cmp8, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %conv2 = trunc i32 %0 to i16
  %arrayidx4 = getelementptr inbounds i16, ptr %a, i64 %indvars.iv
  store i16 %conv2, ptr %arrayidx4, align 2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
