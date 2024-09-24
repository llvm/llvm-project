; REQUIRES: asserts

; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN: -force-tail-folding-style=data-with-evl \
; RUN: -prefer-predicate-over-epilogue=predicate-dont-vectorize \
; RUN: -mtriple=riscv64 -mattr=+v -riscv-v-vector-bits-max=128 -disable-output < %s 2>&1 | FileCheck --check-prefix=IF-EVL %s

; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN: -force-tail-folding-style=none \
; RUN: -prefer-predicate-over-epilogue=predicate-else-scalar-epilogue \
; RUN: -mtriple=riscv64 -mattr=+v -riscv-v-vector-bits-max=128 -disable-output < %s 2>&1 | FileCheck --check-prefix=NO-VP %s

define void @vp_select(ptr noalias %a, ptr noalias %b, ptr noalias %c, i64 %N) {
; IF-EVL: VPlan 'Final VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF={1}' {
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
; IF-EVL-NEXT:     CLONE ir<%arrayidx3> = getelementptr inbounds ir<%c>, vp<%6>
; IF-EVL-NEXT:     vp<%8> = vector-pointer ir<%arrayidx3>
; IF-EVL-NEXT:     WIDEN ir<%1> = vp.load vp<%8>, vp<%5>
; IF-EVL-NEXT:     WIDEN ir<%cmp4> = icmp sgt ir<%0>, ir<%1>
; IF-EVL-NEXT:     WIDEN ir<%2> = vp.sub ir<0>, ir<%1>, vp<%5>
; IF-EVL-NEXT:     WIDEN-SELECT ir<%cond.p> = select ir<%cmp4>, ir<%1>, ir<%2>
; IF-EVL-NEXT:     WIDEN ir<%cond> = vp.add ir<%cond.p>, ir<%0>, vp<%5>
; IF-EVL-NEXT:     CLONE ir<%arrayidx15> = getelementptr inbounds ir<%a>, vp<%6>
; IF-EVL-NEXT:     vp<%9> = vector-pointer ir<%arrayidx15>
; IF-EVL-NEXT:     WIDEN vp.store vp<%9>, ir<%cond>, vp<%5>
; IF-EVL-NEXT:     SCALAR-CAST vp<%10> = zext vp<%5> to i64
; IF-EVL-NEXT:     EMIT vp<%11> = add vp<%10>, vp<%4>
; IF-EVL-NEXT:     EMIT vp<%12> = add vp<%3>, vp<%0>
; IF-EVL-NEXT:     EMIT branch-on-count vp<%12>, vp<%1>
; IF-EVL-NEXT:   No successors
; IF-EVL-NEXT: }

; NO-VP: VPlan 'Final VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF>=1' {
; NO-VP-NEXT: Live-in vp<%0> = VF * UF
; NO-VP-NEXT: Live-in vp<%1> = vector-trip-count
; NO-VP-NEXT: Live-in ir<%N> = original trip-count

; NO-VP: vector.ph:
; NO-VP-NEXT: Successor(s): vector loop

; NO-VP: <x1> vector loop: {
; NO-VP-NEXT:   vector.body:
; NO-VP-NEXT:     EMIT vp<%2> = CANONICAL-INDUCTION ir<0>, vp<%7>
; NO-VP-NEXT:     vp<%3> = SCALAR-STEPS vp<%2>, ir<1>
; NO-VP-NEXT:     CLONE ir<%arrayidx> = getelementptr inbounds ir<%b>, vp<%3>
; NO-VP-NEXT:     vp<%4> = vector-pointer ir<%arrayidx>
; NO-VP-NEXT:     WIDEN ir<%0> = load vp<%4>
; NO-VP-NEXT:     CLONE ir<%arrayidx3> = getelementptr inbounds ir<%c>, vp<%3>
; NO-VP-NEXT:     vp<%5> = vector-pointer ir<%arrayidx3>
; NO-VP-NEXT:     WIDEN ir<%1> = load vp<%5>
; NO-VP-NEXT:     WIDEN ir<%cmp4> = icmp sgt ir<%0>, ir<%1>
; NO-VP-NEXT:     WIDEN ir<%2> = sub ir<0>, ir<%1>
; NO-VP-NEXT:     WIDEN-SELECT ir<%cond.p> = select ir<%cmp4>, ir<%1>, ir<%2>
; NO-VP-NEXT:     WIDEN ir<%cond> = add ir<%cond.p>, ir<%0>
; NO-VP-NEXT:     CLONE ir<%arrayidx15> = getelementptr inbounds ir<%a>, vp<%3>
; NO-VP-NEXT:     vp<%6> = vector-pointer ir<%arrayidx15>
; NO-VP-NEXT:     WIDEN store vp<%6>, ir<%cond>
; NO-VP-NEXT:     EMIT vp<%7> = add nuw vp<%2>, vp<%0>
; NO-VP-NEXT:     EMIT branch-on-count vp<%7>, vp<%1>
; NO-VP-NEXT:   No successors
; NO-VP-NEXT: }


entry:
  %cmp30 = icmp sgt i64 %N, 0
  br i1 %cmp30, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx3 = getelementptr inbounds i32, ptr %c, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx3, align 4
  %cmp4 = icmp sgt i32 %0, %1
  %2 = sub i32 0, %1
  %cond.p = select i1 %cmp4, i32 %1, i32 %2
  %cond = add i32 %cond.p, %0
  %arrayidx15 = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  store i32 %cond, ptr %arrayidx15, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
