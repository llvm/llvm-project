; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN: -force-tail-folding-style=data-with-evl \
; RUN: -prefer-predicate-over-epilogue=predicate-dont-vectorize \
; RUN: -mtriple=riscv64 -mattr=+v -disable-output < %s 2>&1 | FileCheck %s

; REQUIRES: asserts

define void @test_wide_integer_induction(ptr noalias %a, i64 %N) {
; CHECK: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2},UF={1}' {
; CHECK-NEXT: Live-in vp<%0> = VF * UF
; CHECK-NEXT: Live-in vp<%1> = vector-trip-count
; CHECK-NEXT: Live-in ir<%N> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT:   WIDEN-INTRINSIC vp<%3> = callllvm.stepvector()
; CHECK-NEXT:   EMIT vp<%4> = mul vp<%3>, ir<1>
; CHECK-NEXT:   EMIT vp<%induction> = add ir<0>, vp<%4>
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT vp<%5> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; CHECK-NEXT:     EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<%6> = phi ir<0>, vp<%index.evl.next>
; CHECK-NEXT:     WIDEN-PHI ir<%iv> = phi vp<%induction>, vp<%14>
; CHECK-NEXT:     EMIT vp<%avl> = sub ir<%N>, vp<%6>
; CHECK-NEXT:     EMIT vp<%7> = EXPLICIT-VECTOR-LENGTH vp<%avl>
; CHECK-NEXT:     vp<%8> = SCALAR-STEPS vp<%6>, ir<1>
; CHECK-NEXT:     CLONE ir<%arrayidx> = getelementptr inbounds ir<%a>, vp<%8>
; CHECK-NEXT:     vp<%9> = vector-pointer ir<%arrayidx>
; CHECK-NEXT:     WIDEN vp.store vp<%9>, ir<%iv>, vp<%7>
; CHECK-NEXT:     SCALAR-CAST vp<%10> = zext vp<%7> to i64
; CHECK-NEXT:     EMIT vp<%index.evl.next> = add vp<%10>, vp<%6>
; CHECK-NEXT:     SCALAR-CAST vp<%11> = zext vp<%7> to i64
; CHECK-NEXT:     EMIT vp<%12> = mul ir<1>, vp<%11>
; CHECK-NEXT:     WIDEN-INTRINSIC vp<%13> = call llvm.experimental.vp.splat(vp<%12>, ir<true>, vp<%7>)
; CHECK-NEXT:     WIDEN-INTRINSIC vp<%14> = call llvm.vp.add(ir<%iv>, vp<%13>, ir<true>, vp<%7>)
; CHECK-NEXT:     EMIT vp<%index.next> = add vp<%5>, vp<%0>
; CHECK-NEXT:     EMIT branch-on-count vp<%index.next>, vp<%1>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT branch-on-cond ir<true>
; CHECK-NEXT: Successor(s): ir-bb<for.cond.cleanup>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<for.cond.cleanup>:
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph:
; CHECK-NEXT: Successor(s): ir-bb<for.body>
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<for.body>:
; CHECK-NEXT:   IR   %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
; CHECK-NEXT:   IR   %arrayidx = getelementptr inbounds i64, ptr %a, i64 %iv
; CHECK-NEXT:   IR   store i64 %iv, ptr %arrayidx, align 8
; CHECK-NEXT:   IR   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   IR   %exitcond.not = icmp eq i64 %iv.next, %N
; CHECK-NEXT: No successors
; CHECK-NEXT: }
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %iv
  store i64 %iv, ptr %arrayidx, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void
}

define void @test_wide_fp_induction(ptr noalias %a, i64 %N) {
; CHECK: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF={1}' {
; CHECK-NEXT: Live-in vp<%0> = VF * UF
; CHECK-NEXT: Live-in vp<%1> = vector-trip-count
; CHECK-NEXT: Live-in ir<%N> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT:   WIDEN-INTRINSIC vp<%3> = callllvm.stepvector()
; CHECK-NEXT:   WIDEN-CAST vp<%4> = uitofp  vp<%3> to float
; CHECK-NEXT:   EMIT vp<%5> = fmul reassoc nnan ninf nsz arcp contract afn vp<%4>, ir<3.000000e+00>
; CHECK-NEXT:   EMIT vp<%induction> = fsub reassoc nnan ninf nsz arcp contract afn ir<0.000000e+00>, vp<%5>
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT vp<%6> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; CHECK-NEXT:     EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<%7> = phi ir<0>, vp<%index.evl.next>
; CHECK-NEXT:     WIDEN-PHI ir<%f> = phi vp<%induction>, vp<%15>
; CHECK-NEXT:     EMIT vp<%avl> = sub ir<%N>, vp<%7>
; CHECK-NEXT:     EMIT vp<%8> = EXPLICIT-VECTOR-LENGTH vp<%avl>
; CHECK-NEXT:     vp<%9> = SCALAR-STEPS vp<%7>, ir<1>
; CHECK-NEXT:     CLONE ir<%arrayidx> = getelementptr inbounds ir<%a>, vp<%9>
; CHECK-NEXT:     vp<%10> = vector-pointer ir<%arrayidx>
; CHECK-NEXT:     WIDEN vp.store vp<%10>, ir<%f>, vp<%8>
; CHECK-NEXT:     SCALAR-CAST vp<%11> = zext vp<%8> to i64
; CHECK-NEXT:     EMIT vp<%index.evl.next> = add vp<%11>, vp<%7>
; CHECK-NEXT:     SCALAR-CAST vp<%12> = uitofp vp<%8> to float
; CHECK-NEXT:     EMIT vp<%13> = fmul reassoc nnan ninf nsz arcp contract afn ir<3.000000e+00>, vp<%12>
; CHECK-NEXT:     WIDEN-INTRINSIC vp<%14> = call llvm.experimental.vp.splat(vp<%13>, ir<true>, vp<%8>)
; CHECK-NEXT:     WIDEN-INTRINSIC vp<%15> = call llvm.vp.fsub(ir<%f>, vp<%14>, ir<true>, vp<%8>)
; CHECK-NEXT:     EMIT vp<%index.next> = add vp<%6>, vp<%0>
; CHECK-NEXT:     EMIT branch-on-count vp<%index.next>, vp<%1>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT branch-on-cond ir<true>
; CHECK-NEXT: Successor(s): ir-bb<for.cond.cleanup>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<for.cond.cleanup>:
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph:
; CHECK-NEXT: Successor(s): ir-bb<for.body>
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<for.body>:
; CHECK-NEXT:   IR   %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
; CHECK-NEXT:   IR   %f = phi float [ %f.next, %for.body ], [ 0.000000e+00, %entry ]
; CHECK-NEXT:   IR   %arrayidx = getelementptr inbounds float, ptr %a, i64 %iv
; CHECK-NEXT:   IR   store float %f, ptr %arrayidx, align 4
; CHECK-NEXT:   IR   %f.next = fsub fast float %f, 3.000000e+00
; CHECK-NEXT:   IR   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   IR   %exitcond.not = icmp eq i64 %iv.next, %N
; CHECK-NEXT: No successors
; CHECK-NEXT: }
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %f = phi float [ %f.next, %for.body ], [ 0.000000e+00, %entry ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %iv
  store float %f, ptr %arrayidx, align 4
  %f.next = fsub fast float %f, 3.000000e+00
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void
}
