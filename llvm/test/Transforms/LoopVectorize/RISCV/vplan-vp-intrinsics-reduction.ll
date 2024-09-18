; REQUIRES: asserts

; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN: -force-tail-folding-style=data-with-evl \
; RUN: -prefer-predicate-over-epilogue=predicate-dont-vectorize \
; RUN: -mtriple=riscv64 -mattr=+v -riscv-v-vector-bits-max=128 -disable-output < %s 2>&1 | FileCheck --check-prefixes=IF-EVL-OUTLOOP %s

; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN: -prefer-inloop-reductions \
; RUN: -force-tail-folding-style=data-with-evl \
; RUN: -prefer-predicate-over-epilogue=predicate-dont-vectorize \
; RUN: -mtriple=riscv64 -mattr=+v -riscv-v-vector-bits-max=128 -disable-output < %s 2>&1 | FileCheck --check-prefixes=IF-EVL-INLOOP %s

; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN: -force-tail-folding-style=none \
; RUN: -prefer-predicate-over-epilogue=predicate-else-scalar-epilogue \
; RUN: -mtriple=riscv64 -mattr=+v -riscv-v-vector-bits-max=128 -disable-output < %s 2>&1 | FileCheck --check-prefixes=NO-VP-OUTLOOP %s

; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN: -prefer-inloop-reductions \
; RUN: -force-tail-folding-style=none \
; RUN: -prefer-predicate-over-epilogue=predicate-else-scalar-epilogue \
; RUN: -mtriple=riscv64 -mattr=+v -riscv-v-vector-bits-max=128 -disable-output < %s 2>&1 | FileCheck --check-prefixes=NO-VP-INLOOP %s


define i32 @reduction(ptr %a, i64 %n, i32 %start) {
; IF-EVL-OUTLOOP-NOT: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF={1}' {

; IF-EVL-INLOOP: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF={1}' {
; IF-EVL-INLOOP-NEXT: Live-in vp<[[VFUF:%[0-9]+]]> = VF * UF
; IF-EVL-INLOOP-NEXT: Live-in vp<[[VTC:%[0-9]+]]> = vector-trip-count
; IF-EVL-INLOOP-NEXT: Live-in ir<%n> = original trip-count
; IF-EVL-INLOOP-EMPTY:
; IF-EVL-INLOOP:      vector.ph:
; IF-EVL-INLOOP-NEXT: Successor(s): vector loop
; IF-EVL-INLOOP-EMPTY:
; IF-EVL-INLOOP-NEXT: <x1> vector loop: {
; IF-EVL-INLOOP-NEXT:  vector.body:
; IF-EVL-INLOOP-NEXT:    EMIT vp<[[IV:%[0-9]+]]> = CANONICAL-INDUCTION
; IF-EVL-INLOOP-NEXT:    EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<[[EVL_PHI:%[0-9]+]]> = phi ir<0>, vp<[[IV_NEXT:%[0-9]+]]>
; IF-EVL-INLOOP-NEXT:    WIDEN-REDUCTION-PHI ir<[[RDX_PHI:%.+]]> = phi ir<%start>, ir<[[RDX_NEXT:%.+]]>
; IF-EVL-INLOOP-NEXT:    EMIT vp<[[EVL:%.+]]> = EXPLICIT-VECTOR-LENGTH vp<[[EVL_PHI]]>, ir<%n>
; IF-EVL-INLOOP-NEXT:    vp<[[ST:%[0-9]+]]> = SCALAR-STEPS vp<[[EVL_PHI]]>, ir<1>
; IF-EVL-INLOOP-NEXT:    CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; IF-EVL-INLOOP-NEXT:    vp<[[PTR1:%[0-9]+]]> = vector-pointer ir<[[GEP1]]>
; IF-EVL-INLOOP-NEXT:    WIDEN ir<[[LD1:%.+]]> = vp.load vp<[[PTR1]]>, vp<[[EVL]]>
; IF-EVL-INLOOP-NEXT:    REDUCE ir<[[ADD:%.+]]> = ir<[[RDX_PHI]]> + vp.reduce.add (ir<[[LD1]]>, vp<[[EVL]]>)
; IF-EVL-INLOOP-NEXT:    SCALAR-CAST vp<[[CAST:%[0-9]+]]> = zext vp<[[EVL]]> to i64
; IF-EVL-INLOOP-NEXT:    EMIT vp<[[IV_NEXT]]> = add vp<[[CAST]]>, vp<[[EVL_PHI]]>
; IF-EVL-INLOOP-NEXT:    EMIT vp<[[IV_NEXT_EXIT:%[0-9]+]]> = add vp<[[IV]]>, vp<[[VFUF]]>
; IF-EVL-INLOOP-NEXT:    EMIT branch-on-count  vp<[[IV_NEXT_EXIT]]>, vp<[[VTC]]>
; IF-EVL-INLOOP-NEXT:  No successors
; IF-EVL-INLOOP-NEXT: }
; IF-EVL-INLOOP-NEXT: Successor(s): middle.block
; IF-EVL-INLOOP-EMPTY:
; IF-EVL-INLOOP-NEXT: middle.block:
; IF-EVL-INLOOP-NEXT:   EMIT vp<[[RDX:%.+]]> = compute-reduction-result ir<[[RDX_PHI]]>, ir<[[ADD]]>
; IF-EVL-INLOOP-NEXT:   EMIT vp<[[RDX_EX:%.+]]> = extract-from-end vp<[[RDX]]>, ir<1>
; IF-EVL-INLOOP-NEXT:   EMIT branch-on-cond ir<true>
; IF-EVL-INLOOP-NEXT: Successor(s): ir-bb<for.end>, scalar.ph
; IF-EVL-INLOOP-EMPTY:
; IF-EVL-INLOOP-NEXT: ir-bb<for.end>:
; IF-EVL-INLOOP-NEXT: No successors
; IF-EVL-INLOOP-EMPTY:
; IF-EVL-INLOOP-NEXT: scalar.ph:
; IF-EVL-INLOOP-NEXT: No successors
; IF-EVL-INLOOP-EMPTY:
; IF-EVL-INLOOP-NEXT: Live-out i32 %add.lcssa = vp<[[RDX_EX]]>
; IF-EVL-INLOOP-NEXT: }
;

; NO-VP-OUTLOOP: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF>=1' {
; NO-VP-OUTLOOP-NEXT: Live-in vp<[[VFUF:%[0-9]+]]> = VF * UF
; NO-VP-OUTLOOP-NEXT: Live-in vp<[[VTC:%[0-9]+]]> = vector-trip-count
; NO-VP-OUTLOOP-NEXT: Live-in ir<%n> = original trip-count
; NO-VP-OUTLOOP-EMPTY:
; NO-VP-OUTLOOP:      vector.ph:
; NO-VP-OUTLOOP-NEXT: Successor(s): vector loop
; NO-VP-OUTLOOP-EMPTY:
; NO-VP-OUTLOOP-NEXT: <x1> vector loop: {
; NO-VP-OUTLOOP-NEXT:  vector.body:
; NO-VP-OUTLOOP-NEXT:    EMIT vp<[[IV:%[0-9]+]]> = CANONICAL-INDUCTION
; NO-VP-OUTLOOP-NEXT:    WIDEN-REDUCTION-PHI ir<[[RDX_PHI:%.+]]> = phi ir<%start>, ir<[[RDX_NEXT:%.+]]>
; NO-VP-OUTLOOP-NEXT:    vp<[[ST:%[0-9]+]]> = SCALAR-STEPS vp<[[IV]]>, ir<1>
; NO-VP-OUTLOOP-NEXT:    CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; NO-VP-OUTLOOP-NEXT:    vp<[[PTR1:%[0-9]+]]> = vector-pointer ir<[[GEP1]]>
; NO-VP-OUTLOOP-NEXT:    WIDEN ir<[[LD1:%.+]]> = load vp<[[PTR1]]>
; NO-VP-OUTLOOP-NEXT:    WIDEN ir<[[ADD:%.+]]> = add ir<[[LD1]]>, ir<[[RDX_PHI]]>
; NO-VP-OUTLOOP-NEXT:    EMIT vp<[[IV_NEXT_EXIT:%[0-9]+]]> = add nuw vp<[[IV]]>, vp<[[VFUF]]>
; NO-VP-OUTLOOP-NEXT:    EMIT branch-on-count  vp<[[IV_NEXT_EXIT]]>, vp<[[VTC]]>
; NO-VP-OUTLOOP-NEXT:  No successors
; NO-VP-OUTLOOP-NEXT: }
; NO-VP-OUTLOOP-NEXT: Successor(s): middle.block
; NO-VP-OUTLOOP-EMPTY:
; NO-VP-OUTLOOP-NEXT: middle.block:
; NO-VP-OUTLOOP-NEXT:   EMIT vp<[[RDX:%.+]]> = compute-reduction-result ir<[[RDX_PHI]]>, ir<[[ADD]]>
; NO-VP-OUTLOOP-NEXT:   EMIT vp<[[RDX_EX:%.+]]> = extract-from-end vp<[[RDX]]>, ir<1>
; NO-VP-OUTLOOP-NEXT:   EMIT vp<[[BOC:%.+]]> = icmp eq ir<%n>, vp<[[VTC]]>
; NO-VP-OUTLOOP-NEXT:   EMIT branch-on-cond vp<[[BOC]]>
; NO-VP-OUTLOOP-NEXT: Successor(s): ir-bb<for.end>, scalar.ph
; NO-VP-OUTLOOP-EMPTY:
; NO-VP-OUTLOOP-NEXT: ir-bb<for.end>:
; NO-VP-OUTLOOP-NEXT: No successors
; NO-VP-OUTLOOP-EMPTY:
; NO-VP-OUTLOOP-NEXT: scalar.ph:
; NO-VP-OUTLOOP-NEXT: No successors
; NO-VP-OUTLOOP-EMPTY:
; NO-VP-OUTLOOP-NEXT: Live-out i32 %add.lcssa = vp<[[RDX_EX]]>
; NO-VP-OUTLOOP-NEXT: }
;

; NO-VP-INLOOP: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF>=1' {
; NO-VP-INLOOP-NEXT: Live-in vp<[[VFUF:%[0-9]+]]> = VF * UF
; NO-VP-INLOOP-NEXT: Live-in vp<[[VTC:%[0-9]+]]> = vector-trip-count
; NO-VP-INLOOP-NEXT: Live-in ir<%n> = original trip-count
; NO-VP-INLOOP-EMPTY:
; NO-VP-INLOOP:      vector.ph:
; NO-VP-INLOOP-NEXT: Successor(s): vector loop
; NO-VP-INLOOP-EMPTY:
; NO-VP-INLOOP-NEXT: <x1> vector loop: {
; NO-VP-INLOOP-NEXT:  vector.body:
; NO-VP-INLOOP-NEXT:    EMIT vp<[[IV:%[0-9]+]]> = CANONICAL-INDUCTION
; NO-VP-INLOOP-NEXT:    WIDEN-REDUCTION-PHI ir<[[RDX_PHI:%.+]]> = phi ir<%start>, ir<[[RDX_NEXT:%.+]]>
; NO-VP-INLOOP-NEXT:    vp<[[ST:%[0-9]+]]> = SCALAR-STEPS vp<[[IV]]>, ir<1>
; NO-VP-INLOOP-NEXT:    CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; NO-VP-INLOOP-NEXT:    vp<[[PTR1:%[0-9]+]]> = vector-pointer ir<[[GEP1]]>
; NO-VP-INLOOP-NEXT:    WIDEN ir<[[LD1:%.+]]> = load vp<[[PTR1]]>
; NO-VP-INLOOP-NEXT:    REDUCE ir<[[ADD:%.+]]> = ir<[[RDX_PHI]]> + reduce.add (ir<[[LD1]]>)
; NO-VP-INLOOP-NEXT:    EMIT vp<[[IV_NEXT_EXIT:%[0-9]+]]> = add nuw vp<[[IV]]>, vp<[[VFUF]]>
; NO-VP-INLOOP-NEXT:    EMIT branch-on-count  vp<[[IV_NEXT_EXIT]]>, vp<[[VTC]]>
; NO-VP-INLOOP-NEXT:  No successors
; NO-VP-INLOOP-NEXT: }
; NO-VP-INLOOP-NEXT: Successor(s): middle.block
; NO-VP-INLOOP-EMPTY:
; NO-VP-INLOOP-NEXT: middle.block:
; NO-VP-INLOOP-NEXT:   EMIT vp<[[RDX:%.+]]> = compute-reduction-result ir<[[RDX_PHI]]>, ir<[[ADD]]>
; NO-VP-INLOOP-NEXT:   EMIT vp<[[RDX_EX:%.+]]> = extract-from-end vp<[[RDX]]>, ir<1>
; NO-VP-INLOOP-NEXT:   EMIT vp<[[BOC:%.+]]> = icmp eq ir<%n>, vp<[[VTC]]>
; NO-VP-INLOOP-NEXT:   EMIT branch-on-cond vp<[[BOC]]>
; NO-VP-INLOOP-NEXT: Successor(s): ir-bb<for.end>, scalar.ph
; NO-VP-INLOOP-EMPTY:
; NO-VP-INLOOP-NEXT: ir-bb<for.end>:
; NO-VP-INLOOP-NEXT: No successors
; NO-VP-INLOOP-EMPTY:
; NO-VP-INLOOP-NEXT: scalar.ph:
; NO-VP-INLOOP-NEXT: No successors
; NO-VP-INLOOP-EMPTY:
; NO-VP-INLOOP-NEXT: Live-out i32 %add.lcssa = vp<[[RDX_EX]]>
; NO-VP-INLOOP-NEXT: }
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %rdx = phi i32 [ %start, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, %rdx
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:
  ret i32 %add
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.enable", i1 true}
