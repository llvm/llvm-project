; REQUIRES: asserts

; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN: -force-tail-folding-style=data-with-evl \
; RUN: -prefer-predicate-over-epilogue=predicate-dont-vectorize \
; RUN: -mtriple=riscv64 -mattr=+v -riscv-v-vector-bits-max=128 -disable-output < %s 2>&1 | FileCheck --check-prefix=IF-EVL %s

define void @first_order_recurrence(ptr noalias %A, ptr noalias %B, i64 %TC) {
; IF-EVL: VPlan 'Initial VPlan for VF={1},UF>=1'
; IF-EVL-NOT: EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI
;
; IF-EVL: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF={1}' { 
; IF-EVL-NEXT: Live-in vp<[[VF:%[0-9]+]]> = VF
; IF-EVL-NEXT: Live-in vp<[[VFUF:%[0-9]+]]> = VF * UF
; IF-EVL-NEXT: Live-in vp<[[VTC:%[0-9]+]]> = vector-trip-count
; IF-EVL-NEXT: Live-in ir<%TC> = original trip-count
; IF-EVL-EMPTY:
; IF-EVL: ir-bb<entry>:
; IF-EVL-NEXT: Successor(s): vector.ph
; IF-EVL-EMPTY:
; IF-EVL: vector.ph:
; IF-EVL-NEXT:  SCALAR-CAST vp<[[VF32:%[0-9]+]]> = trunc vp<[[VF]]> to i32
; IF-EVL-NEXT: Successor(s): vector loop
; IF-EVL-EMPTY:
; IF-EVL: <x1> vector loop: {
; IF-EVL-NEXT:   vector.body:
; IF-EVL-NEXT:     EMIT vp<[[IV:%[0-9]+]]> = CANONICAL-INDUCTION
; IF-EVL-NEXT:     EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<[[EVL_PHI:%[0-9]+]]> = phi ir<0>, vp<[[IV_NEXT:%.+]]>
; IF-EVL-NEXT:     FIRST-ORDER-RECURRENCE-PHI ir<[[FOR_PHI:%.+]]> = phi ir<33>, ir<[[LD:%.+]]>
; IF-EVL-NEXT:     EMIT vp<[[PREV_EVL:%.+]]> = phi vp<[[VF32]]>, vp<[[EVL:%.+]]>
; IF-EVL-NEXT:     EMIT vp<[[AVL:%.+]]> = sub ir<%TC>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[EVL]]> = EXPLICIT-VECTOR-LENGTH vp<[[AVL]]>
; IF-EVL-NEXT:     vp<[[ST:%[0-9]+]]> = SCALAR-STEPS vp<[[EVL_PHI]]>, ir<1>
; IF-EVL-NEXT:     CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds nuw ir<%A>, vp<[[ST]]
; IF-EVL-NEXT:     vp<[[PTR1:%[0-9]+]]> = vector-pointer ir<[[GEP1]]>
; IF-EVL-NEXT:     WIDEN ir<[[LD]]> = vp.load vp<[[PTR1]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     WIDEN-INTRINSIC vp<[[SPLICE:%[0-9]+]]> = call llvm.experimental.vp.splice(ir<[[FOR_PHI]]>, ir<[[LD]]>, ir<-1>, ir<true>, vp<[[PREV_EVL]]>, vp<[[EVL]]>)
; IF-EVL-NEXT:     WIDEN ir<[[ADD:%.+]]> = add nsw vp<[[SPLICE]]>, ir<[[LD]]>
; IF-EVL-NEXT:     CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds nuw ir<%B>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR2:%[0-9]+]]> = vector-pointer ir<[[GEP2]]>
; IF-EVL-NEXT:     WIDEN vp.store vp<[[PTR2]]>, ir<[[ADD]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     SCALAR-CAST vp<[[CAST:%[0-9]+]]> = zext vp<[[EVL]]> to i64
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT]]> = add vp<[[CAST]]>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT_EXIT:%.+]]> = add vp<[[IV]]>, vp<[[VFUF]]>
; IF-EVL-NEXT:     EMIT branch-on-count  vp<[[IV_NEXT_EXIT]]>, vp<[[VTC]]>
; IF-EVL-NEXT:   No successors
; IF-EVL-NEXT: }
; IF-EVL-NEXT: Successor(s): middle.block
; IF-EVL-EMPTY:
; IF-EVL: middle.block:
; IF-EVL-NEXT:   EMIT vp<[[RESUME_EXTRACT:%.+]]> = extract-from-end ir<[[LD]]>, ir<1>
; IF-EVL-NEXT:   EMIT branch-on-cond ir<true>
; IF-EVL-NEXT: Successor(s): ir-bb<for.end>, scalar.ph
; IF-EVL: Cost of 0 for VF vscale x 4: FIRST-ORDER-RECURRENCE-PHI ir<[[FOR_PHI]]> = phi ir<33>, ir<[[LD]]>
; IF-EVL: Cost of 4 for VF vscale x 4: WIDEN-INTRINSIC vp<[[SPLICE]]> = call llvm.experimental.vp.splice(ir<[[FOR_PHI]]>, ir<[[LD]]>, ir<-1>, ir<true>, vp<[[PREV_EVL]]>, vp<[[EVL]]>)
entry:
  br label %for.body

for.body:
  %indvars = phi i64 [ 0, %entry ], [ %indvars.next, %for.body ]
  %for1 = phi i32 [ 33, %entry ], [ %0, %for.body ]
  %arrayidx = getelementptr inbounds nuw i32, ptr %A, i64 %indvars
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %for1, %0
  %arrayidx2 = getelementptr inbounds nuw i32, ptr %B, i64 %indvars
  store i32 %add, ptr %arrayidx2, align 4
  %indvars.next = add nuw nsw i64 %indvars, 1
  %exitcond.not = icmp eq i64 %indvars.next, %TC
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.enable", i1 true}
