; REQUIRES: asserts

; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN: -mtriple=riscv64 -mattr=+v -riscv-v-vector-bits-max=128 \
; RUN: -riscv-v-vector-bits-min=128 -disable-output < %s 2>&1 | FileCheck --check-prefix=FAST %s

define void @select_with_fastmath_flags(ptr noalias %a, ptr noalias %b, ptr noalias %c, i64 %N) {
; FAST: VPlan 'Final VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF={1}' {
; FAST-NEXT: ir<[[VFUF:%.+]]> = VF * UF
; FAST-NEXT: Live-in ir<[[VTC:%.+]]> = vector-trip-count
; FAST-NEXT: Live-in ir<%N> = original trip-count

; FAST: <x1> vector loop: {
; FAST-NEXT:   vector.body:
; FAST-NEXT:     SCALAR-PHI vp<[[IV:%.+]]> = phi ir<0>, vp<[[IV_NEXT_EXIT:%.+]]>
; FAST-NEXT:     vp<[[ST:%.+]]> = SCALAR-STEPS vp<[[IV]]>, ir<1>
; FAST-NEXT:     CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds nuw ir<%b>, vp<[[ST]]>
; FAST-NEXT:     vp<[[PTR1:%.+]]> = vector-pointer ir<[[GEP1]]>
; FAST-NEXT:     WIDEN ir<[[LD1:%.+]]> = load vp<[[PTR1]]>
; FAST-NEXT:     CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds nuw ir<%c>, vp<[[ST]]>
; FAST-NEXT:     vp<[[PTR2:%.+]]> = vector-pointer ir<[[GEP2]]>
; FAST-NEXT:     WIDEN ir<[[LD2:%.+]]> = load vp<[[PTR2]]>
; FAST-NEXT:     WIDEN ir<[[FCMP:%.+]]> = fcmp ogt ir<[[LD1]]>, ir<[[LD2]]>
; FAST-NEXT:     WIDEN ir<[[FADD:%.+]]> = fadd reassoc nnan ninf nsz arcp contract afn ir<[[LD1]]>, ir<1.000000e+01>
; FAST-NEXT:     WIDEN-SELECT ir<[[SELECT:%.+]]> = select reassoc nnan ninf nsz arcp contract afn ir<[[FCMP]]>, ir<[[FADD]]>, ir<[[LD2]]>
; FAST-NEXT:     CLONE ir<[[GEP3:%.+]]> = getelementptr inbounds nuw ir<%a>, vp<[[ST]]>
; FAST-NEXT:     vp<[[PTR3:%.+]]> = vector-pointer ir<[[GEP3]]>
; FAST-NEXT:     WIDEN store vp<[[PTR3]]>, ir<[[SELECT]]>
; FAST-NEXT:     EMIT vp<[[IV_NEXT_EXIT]]> = add nuw vp<[[IV]]>, ir<[[VFUF]]>
; FAST-NEXT:     EMIT branch-on-count vp<[[IV_NEXT_EXIT]]>, ir<%n.vec>
; FAST-NEXT:   No successors
; FAST-NEXT: }

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep = getelementptr inbounds nuw float, ptr %b, i64 %iv
  %0 = load float, ptr %gep, align 4
  %gep3 = getelementptr inbounds nuw float, ptr %c, i64 %iv
  %1 = load float, ptr %gep3, align 4
  %cmp4 = fcmp fast ogt float %0, %1
  %add = fadd fast float %0, 1.000000e+01
  %cond = select fast i1 %cmp4, float %add, float %1
  %gep11 = getelementptr inbounds nuw float, ptr %a, i64 %iv
  store float %cond, ptr %gep11, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret void
}
