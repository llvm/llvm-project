; REQUIRES: asserts

; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN: -force-tail-folding-style=data-with-evl \
; RUN: -prefer-predicate-over-epilogue=predicate-dont-vectorize \
; RUN: -mtriple=riscv64 -mattr=+v -riscv-v-vector-bits-max=128 -disable-output < %s 2>&1 | FileCheck --check-prefix=IF-EVL %s
define void @vp_select(ptr noalias %a, ptr noalias %b, ptr noalias %c, i64 %N) {
; IF-EVL: VPlan 'Final VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF={1}' {
; IF-EVL-NEXT: Live-in ir<[[VFUF:%.+]]> = VF * UF
; IF-EVL-NEXT: Live-in ir<[[VTC:%.+]]> = vector-trip-count
; IF-EVL-NEXT: Live-in ir<%N> = original trip-count
; IF-EVL: vector.ph:
; IF-EVL-NEXT: Successor(s): vector loop
; IF-EVL: <x1> vector loop: {
; IF-EVL-NEXT:   vector.body:
; IF-EVL-NEXT:     SCALAR-PHI vp<[[IV:%[0-9]+]]> = phi ir<0>, vp<[[IV_NEXT_EXIT:%.+]]>
; IF-EVL-NEXT:     SCALAR-PHI vp<[[EVL_PHI:%[0-9]+]]>  = phi ir<0>, vp<[[IV_NEXT:%.+]]>
; IF-EVL-NEXT:     EMIT vp<[[AVL:%.+]]> = sub ir<%N>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[EVL:%.+]]> = EXPLICIT-VECTOR-LENGTH vp<[[AVL]]>
; IF-EVL-NEXT:     vp<[[ST:%[0-9]+]]> = SCALAR-STEPS vp<[[EVL_PHI]]>, ir<1>
; IF-EVL-NEXT:     CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds ir<%b>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR1:%[0-9]+]]> = vector-pointer ir<[[GEP1]]>
; IF-EVL-NEXT:     WIDEN ir<[[LD1:%.+]]> = vp.load vp<[[PTR1]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%c>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR2:%[0-9]+]]> = vector-pointer ir<[[GEP2]]>
; IF-EVL-NEXT:     WIDEN ir<[[LD2:%.+]]> = vp.load vp<[[PTR2]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     WIDEN ir<[[CMP:%.+]]> = icmp sgt ir<[[LD1]]>, ir<[[LD2]]>
; IF-EVL-NEXT:     WIDEN ir<[[SUB:%.+]]> = vp.sub ir<0>, ir<[[LD2]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     WIDEN-INTRINSIC vp<[[SELECT:%.+]]> = call llvm.vp.select(ir<[[CMP]]>, ir<[[LD2]]>, ir<[[SUB]]>, vp<[[EVL]]>)
; IF-EVL-NEXT:     WIDEN ir<[[ADD:%.+]]> = vp.add vp<[[SELECT]]>, ir<[[LD1]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     CLONE ir<[[GEP3:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR3:%.+]]> = vector-pointer ir<[[GEP3]]>
; IF-EVL-NEXT:     WIDEN vp.store vp<[[PTR3]]>, ir<[[ADD]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     SCALAR-CAST vp<[[CAST:%[0-9]+]]> = zext vp<[[EVL]]> to i64
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT]]> = add vp<[[CAST]]>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT_EXIT]]> = add vp<[[IV]]>, ir<[[VFUF]]>
; IF-EVL-NEXT:     EMIT branch-on-count vp<[[IV_NEXT_EXIT]]>,  ir<[[VTC]]>
; IF-EVL-NEXT:   No successors
; IF-EVL-NEXT: }
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %gep = getelementptr inbounds i32, ptr %b, i64 %iv
  %0 = load i32, ptr %gep, align 4
  %gep3 = getelementptr inbounds i32, ptr %c, i64 %iv
  %1 = load i32, ptr %gep3, align 4
  %cmp4 = icmp sgt i32 %0, %1
  %2 = sub i32 0, %1
  %cond.p = select i1 %cmp4, i32 %1, i32 %2
  %cond = add i32 %cond.p, %0
  %gep15 = getelementptr inbounds i32, ptr %a, i64 %iv
  store i32 %cond, ptr %gep15, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret void
}

define void @vp_select_with_fastflags(ptr %a, ptr %b, ptr %c, i64 %N) {
; IF-EVL: VPlan 'Final VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF={1}' {
; IF-EVL-NEXT: Live-in ir<[[VFUF:%.+]]> = VF * UF
; IF-EVL-NEXT: Live-in ir<[[VTC:%.+]]> = vector-trip-count
; IF-EVL-NEXT: Live-in ir<%N> = original trip-count
; IF-EVL: vector.ph:
; IF-EVL-NEXT: Successor(s): vector loop
; IF-EVL: <x1> vector loop: {
; IF-EVL-NEXT:   vector.body:
; IF-EVL-NEXT:     SCALAR-PHI vp<[[IV:%[0-9]+]]> = phi ir<0>, vp<[[IV_NEXT_EXIT:%.+]]>
; IF-EVL-NEXT:     SCALAR-PHI vp<[[EVL_PHI:%[0-9]+]]>  = phi ir<0>, vp<[[IV_NEXT:%.+]]>
; IF-EVL-NEXT:     EMIT vp<[[AVL:%.+]]> = sub ir<%N>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[EVL:%.+]]> = EXPLICIT-VECTOR-LENGTH vp<[[AVL]]>
; IF-EVL-NEXT:     vp<[[ST:%[0-9]+]]> = SCALAR-STEPS vp<[[EVL_PHI]]>, ir<1>
; IF-EVL-NEXT:     CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds ir<%b>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR1:%[0-9]+]]> = vector-pointer ir<[[GEP1]]>
; IF-EVL-NEXT:     WIDEN ir<[[LD1:%.+]]> = vp.load vp<[[PTR1]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%c>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR2:%[0-9]+]]> = vector-pointer ir<[[GEP2]]>
; IF-EVL-NEXT:     WIDEN ir<[[LD2:%.+]]> = vp.load vp<[[PTR2]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     WIDEN ir<[[FCMP:%.+]]> = fcmp ogt ir<[[LD1]]>, ir<[[LD2]]>
; IF-EVL-NEXT:     WIDEN ir<[[FADD:%.+]]> = vp.fadd reassoc nnan ninf nsz arcp contract afn ir<[[LD1]]>, ir<1.000000e+01>, vp<[[EVL]]>
; IF-EVL-NEXT:     WIDEN-INTRINSIC vp<[[SELECT:%.+]]> = call llvm.vp.select(ir<[[FCMP]]>, ir<[[FADD]]>, ir<[[LD2]]>, vp<[[EVL]]>)
; IF-EVL-NEXT:     CLONE ir<[[GEP3:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR3:%.+]]> = vector-pointer ir<[[GEP3]]>
; IF-EVL-NEXT:     WIDEN vp.store vp<[[PTR3]]>, vp<[[SELECT]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     SCALAR-CAST vp<[[CAST:%[0-9]+]]> = zext vp<[[EVL]]> to i64
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT]]> = add vp<[[CAST]]>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT_EXIT]]> = add vp<[[IV]]>, ir<[[VFUF]]>
; IF-EVL-NEXT:     EMIT branch-on-count vp<[[IV_NEXT_EXIT]]>,  ir<[[VTC]]>
; IF-EVL-NEXT:   No successors
; IF-EVL-NEXT: }
;
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
