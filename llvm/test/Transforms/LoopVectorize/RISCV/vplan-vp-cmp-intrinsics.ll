; REQUIRES: asserts

; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN: -force-tail-folding-style=data-with-evl \
; RUN: -prefer-predicate-over-epilogue=predicate-dont-vectorize \
; RUN: -mtriple=riscv64 -mattr=+v -riscv-v-vector-bits-max=128 -disable-output < %s 2>&1 | FileCheck --check-prefix=IF-EVL %s

define void @vp_icmp(ptr noalias %a, ptr noalias %b, ptr noalias %c, i64 %N) {
; IF-EVL: VPlan 'Final VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF={1}' {
; IF-EVL-NEXT: Live-in vp<[[VFUF:%[0-9]+]]> = VF * UF
; IF-EVL-NEXT: Live-in vp<[[VTC:%[0-9]+]]> = vector-trip-count
; IF-EVL-NEXT: Live-in ir<%N> = original trip-count

; IF-EVL: vector.ph:
; IF-EVL-NEXT: Successor(s): vector loop

; IF-EVL: <x1> vector loop: {
; IF-EVL-NEXT:   vector.body:
; IF-EVL-NEXT:     EMIT vp<[[IV:%[0-9]+]]> = CANONICAL-INDUCTION
; IF-EVL-NEXT:     EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<[[EVL_PHI:%[0-9]+]]>  = phi ir<0>, vp<[[IV_NEX:%[0-9]+]]>
; IF-EVL-NEXT:     EMIT vp<[[AVL:%.+]]> = sub ir<%N>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[EVL:%.+]]> = EXPLICIT-VECTOR-LENGTH vp<[[AVL]]>
; IF-EVL-NEXT:     vp<[[ST:%[0-9]+]]> = SCALAR-STEPS vp<[[EVL_PHI]]>, ir<1>
; IF-EVL-NEXT:     CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds ir<%b>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR1:%[0-9]+]]> = vector-pointer ir<[[GEP1]]>
; IF-EVL-NEXT:     WIDEN ir<[[LD1:%.+]]> = vp.load vp<[[PTR1]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%c>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR2:%[0-9]+]]> = vector-pointer ir<[[GEP2]]>
; IF-EVL-NEXT:     WIDEN ir<[[LD2:%.+]]> = vp.load vp<[[PTR2]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     WIDEN ir<[[ICMP:%.+]]> = vp.icmp sgt ir<[[LD1]]>, ir<[[LD2]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     WIDEN-CAST ir<[[ZEXT:%.+]]> = zext ir<[[ICMP]]> to i32
; IF-EVL-NEXT:     CLONE ir<[[GEP3:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR3:%[0-9]+]]> = vector-pointer ir<[[GEP3]]>
; IF-EVL-NEXT:     WIDEN vp.store vp<[[PTR3]]>, ir<[[ZEXT]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     SCALAR-CAST vp<[[CAST:%[0-9]+]]> = zext vp<[[EVL]]> to i64
; IF-EVL-NEXT:     EMIT vp<[[IV_NEX]]> = add vp<[[CAST]]>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT_EXIT:%[0-9]+]]> = add vp<[[IV]]>, vp<[[VFUF]]>
; IF-EVL-NEXT:     EMIT branch-on-count vp<[[IV_NEXT_EXIT]]>,  vp<[[VTC]]>
; IF-EVL-NEXT:   No successors
; IF-EVL-NEXT: }

entry:
  %cmp12 = icmp sgt i64 %N, 0
  br i1 %cmp12, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx3 = getelementptr inbounds i32, ptr %c, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx3, align 4
  %cmp4 = icmp sgt i32 %0, %1
  %conv5 = zext i1 %cmp4 to i32
  %arrayidx7 = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  store i32 %conv5, ptr %arrayidx7, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @vp_fcmp(ptr noalias %a, ptr noalias %b, ptr noalias %c, i64 %N) {
; IF-EVL: VPlan 'Final VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF={1}' {
; IF-EVL-NEXT: Live-in vp<[[VFUF:%[0-9]+]]> = VF * UF
; IF-EVL-NEXT: Live-in vp<[[VTC:%[0-9]+]]> = vector-trip-count
; IF-EVL-NEXT: Live-in ir<%N> = original trip-count

; IF-EVL: vector.ph:
; IF-EVL-NEXT: Successor(s): vector loop

; IF-EVL: <x1> vector loop: {
; IF-EVL-NEXT:   vector.body:
; IF-EVL-NEXT:     EMIT vp<[[IV:%[0-9]+]]> = CANONICAL-INDUCTION
; IF-EVL-NEXT:     EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<[[EVL_PHI:%[0-9]+]]>  = phi ir<0>, vp<[[IV_NEX:%[0-9]+]]>
; IF-EVL-NEXT:     EMIT vp<[[AVL:%.+]]> = sub ir<%N>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[EVL:%.+]]> = EXPLICIT-VECTOR-LENGTH vp<[[AVL]]>
; IF-EVL-NEXT:     vp<[[ST:%[0-9]+]]> = SCALAR-STEPS vp<[[EVL_PHI]]>, ir<1>
; IF-EVL-NEXT:     CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds ir<%b>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR1:%[0-9]+]]> = vector-pointer ir<[[GEP1]]>
; IF-EVL-NEXT:     WIDEN ir<[[LD1:%.+]]> = vp.load vp<[[PTR1]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%c>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR2:%[0-9]+]]> = vector-pointer ir<[[GEP2]]>
; IF-EVL-NEXT:     WIDEN ir<[[LD2:%.+]]> = vp.load vp<[[PTR2]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     WIDEN ir<[[FCMP:%.+]]> = vp.fcmp ogt ir<[[LD1]]>, ir<[[LD2]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     WIDEN-CAST ir<[[UITOFP:%.+]]> = uitofp  ir<[[FCMP]]> to float
; IF-EVL-NEXT:     CLONE ir<[[GEP3:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR3:%[0-9]+]]> = vector-pointer ir<[[GEP3]]>
; IF-EVL-NEXT:     WIDEN vp.store vp<[[PTR3]]>, ir<[[UITOFP]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     SCALAR-CAST vp<[[CAST:%[0-9]+]]> = zext vp<[[EVL]]> to i64
; IF-EVL-NEXT:     EMIT vp<[[IV_NEX]]> = add vp<[[CAST]]>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT_EXIT:%[0-9]+]]> = add vp<[[IV]]>, vp<[[VFUF]]>
; IF-EVL-NEXT:     EMIT branch-on-count vp<[[IV_NEXT_EXIT]]>,  vp<[[VTC]]>
; IF-EVL-NEXT:   No successors
; IF-EVL-NEXT: }

entry:
  %cmp13 = icmp sgt i64 %N, 0
  br i1 %cmp13, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %arrayidx3 = getelementptr inbounds float, ptr %c, i64 %indvars.iv
  %1 = load float, ptr %arrayidx3, align 4
  %cmp4 = fcmp ogt float %0, %1
  %conv6 = uitofp i1 %cmp4 to float
  %arrayidx8 = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  store float %conv6, ptr %arrayidx8, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}