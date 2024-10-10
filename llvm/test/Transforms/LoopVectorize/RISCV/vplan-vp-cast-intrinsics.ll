; REQUIRES: asserts

; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN: -force-tail-folding-style=data-with-evl \
; RUN: -prefer-predicate-over-epilogue=predicate-dont-vectorize \
; RUN: -mtriple=riscv64 -mattr=+v -riscv-v-vector-bits-max=128 -disable-output < %s 2>&1 | FileCheck --check-prefix=IF-EVL %s

define void @vp_sext(ptr noalias %a, ptr noalias %b, i64 %N) {
; IF-EVL: VPlan 'Final VPlan for VF={vscale x 1,vscale x 2},UF={1}' {
; IF-EVL-NEXT: Live-in vp<[[VFUF:%[0-9]+]]> = VF * UF
; IF-EVL-NEXT: Live-in vp<[[VTC:%[0-9]+]]> = vector-trip-count
; IF-EVL-NEXT: Live-in ir<%N> = original trip-count

; IF-EVL: vector.ph:
; IF-EVL-NEXT: Successor(s): vector loop

; IF-EVL: <x1> vector loop: {
; IF-EVL-NEXT:  vector.body:
; IF-EVL-NEXT:    EMIT vp<[[IV:%[0-9]+]]> = CANONICAL-INDUCTION
; IF-EVL-NEXT:    EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<[[EVL_PHI:%[0-9]+]]>  = phi ir<0>, vp<[[IV_NEX:%[0-9]+]]>
; IF-EVL-NEXT:    EMIT vp<[[AVL:%.+]]> = sub ir<%N>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:    EMIT vp<[[EVL:%.+]]> = EXPLICIT-VECTOR-LENGTH vp<[[AVL]]>
; IF-EVL-NEXT:    vp<[[ST:%[0-9]+]]> = SCALAR-STEPS vp<[[EVL_PHI]]>, ir<1>
; IF-EVL-NEXT:    CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds ir<%b>, vp<[[ST]]>
; IF-EVL-NEXT:    vp<[[PTR1:%[0-9]+]]> = vector-pointer ir<[[GEP1]]>
; IF-EVL-NEXT:    WIDEN ir<[[LD1:%.+]]> = vp.load vp<[[PTR1]]>, vp<[[EVL]]>
; IF-EVL-NEXT:    WIDEN-CAST ir<[[SEXT:%.+]]> = vp.sext ir<[[LD1]]>, vp<[[EVL]]>
; IF-EVL-NEXT:    CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; IF-EVL-NEXT:    vp<[[PTR2:%[0-9]+]]> = vector-pointer ir<[[GEP2]]>
; IF-EVL-NEXT:    WIDEN vp.store vp<[[PTR2]]>, ir<[[SEXT]]>, vp<[[EVL]]>
; IF-EVL-NEXT:    SCALAR-CAST vp<[[CAST:%[0-9]+]]> = zext vp<[[EVL]]> to i64
; IF-EVL-NEXT:    EMIT vp<[[IV_NEX]]> = add vp<[[CAST]]>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:    EMIT vp<[[IV_NEXT_EXIT:%[0-9]+]]> = add vp<[[IV]]>, vp<[[VFUF]]>
; IF-EVL-NEXT:    EMIT branch-on-count vp<[[IV_NEXT_EXIT]]>,  vp<[[VTC]]>
; IF-EVL-NEXT:  No successors
; IF-EVL-NEXT: }


entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %conv2 = sext i32 %0 to i64
  %arrayidx4 = getelementptr inbounds i64, ptr %a, i64 %iv
  store i64 %conv2, ptr %arrayidx4, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret void
}

define void @vp_zext(ptr noalias %a, ptr noalias %b, i64 %N) {
; IF-EVL: VPlan 'Final VPlan for VF={vscale x 1,vscale x 2},UF={1}' {
; IF-EVL-NEXT: Live-in vp<[[VFUF:%[0-9]+]]> = VF * UF
; IF-EVL-NEXT: Live-in vp<[[VTC:%[0-9]+]]> = vector-trip-count
; IF-EVL-NEXT: Live-in ir<%N> = original trip-count

; IF-EVL: vector.ph:
; IF-EVL-NEXT: Successor(s): vector loop

; IF-EVL: <x1> vector loop: {
; IF-EVL-NEXT:   vector.body:
; IF-EVL-NEXT:     EMIT vp<[[IV:%[0-9]+]]> = CANONICAL-INDUCTION ir<0>, vp<[[IV_NEXT:%[0-9]+]]>
; IF-EVL-NEXT:     EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<[[EVL_PHI:%[0-9]+]]>  = phi ir<0>, vp<[[IV_NEX:%[0-9]+]]>
; IF-EVL-NEXT:     EMIT vp<[[AVL:%.+]]> = sub ir<%N>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[EVL:%.+]]> = EXPLICIT-VECTOR-LENGTH vp<[[AVL]]>
; IF-EVL-NEXT:     vp<[[ST:%[0-9]+]]> = SCALAR-STEPS vp<[[EVL_PHI]]>, ir<1>
; IF-EVL-NEXT:     CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds ir<%b>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR1:%[0-9]+]]> = vector-pointer ir<[[GEP1]]>
; IF-EVL-NEXT:     WIDEN ir<[[LD1:%.+]]> = vp.load vp<[[PTR1]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     WIDEN-CAST ir<[[ZEXT:%.+]]> = vp.zext ir<[[LD1]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR2:%[0-9]+]]> = vector-pointer ir<[[GEP2]]>
; IF-EVL-NEXT:     WIDEN vp.store vp<[[PTR2]]>, ir<[[ZEXT]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     SCALAR-CAST vp<[[CAST:%[0-9]+]]> = zext vp<[[EVL]]> to i64
; IF-EVL-NEXT:     EMIT vp<[[IV_NEX]]> = add vp<[[CAST]]>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT_EXIT:%[0-9]+]]> = add vp<[[IV]]>, vp<[[VFUF]]>
; IF-EVL-NEXT:     EMIT branch-on-count vp<[[IV_NEXT_EXIT]]>,  vp<[[VTC]]>
; IF-EVL-NEXT:   No successors
; IF-EVL-NEXT: }

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %conv2 = zext i32 %0 to i64
  %arrayidx4 = getelementptr inbounds i64, ptr %a, i64 %iv
  store i64 %conv2, ptr %arrayidx4, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret void
}

define void @vp_truncate(ptr noalias %a, ptr noalias %b, i64 %N) {
; IF-EVL: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF={1}' {
; IF-EVL-NEXT: Live-in vp<[[VFUF:%[0-9]+]]> = VF * UF
; IF-EVL-NEXT: Live-in vp<[[VTC:%[0-9]+]]> = vector-trip-count
; IF-EVL-NEXT: Live-in ir<%N> = original trip-count

; IF-EVL: vector.ph:
; IF-EVL-NEXT: Successor(s): vector loop

; IF-EVL: <x1> vector loop: {
; IF-EVL-NEXT:   vector.body:
; IF-EVL-NEXT:     EMIT vp<[[IV:%[0-9]+]]> = CANONICAL-INDUCTION
; IF-EVL-NEXT:     EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<[[EVL_PHI:%[0-9]+]]>  = phi ir<0>, vp<[[IV_NEXT:%[0-9]+]]>
; IF-EVL-NEXT:     EMIT vp<[[AVL:%.+]]> = sub ir<%N>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[EVL:%.+]]> = EXPLICIT-VECTOR-LENGTH vp<[[AVL]]>
; IF-EVL-NEXT:     vp<[[ST:%[0-9]+]]> = SCALAR-STEPS vp<[[EVL_PHI]]>, ir<1>
; IF-EVL-NEXT:     CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds ir<%b>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR1:%[0-9]+]]> = vector-pointer ir<[[GEP1]]>
; IF-EVL-NEXT:     WIDEN ir<[[LD1:%.+]]> = vp.load vp<[[PTR1]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     WIDEN-CAST ir<[[TRUNC:%.+]]> = vp.trunc ir<[[LD1]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR2:%[0-9]+]]> = vector-pointer ir<[[GEP2]]>
; IF-EVL-NEXT:     WIDEN vp.store vp<[[PTR2]]>, ir<[[TRUNC]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     SCALAR-CAST vp<[[CAST:%[0-9]+]]> = zext vp<[[EVL]]> to i64
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT]]> = add vp<[[CAST]]>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT_EXIT:%[0-9]+]]> = add vp<[[IV]]>, vp<[[VFUF]]>
; IF-EVL-NEXT:     EMIT branch-on-count vp<[[IV_NEXT_EXIT]]>, vp<[[VTC]]>
; IF-EVL-NEXT:   No successors
; IF-EVL-NEXT: }

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %conv2 = trunc i32 %0 to i16
  %arrayidx4 = getelementptr inbounds i16, ptr %a, i64 %iv
  store i16 %conv2, ptr %arrayidx4, align 2
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret void
}

define void @vp_fpext(ptr noalias %a, ptr noalias %b, i64 %N) {
; IF-EVL: VPlan 'Final VPlan for VF={vscale x 1,vscale x 2},UF={1}' {
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
; IF-EVL-NEXT:     WIDEN-CAST ir<[[FPEXT:%.+]]> = vp.fpext ir<[[LD1]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR2:%[0-9]+]]> = vector-pointer ir<[[GEP2]]>
; IF-EVL-NEXT:     WIDEN vp.store vp<[[PTR2]]>, ir<[[FPEXT]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     SCALAR-CAST vp<[[CAST:%[0-9]+]]> = zext vp<[[EVL]]> to i64
; IF-EVL-NEXT:     EMIT vp<[[IV_NEX]]> = add vp<[[CAST]]>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT_EXIT:%[0-9]+]]> = add vp<[[IV]]>, vp<[[VFUF]]>
; IF-EVL-NEXT:     EMIT branch-on-count vp<[[IV_NEXT_EXIT]]>,  vp<[[VTC]]>
; IF-EVL-NEXT:   No successors
; IF-EVL-NEXT: }

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, ptr %b, i64 %iv
  %0 = load float, ptr %arrayidx, align 4
  %conv2 = fpext float %0 to double
  %arrayidx4 = getelementptr inbounds double, ptr %a, i64 %iv
  store double %conv2, ptr %arrayidx4, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret void
}

define void @vp_fptrunct(ptr noalias %a, ptr noalias %b, i64 %N) {
; IF-EVL: VPlan 'Final VPlan for VF={vscale x 1,vscale x 2},UF={1}' {
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
; IF-EVL-NEXT:     WIDEN-CAST ir<[[FPTRUNC:%.+]]> = vp.fptrunc ir<[[LD1]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR2:%[0-9]+]]> = vector-pointer ir<[[GEP2]]>
; IF-EVL-NEXT:     WIDEN vp.store vp<[[PTR2]]>, ir<[[FPTRUNC]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     SCALAR-CAST vp<[[CAST:%[0-9]+]]> = zext vp<[[EVL]]> to i64
; IF-EVL-NEXT:     EMIT vp<[[IV_NEX:%[0-9]+]]> = add vp<[[CAST]]>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT_EXIT:%[0-9]+]]> = add vp<[[IV]]>, vp<[[VFUF]]>
; IF-EVL-NEXT:     EMIT branch-on-count vp<[[IV_NEXT_EXIT]]>,  vp<[[VTC]]>
; IF-EVL-NEXT:   No successors
; IF-EVL-NEXT: }

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double, ptr %b, i64 %iv
  %0 = load double, ptr %arrayidx, align 8
  %conv2 = fptrunc double %0 to float
  %arrayidx4 = getelementptr inbounds float, ptr %a, i64 %iv
  store float %conv2, ptr %arrayidx4, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret void
}

define void @vp_fptosi(ptr noalias %a, ptr noalias %b, i64 %N) {
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
; IF-EVL-NEXT:     WIDEN-CAST ir<[[FPTOSI:%.+]]> = vp.fptoui ir<[[LD1]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR2:%[0-9]+]]> = vector-pointer ir<[[GEP2]]>
; IF-EVL-NEXT:     WIDEN vp.store vp<[[PTR2]]>, ir<[[FPTOSI]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     SCALAR-CAST vp<[[CAST:%[0-9]+]]> = zext vp<[[EVL]]> to i64
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT]]> = add vp<[[CAST]]>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT_EXIT:%[0-9]+]]> = add vp<[[IV]]>, vp<[[VFUF]]>
; IF-EVL-NEXT:     EMIT branch-on-count vp<[[IV_NEXT_EXIT]]>,  vp<[[VTC]]>
; IF-EVL-NEXT:   No successors
; IF-EVL-NEXT: }

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, ptr %b, i64 %iv
  %0 = load float, ptr %arrayidx, align 4
  %conv2 = fptoui float %0 to i32
  %arrayidx4 = getelementptr inbounds i32, ptr %a, i64 %iv
  store i32 %conv2, ptr %arrayidx4, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret void
}

define void @vp_inttofp(ptr noalias %a, ptr noalias %b, i64 %N) {
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
; IF-EVL-NEXT:     WIDEN-CAST ir<[[SITOFP:%.+]]> = vp.sitofp ir<[[LD1]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR2:%[0-9]+]]> = vector-pointer ir<[[GEP2]]>
; IF-EVL-NEXT:     WIDEN vp.store vp<[[PTR2]]>, ir<[[SITOFP]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     SCALAR-CAST vp<[[CAST:%[0-9]+]]> = zext vp<[[EVL]]> to i64
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT]]> = add vp<[[CAST]]>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT_EXIT:%[0-9]+]]> = add vp<[[IV]]>, vp<[[VFUF]]>
; IF-EVL-NEXT:     EMIT branch-on-count vp<[[IV_NEXT_EXIT]]>,  vp<[[VTC]]>
; IF-EVL-NEXT:   No successors
; IF-EVL-NEXT: }

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %conv2 = sitofp i32 %0 to float
  %arrayidx4 = getelementptr inbounds float, ptr %a, i64 %iv
  store float %conv2, ptr %arrayidx4, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret void
}

define void @vp_uinttofp(ptr noalias %a, ptr noalias %b, i64 %N) {
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
; IF-EVL-NEXT:     WIDEN-CAST ir<[[UITOFP:%.+]]> = vp.uitofp ir<[[LD1]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR2:%[0-9]+]]> = vector-pointer ir<[[GEP2]]>
; IF-EVL-NEXT:     WIDEN vp.store vp<[[PTR2]]>, ir<[[UITOFP]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     SCALAR-CAST vp<[[CAST:%[0-9]+]]> = zext vp<[[EVL]]> to i64
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT]]> = add vp<[[CAST]]>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT_EXIT:%[0-9]+]]> = add vp<[[IV]]>, vp<[[VFUF]]>
; IF-EVL-NEXT:     EMIT branch-on-count vp<[[IV_NEXT_EXIT]]>,  vp<[[VTC]]>
; IF-EVL-NEXT:   No successors
; IF-EVL-NEXT: }

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %conv2 = uitofp i32 %0 to float
  %arrayidx4 = getelementptr inbounds float, ptr %a, i64 %iv
  store float %conv2, ptr %arrayidx4, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret void
}
