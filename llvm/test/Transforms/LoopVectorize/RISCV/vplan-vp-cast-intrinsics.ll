; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN: -force-tail-folding-style=data-with-evl \
; RUN: -prefer-predicate-over-epilogue=predicate-dont-vectorize \
; RUN: -mtriple=riscv64 -mattr=+v -riscv-v-vector-bits-max=128 -disable-output < %s 2>&1 | FileCheck --check-prefix=IF-EVL %s

define void @vp_sext(ptr %a, ptr %b, i64 %N) {
; IF-EVL: VPlan 'Initial VPlan for VF={1},UF>=1'
; IF-EVL-NOT: EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI
;
; IF-EVL: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2},UF={1}' {
; IF-EVL-NEXT: Live-in vp<[[VFUF:%[0-9]+]]> = VF * UF
; IF-EVL-NEXT: Live-in vp<[[VTC:%[0-9]+]]> = vector-trip-count
; IF-EVL-NEXT: Live-in ir<%N> = original trip-count

; IF-EVL: vector.ph:
; IF-EVL-NEXT: Successor(s): vector loop

; IF-EVL:   <x1> vector loop: {
; IF-EVL-NEXT:   vector.body:
; IF-EVL-NEXT:     EMIT vp<[[IV:%[0-9]+]]> = CANONICAL-INDUCTION
; IF-EVL-NEXT:     EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<[[EVL_PHI:%[0-9]+]]>  = phi ir<0>, vp<[[IV_NEXT:%.+]]>
; IF-EVL-NEXT:     EMIT vp<[[AVL:%.+]]> = sub ir<%N>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[EVL:%.+]]> = EXPLICIT-VECTOR-LENGTH vp<[[AVL]]>
; IF-EVL-NEXT:     vp<[[ST:%[0-9]+]]> = SCALAR-STEPS vp<[[EVL_PHI]]>, ir<1>, vp<[[EVL]]>
; IF-EVL-NEXT:     CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds ir<%b>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR1:%[0-9]+]]> = vector-pointer ir<[[GEP1]]>
; IF-EVL-NEXT:     WIDEN ir<[[LD1:%.+]]> = vp.load vp<[[PTR1]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     WIDEN-CAST ir<[[SEXT:%.+]]> = sext ir<[[LD1]]> to i64
; IF-EVL-NEXT:     CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR2:%[0-9]+]]> = vector-pointer ir<[[GEP2]]>
; IF-EVL-NEXT:     WIDEN vp.store vp<[[PTR2]]>, ir<[[SEXT]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     EMIT-SCALAR vp<[[CAST:%[0-9]+]]> = zext vp<[[EVL]]> to i64
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT]]> = add vp<[[CAST]]>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT_EXIT:%.+]]> = add vp<[[IV]]>, vp<[[VFUF]]>
; IF-EVL-NEXT:     EMIT branch-on-count vp<[[IV_NEXT_EXIT]]>,  vp<[[VTC]]>
; IF-EVL-NEXT:   No successors
; IF-EVL-NEXT: }
; IF-EVL-NEXT: Successor(s): middle.block


entry:
  br label %loop

loop:
  %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
  %gep = getelementptr inbounds i32, ptr %b, i64 %iv
  %0 = load i32, ptr %gep, align 4
  %conv2 = sext i32 %0 to i64
  %gep4 = getelementptr inbounds i64, ptr %a, i64 %iv
  store i64 %conv2, ptr %gep4, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}

define void @vp_zext(ptr %a, ptr %b, i64 %N) {
; IF-EVL: VPlan 'Initial VPlan for VF={1},UF>=1'
; IF-EVL-NOT: EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI
;
; IF-EVL: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2},UF={1}' {
; IF-EVL-NEXT: Live-in vp<[[VFUF:%[0-9]+]]> = VF * UF
; IF-EVL-NEXT: Live-in vp<[[VTC:%[0-9]+]]> = vector-trip-count
; IF-EVL-NEXT: Live-in ir<%N> = original trip-count

; IF-EVL: vector.ph:
; IF-EVL-NEXT: Successor(s): vector loop

; IF-EVL: <x1> vector loop: {
; IF-EVL-NEXT:   vector.body:
; IF-EVL-NEXT:     EMIT vp<[[IV:%[0-9]+]]> = CANONICAL-INDUCTION
; IF-EVL-NEXT:     EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<[[EVL_PHI:%[0-9]+]]>  = phi ir<0>, vp<[[IV_NEXT:%.+]]>
; IF-EVL-NEXT:     EMIT vp<[[AVL:%.+]]> = sub ir<%N>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[EVL:%.+]]> = EXPLICIT-VECTOR-LENGTH vp<[[AVL]]>
; IF-EVL-NEXT:     vp<[[ST:%[0-9]+]]> = SCALAR-STEPS vp<[[EVL_PHI]]>, ir<1>, vp<[[EVL]]>
; IF-EVL-NEXT:     CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds ir<%b>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR1:%[0-9]+]]> = vector-pointer ir<[[GEP1]]>
; IF-EVL-NEXT:     WIDEN ir<[[LD1:%.+]]> = vp.load vp<[[PTR1]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     WIDEN-CAST ir<[[ZEXT:%.+]]> = zext ir<[[LD1]]> to i64
; IF-EVL-NEXT:     CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR2:%[0-9]+]]> = vector-pointer ir<[[GEP2]]>
; IF-EVL-NEXT:     WIDEN vp.store vp<[[PTR2]]>, ir<[[ZEXT]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     EMIT-SCALAR vp<[[CAST:%[0-9]+]]> = zext vp<[[EVL]]> to i64
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT]]> = add vp<[[CAST]]>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT_EXIT:%.+]]> = add vp<[[IV]]>, vp<[[VFUF]]>
; IF-EVL-NEXT:     EMIT branch-on-count vp<[[IV_NEXT_EXIT]]>,  vp<[[VTC]]>
; IF-EVL-NEXT:   No successors
; IF-EVL-NEXT: }

entry:
  br label %loop

loop:
  %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
  %gep = getelementptr inbounds i32, ptr %b, i64 %iv
  %0 = load i32, ptr %gep, align 4
  %conv2 = zext i32 %0 to i64
  %gep4 = getelementptr inbounds i64, ptr %a, i64 %iv
  store i64 %conv2, ptr %gep4, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}

define void @vp_trunc(ptr %a, ptr %b, i64 %N) {
; IF-EVL: VPlan 'Initial VPlan for VF={1},UF>=1'
; IF-EVL-NOT: EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI
;
; IF-EVL: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF={1}' {
; IF-EVL-NEXT: Live-in vp<[[VFUF:%[0-9]+]]> = VF * UF
; IF-EVL-NEXT: Live-in vp<[[VTC:%[0-9]+]]> = vector-trip-count
; IF-EVL-NEXT: Live-in ir<%N> = original trip-count

; IF-EVL: vector.ph:
; IF-EVL-NEXT: Successor(s): vector loop

; IF-EVL: <x1> vector loop: {
; IF-EVL-NEXT:   vector.body:
; IF-EVL-NEXT:     EMIT vp<[[IV:%[0-9]+]]> = CANONICAL-INDUCTION
; IF-EVL-NEXT:     EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<[[EVL_PHI:%[0-9]+]]>  = phi ir<0>, vp<[[IV_NEXT:%.+]]>
; IF-EVL-NEXT:     EMIT vp<[[AVL:%.+]]> = sub ir<%N>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[EVL:%.+]]> = EXPLICIT-VECTOR-LENGTH vp<[[AVL]]>
; IF-EVL-NEXT:     vp<[[ST:%[0-9]+]]> = SCALAR-STEPS vp<[[EVL_PHI]]>, ir<1>, vp<[[EVL]]>
; IF-EVL-NEXT:     CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds ir<%b>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR1:%[0-9]+]]> = vector-pointer ir<[[GEP1]]>
; IF-EVL-NEXT:     WIDEN ir<[[LD1:%.+]]> = vp.load vp<[[PTR1]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     WIDEN-CAST ir<[[TRUNC:%.+]]> = trunc ir<[[LD1]]> to i16
; IF-EVL-NEXT:     CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR2:%[0-9]+]]> = vector-pointer ir<[[GEP2]]>
; IF-EVL-NEXT:     WIDEN vp.store vp<[[PTR2]]>, ir<[[TRUNC]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     EMIT-SCALAR vp<[[CAST:%[0-9]+]]> = zext vp<[[EVL]]> to i64
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT]]> = add vp<[[CAST]]>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT_EXIT:%.+]]> = add vp<[[IV]]>, vp<[[VFUF]]>
; IF-EVL-NEXT:     EMIT branch-on-count vp<[[IV_NEXT_EXIT]]>, vp<[[VTC]]>
; IF-EVL-NEXT:   No successors
; IF-EVL-NEXT: }

entry:
  br label %loop

loop:
  %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
  %gep = getelementptr inbounds i32, ptr %b, i64 %iv
  %0 = load i32, ptr %gep, align 4
  %conv2 = trunc i32 %0 to i16
  %gep4 = getelementptr inbounds i16, ptr %a, i64 %iv
  store i16 %conv2, ptr %gep4, align 2
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}

define void @vp_fpext(ptr %a, ptr %b, i64 %N) {
; IF-EVL: VPlan 'Initial VPlan for VF={1},UF>=1'
; IF-EVL-NOT: EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI
;
; IF-EVL: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2},UF={1}' {
; IF-EVL-NEXT: Live-in vp<[[VFUF:%[0-9]+]]> = VF * UF
; IF-EVL-NEXT: Live-in vp<[[VTC:%[0-9]+]]> = vector-trip-count
; IF-EVL-NEXT: Live-in ir<%N> = original trip-count

; IF-EVL: vector.ph:
; IF-EVL-NEXT: Successor(s): vector loop

; IF-EVL: <x1> vector loop: {
; IF-EVL-NEXT:   vector.body:
; IF-EVL-NEXT:     EMIT vp<[[IV:%[0-9]+]]> = CANONICAL-INDUCTION
; IF-EVL-NEXT:     EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<[[EVL_PHI:%[0-9]+]]>  = phi ir<0>, vp<[[IV_NEXT:%.+]]>
; IF-EVL-NEXT:     EMIT vp<[[AVL:%.+]]> = sub ir<%N>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[EVL:%.+]]> = EXPLICIT-VECTOR-LENGTH vp<[[AVL]]>
; IF-EVL-NEXT:     vp<[[ST:%[0-9]+]]> = SCALAR-STEPS vp<[[EVL_PHI]]>, ir<1>, vp<[[EVL]]>
; IF-EVL-NEXT:     CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds ir<%b>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR1:%[0-9]+]]> = vector-pointer ir<[[GEP1]]>
; IF-EVL-NEXT:     WIDEN ir<[[LD1:%.+]]> = vp.load vp<[[PTR1]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     WIDEN-CAST ir<[[FPEXT:%.+]]> = fpext ir<[[LD1]]> to double
; IF-EVL-NEXT:     CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR2:%[0-9]+]]> = vector-pointer ir<[[GEP2]]>
; IF-EVL-NEXT:     WIDEN vp.store vp<[[PTR2]]>, ir<[[FPEXT]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     EMIT-SCALAR vp<[[CAST:%[0-9]+]]> = zext vp<[[EVL]]> to i64
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT]]> = add vp<[[CAST]]>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT_EXIT:%.+]]> = add vp<[[IV]]>, vp<[[VFUF]]>
; IF-EVL-NEXT:     EMIT branch-on-count vp<[[IV_NEXT_EXIT]]>,  vp<[[VTC]]>
; IF-EVL-NEXT:   No successors
; IF-EVL-NEXT: }

entry:
  br label %loop

loop:
  %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
  %gep = getelementptr inbounds float, ptr %b, i64 %iv
  %0 = load float, ptr %gep, align 4
  %conv2 = fpext float %0 to double
  %gep4 = getelementptr inbounds double, ptr %a, i64 %iv
  store double %conv2, ptr %gep4, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}

define void @vp_fptrunc(ptr %a, ptr %b, i64 %N) {
; IF-EVL: VPlan 'Initial VPlan for VF={1},UF>=1'
; IF-EVL-NOT: EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI
;
; IF-EVL: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2},UF={1}' {
; IF-EVL-NEXT: Live-in vp<[[VFUF:%[0-9]+]]> = VF * UF
; IF-EVL-NEXT: Live-in vp<[[VTC:%[0-9]+]]> = vector-trip-count
; IF-EVL-NEXT: Live-in ir<%N> = original trip-count

; IF-EVL: vector.ph:
; IF-EVL-NEXT: Successor(s): vector loop

; IF-EVL: <x1> vector loop: {
; IF-EVL-NEXT:   vector.body:
; IF-EVL-NEXT:     EMIT vp<[[IV:%[0-9]+]]> = CANONICAL-INDUCTION
; IF-EVL-NEXT:     EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<[[EVL_PHI:%[0-9]+]]>  = phi ir<0>, vp<[[IV_NEXT:%.+]]>
; IF-EVL-NEXT:     EMIT vp<[[AVL:%.+]]> = sub ir<%N>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[EVL:%.+]]> = EXPLICIT-VECTOR-LENGTH vp<[[AVL]]>
; IF-EVL-NEXT:     vp<[[ST:%[0-9]+]]> = SCALAR-STEPS vp<[[EVL_PHI]]>, ir<1>, vp<[[EVL]]>
; IF-EVL-NEXT:     CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds ir<%b>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR1:%[0-9]+]]> = vector-pointer ir<[[GEP1]]>
; IF-EVL-NEXT:     WIDEN ir<[[LD1:%.+]]> = vp.load vp<[[PTR1]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     WIDEN-CAST ir<[[FPTRUNC:%.+]]> = fptrunc ir<[[LD1]]> to float
; IF-EVL-NEXT:     CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR2:%[0-9]+]]> = vector-pointer ir<[[GEP2]]>
; IF-EVL-NEXT:     WIDEN vp.store vp<[[PTR2]]>, ir<[[FPTRUNC]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     EMIT-SCALAR vp<[[CAST:%[0-9]+]]> = zext vp<[[EVL]]> to i64
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT]]> = add vp<[[CAST]]>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT_EXIT:%.+]]> = add vp<[[IV]]>, vp<[[VFUF]]>
; IF-EVL-NEXT:     EMIT branch-on-count vp<[[IV_NEXT_EXIT]]>,  vp<[[VTC]]>
; IF-EVL-NEXT:   No successors
; IF-EVL-NEXT: }

entry:
  br label %loop

loop:
  %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
  %gep = getelementptr inbounds double, ptr %b, i64 %iv
  %0 = load double, ptr %gep, align 8
  %conv2 = fptrunc double %0 to float
  %gep4 = getelementptr inbounds float, ptr %a, i64 %iv
  store float %conv2, ptr %gep4, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}

define void @vp_sitofp(ptr %a, ptr %b, i64 %N) {
; IF-EVL: VPlan 'Initial VPlan for VF={1},UF>=1'
; IF-EVL-NOT: EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI
;
; IF-EVL: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF={1}' {
; IF-EVL-NEXT: Live-in vp<[[VFUF:%[0-9]+]]> = VF * UF
; IF-EVL-NEXT: Live-in vp<[[VTC:%[0-9]+]]> = vector-trip-count
; IF-EVL-NEXT: Live-in ir<%N> = original trip-count

; IF-EVL: vector.ph:
; IF-EVL-NEXT: Successor(s): vector loop

; IF-EVL: <x1> vector loop: {
; IF-EVL-NEXT:   vector.body:
; IF-EVL-NEXT:     EMIT vp<[[IV:%[0-9]+]]> = CANONICAL-INDUCTION
; IF-EVL-NEXT:     EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<[[EVL_PHI:%[0-9]+]]>  = phi ir<0>, vp<[[IV_NEXT:%.+]]>
; IF-EVL-NEXT:     EMIT vp<[[AVL:%.+]]> = sub ir<%N>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[EVL:%.+]]> = EXPLICIT-VECTOR-LENGTH vp<[[AVL]]>
; IF-EVL-NEXT:     vp<[[ST:%[0-9]+]]> = SCALAR-STEPS vp<[[EVL_PHI]]>, ir<1>, vp<[[EVL]]>
; IF-EVL-NEXT:     CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds ir<%b>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR1:%[0-9]+]]> = vector-pointer ir<[[GEP1]]>
; IF-EVL-NEXT:     WIDEN ir<[[LD1:%.+]]> = vp.load vp<[[PTR1]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     WIDEN-CAST ir<[[SITOFP:%.+]]> = sitofp ir<[[LD1]]> to float
; IF-EVL-NEXT:     CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR2:%[0-9]+]]> = vector-pointer ir<[[GEP2]]>
; IF-EVL-NEXT:     WIDEN vp.store vp<[[PTR2]]>, ir<[[SITOFP]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     EMIT-SCALAR vp<[[CAST:%[0-9]+]]> = zext vp<[[EVL]]> to i64
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT]]> = add vp<[[CAST]]>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT_EXIT:%.+]]> = add vp<[[IV]]>, vp<[[VFUF]]>
; IF-EVL-NEXT:     EMIT branch-on-count vp<[[IV_NEXT_EXIT]]>,  vp<[[VTC]]>
; IF-EVL-NEXT:   No successors
; IF-EVL-NEXT: }

entry:
  br label %loop

loop:
  %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
  %gep = getelementptr inbounds i32, ptr %b, i64 %iv
  %0 = load i32, ptr %gep, align 4
  %conv2 = sitofp i32 %0 to float
  %gep4 = getelementptr inbounds float, ptr %a, i64 %iv
  store float %conv2, ptr %gep4, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}

define void @vp_uitofp(ptr %a, ptr %b, i64 %N) {
; IF-EVL: VPlan 'Initial VPlan for VF={1},UF>=1'
; IF-EVL-NOT: EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI
;
; IF-EVL: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF={1}' {
; IF-EVL-NEXT: Live-in vp<[[VFUF:%[0-9]+]]> = VF * UF
; IF-EVL-NEXT: Live-in vp<[[VTC:%[0-9]+]]> = vector-trip-count
; IF-EVL-NEXT: Live-in ir<%N> = original trip-count

; IF-EVL: vector.ph:
; IF-EVL-NEXT: Successor(s): vector loop

; IF-EVL: <x1> vector loop: {
; IF-EVL-NEXT:   vector.body:
; IF-EVL-NEXT:     EMIT vp<[[IV:%[0-9]+]]> = CANONICAL-INDUCTION
; IF-EVL-NEXT:     EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<[[EVL_PHI:%[0-9]+]]>  = phi ir<0>, vp<[[IV_NEXT:%.+]]>
; IF-EVL-NEXT:     EMIT vp<[[AVL:%.+]]> = sub ir<%N>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[EVL:%.+]]> = EXPLICIT-VECTOR-LENGTH vp<[[AVL]]>
; IF-EVL-NEXT:     vp<[[ST:%[0-9]+]]> = SCALAR-STEPS vp<[[EVL_PHI]]>, ir<1>, vp<[[EVL]]
; IF-EVL-NEXT:     CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds ir<%b>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR1:%[0-9]+]]> = vector-pointer ir<[[GEP1]]>
; IF-EVL-NEXT:     WIDEN ir<[[LD1:%.+]]> = vp.load vp<[[PTR1]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     WIDEN-CAST ir<[[UITOFP:%.+]]> = uitofp ir<[[LD1]]> to float
; IF-EVL-NEXT:     CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR2:%[0-9]+]]> = vector-pointer ir<[[GEP2]]>
; IF-EVL-NEXT:     WIDEN vp.store vp<[[PTR2]]>, ir<[[UITOFP]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     EMIT-SCALAR vp<[[CAST:%[0-9]+]]> = zext vp<[[EVL]]> to i64
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT]]> = add vp<[[CAST]]>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT_EXIT:%.+]]> = add vp<[[IV]]>, vp<[[VFUF]]>
; IF-EVL-NEXT:     EMIT branch-on-count vp<[[IV_NEXT_EXIT]]>,  vp<[[VTC]]>
; IF-EVL-NEXT:   No successors
; IF-EVL-NEXT: }

entry:
  br label %loop

loop:
  %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
  %gep = getelementptr inbounds i32, ptr %b, i64 %iv
  %0 = load i32, ptr %gep, align 4
  %conv2 = uitofp i32 %0 to float
  %gep4 = getelementptr inbounds float, ptr %a, i64 %iv
  store float %conv2, ptr %gep4, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}

define void @vp_fptosi(ptr %a, ptr %b, i64 %N) {
; IF-EVL: VPlan 'Initial VPlan for VF={1},UF>=1'
; IF-EVL-NOT: EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI
;
; IF-EVL: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF={1}' {
; IF-EVL-NEXT: Live-in vp<[[VFUF:%[0-9]+]]> = VF * UF
; IF-EVL-NEXT: Live-in vp<[[VTC:%[0-9]+]]> = vector-trip-count
; IF-EVL-NEXT: Live-in ir<%N> = original trip-count

; IF-EVL: vector.ph:
; IF-EVL-NEXT: Successor(s): vector loop

; IF-EVL: <x1> vector loop: {
; IF-EVL-NEXT:   vector.body:
; IF-EVL-NEXT:     EMIT vp<[[IV:%[0-9]+]]> = CANONICAL-INDUCTION
; IF-EVL-NEXT:     EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<[[EVL_PHI:%[0-9]+]]>  = phi ir<0>, vp<[[IV_NEXT:%.+]]>
; IF-EVL-NEXT:     EMIT vp<[[AVL:%.+]]> = sub ir<%N>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[EVL:%.+]]> = EXPLICIT-VECTOR-LENGTH vp<[[AVL]]>
; IF-EVL-NEXT:     vp<[[ST:%[0-9]+]]> = SCALAR-STEPS vp<[[EVL_PHI]]>, ir<1>, vp<[[EVL]]>
; IF-EVL-NEXT:     CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds ir<%b>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR1:%[0-9]+]]> = vector-pointer ir<[[GEP1]]>
; IF-EVL-NEXT:     WIDEN ir<[[LD1:%.+]]> = vp.load vp<[[PTR1]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     WIDEN-CAST ir<[[FPTOSI:%.+]]> = fptosi ir<[[LD1]]> to i32
; IF-EVL-NEXT:     CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR2:%[0-9]+]]> = vector-pointer ir<[[GEP2]]>
; IF-EVL-NEXT:     WIDEN vp.store vp<[[PTR2]]>, ir<[[FPTOSI]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     EMIT-SCALAR vp<[[CAST:%[0-9]+]]> = zext vp<[[EVL]]> to i64
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT]]> = add vp<[[CAST]]>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT_EXIT:%.+]]> = add vp<[[IV]]>, vp<[[VFUF]]>
; IF-EVL-NEXT:     EMIT branch-on-count vp<[[IV_NEXT_EXIT]]>,  vp<[[VTC]]>
; IF-EVL-NEXT:   No successors
; IF-EVL-NEXT: }

entry:
  br label %loop

loop:
  %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
  %gep = getelementptr inbounds float, ptr %b, i64 %iv
  %0 = load float, ptr %gep, align 4
  %conv2 = fptosi float %0 to i32
  %gep4 = getelementptr inbounds i32, ptr %a, i64 %iv
  store i32 %conv2, ptr %gep4, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}

define void @vp_fptoui(ptr %a, ptr %b, i64 %N) {
; IF-EVL: VPlan 'Initial VPlan for VF={1},UF>=1'
; IF-EVL-NOT: EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI
;
; IF-EVL: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF={1}' {
; IF-EVL-NEXT: Live-in vp<[[VFUF:%[0-9]+]]> = VF * UF
; IF-EVL-NEXT: Live-in vp<[[VTC:%[0-9]+]]> = vector-trip-count
; IF-EVL-NEXT: Live-in ir<%N> = original trip-count

; IF-EVL: vector.ph:
; IF-EVL-NEXT: Successor(s): vector loop

; IF-EVL: <x1> vector loop: {
; IF-EVL-NEXT:   vector.body:
; IF-EVL-NEXT:     EMIT vp<[[IV:%[0-9]+]]> = CANONICAL-INDUCTION
; IF-EVL-NEXT:     EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<[[EVL_PHI:%[0-9]+]]>  = phi ir<0>, vp<[[IV_NEXT:%.+]]>
; IF-EVL-NEXT:     EMIT vp<[[AVL:%.+]]> = sub ir<%N>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[EVL:%.+]]> = EXPLICIT-VECTOR-LENGTH vp<[[AVL]]>
; IF-EVL-NEXT:     vp<[[ST:%[0-9]+]]> = SCALAR-STEPS vp<[[EVL_PHI]]>, ir<1>, vp<[[EVL]]>
; IF-EVL-NEXT:     CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds ir<%b>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR1:%[0-9]+]]> = vector-pointer ir<[[GEP1]]>
; IF-EVL-NEXT:     WIDEN ir<[[LD1:%.+]]> = vp.load vp<[[PTR1]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     WIDEN-CAST ir<[[FPTOUI:%.+]]> = fptoui ir<[[LD1]]> to i32
; IF-EVL-NEXT:     CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR2:%[0-9]+]]> = vector-pointer ir<[[GEP2]]>
; IF-EVL-NEXT:     WIDEN vp.store vp<[[PTR2]]>, ir<[[FPTOUI]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     EMIT-SCALAR vp<[[CAST:%[0-9]+]]> = zext vp<[[EVL]]> to i64
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT]]> = add vp<[[CAST]]>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT_EXIT:%.+]]> = add vp<[[IV]]>, vp<[[VFUF]]>
; IF-EVL-NEXT:     EMIT branch-on-count vp<[[IV_NEXT_EXIT]]>,  vp<[[VTC]]>
; IF-EVL-NEXT:   No successors
; IF-EVL-NEXT: }

entry:
  br label %loop

loop:
  %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
  %gep = getelementptr inbounds float, ptr %b, i64 %iv
  %0 = load float, ptr %gep, align 4
  %conv2 = fptoui float %0 to i32
  %gep4 = getelementptr inbounds i32, ptr %a, i64 %iv
  store i32 %conv2, ptr %gep4, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}

define void @vp_inttoptr(ptr %a, ptr %b, i64 %N) {
; IF-EVL: VPlan 'Initial VPlan for VF={1},UF>=1'
; IF-EVL-NOT: EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI
;
; IF-EVL: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2},UF={1}' {
; IF-EVL-NEXT: Live-in vp<[[VFUF:%[0-9]+]]> = VF * UF
; IF-EVL-NEXT: Live-in vp<[[VTC:%[0-9]+]]> = vector-trip-count
; IF-EVL-NEXT: Live-in ir<%N> = original trip-count

; IF-EVL: vector.ph:
; IF-EVL-NEXT: Successor(s): vector loop

; IF-EVL: <x1> vector loop: {
; IF-EVL-NEXT:   vector.body:
; IF-EVL-NEXT:     EMIT vp<[[IV:%[0-9]+]]> = CANONICAL-INDUCTION
; IF-EVL-NEXT:     EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<[[EVL_PHI:%[0-9]+]]>  = phi ir<0>, vp<[[IV_NEXT:%.+]]>
; IF-EVL-NEXT:     EMIT vp<[[AVL:%.+]]> = sub ir<%N>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[EVL:%.+]]> = EXPLICIT-VECTOR-LENGTH vp<[[AVL]]>
; IF-EVL-NEXT:     vp<[[ST:%[0-9]+]]> = SCALAR-STEPS vp<[[EVL_PHI]]>, ir<1>, vp<[[EVL]]>
; IF-EVL-NEXT:     CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds ir<%b>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR1:%[0-9]+]]> = vector-pointer ir<[[GEP1]]>
; IF-EVL-NEXT:     WIDEN ir<[[LD1:%.+]]> = vp.load vp<[[PTR1]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     WIDEN-CAST ir<[[INTTOPTR:%.+]]> = inttoptr ir<[[LD1]]> to ptr
; IF-EVL-NEXT:     CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR2:%[0-9]+]]> = vector-pointer ir<[[GEP2]]>
; IF-EVL-NEXT:     WIDEN vp.store vp<[[PTR2]]>, ir<[[INTTOPTR]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     EMIT-SCALAR vp<[[CAST:%[0-9]+]]> = zext vp<[[EVL]]> to i64
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT]]> = add vp<[[CAST]]>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT_EXIT:%.+]]> = add vp<[[IV]]>, vp<[[VFUF]]>
; IF-EVL-NEXT:     EMIT branch-on-count vp<[[IV_NEXT_EXIT]]>,  vp<[[VTC]]>
; IF-EVL-NEXT:   No successors
; IF-EVL-NEXT: }

entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep = getelementptr inbounds i64, ptr %b, i64 %iv
  %0 = load i64, ptr %gep, align 8
  %1 = inttoptr i64 %0 to ptr
  %gep2 = getelementptr inbounds ptr, ptr %a, i64 %iv
  store ptr %1, ptr %gep2, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}

define void @vp_ptrtoint(ptr %a, ptr %b, i64 %N) {
; IF-EVL: VPlan 'Initial VPlan for VF={1},UF>=1'
; IF-EVL-NOT: EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI

; IF-EVL: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2},UF={1}' {
; IF-EVL-NEXT: Live-in vp<[[VFUF:%.+]]> = VF * UF
; IF-EVL-NEXT: Live-in vp<[[VTC:%.+]]> = vector-trip-count
; IF-EVL-NEXT: Live-in ir<[[N:%.+]]> = original trip-count

; IF-EVL: vector.ph:
; IF-EVL-NEXT: Successor(s): vector loop

; IF-EVL: <x1> vector loop: {
; IF-EVL-NEXT:   vector.body:
; IF-EVL-NEXT:     EMIT vp<[[INDEX:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[INDEX_NEXT:%.+]]>
; IF-EVL-NEXT:     EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<[[INDEX_EVL:%.+]]> = phi ir<0>, vp<[[INDEX_EVL_NEXT:%.+]]>
; IF-EVL-NEXT:     ir<[[IV:%.+]]> = WIDEN-INDUCTION  ir<0>, ir<1>, vp<[[EVL]]>
; IF-EVL-NEXT:     EMIT vp<[[AVL:%.+]]> = sub ir<[[N]]>, vp<[[INDEX_EVL]]>
; IF-EVL-NEXT:     EMIT vp<[[EVL:%.+]]> = EXPLICIT-VECTOR-LENGTH vp<[[AVL]]>
; IF-EVL-NEXT:     vp<[[SCALAR_STEPS:%.+]]> = SCALAR-STEPS vp<[[INDEX_EVL]]>, ir<1>, vp<[[EVL]]>
; IF-EVL-NEXT:     WIDEN-GEP Inv[Var] ir<[[GEP:%.+]]> = getelementptr inbounds ir<%b>, ir<[[IV]]>
; IF-EVL-NEXT:     WIDEN-CAST ir<[[PTRTOINT:%.+]]> = ptrtoint ir<[[GEP]]> to i64
; IF-EVL-NEXT:     CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%a>, vp<[[SCALAR_STEPS]]>
; IF-EVL-NEXT:     vp<[[VECTOR_PTR:%.+]]> = vector-pointer ir<[[GEP2]]>
; IF-EVL-NEXT:     WIDEN vp.store vp<[[VECTOR_PTR]]>, ir<[[PTRTOINT]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     EMIT-SCALAR vp<[[ZEXT:%.+]]> = zext vp<[[EVL]]> to i64
; IF-EVL-NEXT:     EMIT vp<[[INDEX_EVL_NEXT]]> = add vp<[[ZEXT]]>, vp<[[INDEX_EVL]]>
; IF-EVL-NEXT:     EMIT vp<[[INDEX_NEXT]]> = add vp<[[INDEX]]>, vp<[[VFUF]]>
; IF-EVL-NEXT:     EMIT branch-on-count vp<[[INDEX_NEXT]]>, vp<[[VTC]]>
; IF-EVL-NEXT:   No successors
; IF-EVL-NEXT: }
; IF-EVL-NEXT: Successor(s): middle.block
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep = getelementptr inbounds i32, ptr %b, i64 %iv
  %0 = ptrtoint ptr %gep to i64
  %gep2 = getelementptr inbounds i64, ptr %a, i64 %iv
  store i64 %0, ptr %gep2, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}
