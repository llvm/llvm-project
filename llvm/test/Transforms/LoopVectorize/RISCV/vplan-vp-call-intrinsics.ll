; REQUIRES: asserts

; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN: -force-tail-folding-style=data-with-evl \
; RUN: -prefer-predicate-over-epilogue=predicate-dont-vectorize \
; RUN: -mtriple=riscv64 -mattr=+v -riscv-v-vector-bits-max=128 -disable-output < %s 2>&1 | FileCheck --check-prefix=IF-EVL %s

define void @vp_smax(ptr noalias %a, ptr noalias %b, ptr noalias %c, i64 %N) {
; IF-EVL: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF={1}' {
; IF-EVL-NEXT: Live-in vp<[[VFUF:%[0-9]+]]> = VF * UF
; IF-EVL-NEXT: Live-in vp<[[VTC:%[0-9]+]]> = vector-trip-count
; IF-EVL-NEXT: Live-in ir<%N> = original trip-count

; IF-EVL: vector.ph:
; IF-EVL-NEXT: Successor(s): vector loop

; IF-EVL: <x1> vector loop: {
; IF-EVL-NEXT:   vector.body:
; IF-EVL-NEXT:     EMIT vp<[[IV:%[0-9]+]]> = CANONICAL-INDUCTION
; IF-EVL-NEXT:     EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<[[EVL_PHI:%[0-9]+]]> = phi ir<0>, vp<[[IV_NEXT:%[0-9]+]]>
; IF-EVL-NEXT:     EMIT vp<[[AVL:%.+]]> = sub ir<%N>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[EVL:%.+]]> = EXPLICIT-VECTOR-LENGTH vp<[[AVL]]>
; IF-EVL-NEXT:     vp<[[ST:%[0-9]+]]> = SCALAR-STEPS vp<[[EVL_PHI]]>, ir<1>
; IF-EVL-NEXT:     CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds ir<%b>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR1:%[0-9]+]]> = vector-pointer ir<[[GEP1]]>
; IF-EVL-NEXT:     WIDEN ir<[[LD1:%.+]]> = vp.load vp<[[PTR1]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%c>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR2:%[0-9]+]]> = vector-pointer ir<[[GEP2]]>
; IF-EVL-NEXT:     WIDEN ir<[[LD2:%.+]]> = vp.load vp<[[PTR2]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     WIDEN-INTRINSIC ir<[[SMAX:%.+]]> = call llvm.smax(ir<[[LD1]]>, ir<[[LD2]]>)
; IF-EVL-NEXT:     CLONE ir<[[GEP3:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR3:%[0-9]+]]> = vector-pointer ir<[[GEP3]]>
; IF-EVL-NEXT:     WIDEN vp.store vp<[[PTR3]]>, ir<[[SMAX]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     SCALAR-CAST vp<[[CAST:%[0-9]+]]> = zext vp<[[EVL]]> to i64
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT]]> = add vp<[[CAST]]>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT_EXIT:%[0-9]+]]> = add vp<[[IV]]>, vp<[[VFUF]]>
; IF-EVL-NEXT:     EMIT branch-on-count  vp<[[IV_NEXT_EXIT]]>, vp<[[VTC]]>
; IF-EVL-NEXT:   No successors
; IF-EVL-NEXT: }

entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx3 = getelementptr inbounds i32, ptr %c, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx3, align 4
  %. = tail call i32 @llvm.smax.i32(i32 %0, i32 %1)
  %arrayidx11 = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  store i32 %., ptr %arrayidx11, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %N
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret void
}

define void @vp_smin(ptr noalias %a, ptr noalias %b, ptr noalias %c, i64 %N) {
; IF-EVL: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF={1}' {
; IF-EVL-NEXT: Live-in vp<[[VFUF:%[0-9]+]]> = VF * UF
; IF-EVL-NEXT: Live-in vp<[[VTC:%[0-9]+]]> = vector-trip-count
; IF-EVL-NEXT: Live-in ir<%N> = original trip-count

; IF-EVL: vector.ph:
; IF-EVL-NEXT: Successor(s): vector loop

; IF-EVL: <x1> vector loop: {
; IF-EVL-NEXT:   vector.body:
; IF-EVL-NEXT:     EMIT vp<[[IV:%[0-9]+]]> = CANONICAL-INDUCTION
; IF-EVL-NEXT:     EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<[[EVL_PHI:%[0-9]+]]> = phi ir<0>, vp<[[IV_NEXT:%[0-9]+]]>
; IF-EVL-NEXT:     EMIT vp<[[AVL:%.+]]> = sub ir<%N>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[EVL:%.+]]> = EXPLICIT-VECTOR-LENGTH vp<[[AVL]]>
; IF-EVL-NEXT:     vp<[[ST:%[0-9]+]]> = SCALAR-STEPS vp<[[EVL_PHI]]>, ir<1>
; IF-EVL-NEXT:     CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds ir<%b>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR1:%[0-9]+]]> = vector-pointer ir<[[GEP1]]>
; IF-EVL-NEXT:     WIDEN ir<[[LD1:%.+]]> = vp.load vp<[[PTR1]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%c>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR2:%[0-9]+]]> = vector-pointer ir<[[GEP2]]>
; IF-EVL-NEXT:     WIDEN ir<[[LD2:%.+]]> = vp.load vp<[[PTR2]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     WIDEN-INTRINSIC ir<[[SMIN:%.+]]> = call llvm.smin(ir<[[LD1]]>, ir<[[LD2]]>)
; IF-EVL-NEXT:     CLONE ir<[[GEP3:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR3:%[0-9]+]]> = vector-pointer ir<[[GEP3]]>
; IF-EVL-NEXT:     WIDEN vp.store vp<[[PTR3]]>, ir<[[SMIN]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     SCALAR-CAST vp<[[CAST:%[0-9]+]]> = zext vp<[[EVL]]> to i64
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT]]> = add vp<[[CAST]]>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT_EXIT:%[0-9]+]]> = add vp<[[IV]]>, vp<[[VFUF]]>
; IF-EVL-NEXT:     EMIT branch-on-count  vp<[[IV_NEXT_EXIT]]>, vp<[[VTC]]>
; IF-EVL-NEXT:   No successors
; IF-EVL-NEXT: }

entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx3 = getelementptr inbounds i32, ptr %c, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx3, align 4
  %. = tail call i32 @llvm.smin.i32(i32 %0, i32 %1)
  %arrayidx11 = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  store i32 %., ptr %arrayidx11, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %N
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret void
}

define void @vp_umax(ptr noalias %a, ptr noalias %b, ptr noalias %c, i64 %N) {
; IF-EVL: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF={1}' {
; IF-EVL-NEXT: Live-in vp<[[VFUF:%[0-9]+]]> = VF * UF
; IF-EVL-NEXT: Live-in vp<[[VTC:%[0-9]+]]> = vector-trip-count
; IF-EVL-NEXT: Live-in ir<%N> = original trip-count

; IF-EVL: vector.ph:
; IF-EVL-NEXT: Successor(s): vector loop

; IF-EVL: <x1> vector loop: {
; IF-EVL-NEXT:   vector.body:
; IF-EVL-NEXT:     EMIT vp<[[IV:%[0-9]+]]> = CANONICAL-INDUCTION
; IF-EVL-NEXT:     EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<[[EVL_PHI:%[0-9]+]]> = phi ir<0>, vp<[[IV_NEXT:%[0-9]+]]>
; IF-EVL-NEXT:     EMIT vp<[[AVL:%.+]]> = sub ir<%N>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[EVL:%.+]]> = EXPLICIT-VECTOR-LENGTH vp<[[AVL]]>
; IF-EVL-NEXT:     vp<[[ST:%[0-9]+]]> = SCALAR-STEPS vp<[[EVL_PHI]]>, ir<1>
; IF-EVL-NEXT:     CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds ir<%b>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR1:%[0-9]+]]> = vector-pointer ir<[[GEP1]]>
; IF-EVL-NEXT:     WIDEN ir<[[LD1:%.+]]> = vp.load vp<[[PTR1]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%c>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR2:%[0-9]+]]> = vector-pointer ir<[[GEP2]]>
; IF-EVL-NEXT:     WIDEN ir<[[LD2:%.+]]> = vp.load vp<[[PTR2]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     WIDEN-INTRINSIC ir<[[UMAX:%.+]]> = call llvm.umax(ir<[[LD1]]>, ir<[[LD2]]>)
; IF-EVL-NEXT:     CLONE ir<[[GEP3:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR3:%[0-9]+]]> = vector-pointer ir<[[GEP3]]>
; IF-EVL-NEXT:     WIDEN vp.store vp<[[PTR3]]>, ir<[[UMAX]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     SCALAR-CAST vp<[[CAST:%[0-9]+]]> = zext vp<[[EVL]]> to i64
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT]]> = add vp<[[CAST]]>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT_EXIT:%[0-9]+]]> = add vp<[[IV]]>, vp<[[VFUF]]>
; IF-EVL-NEXT:     EMIT branch-on-count  vp<[[IV_NEXT_EXIT]]>, vp<[[VTC]]>
; IF-EVL-NEXT:   No successors
; IF-EVL-NEXT: }

entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx3 = getelementptr inbounds i32, ptr %c, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx3, align 4
  %. = tail call i32 @llvm.umax.i32(i32 %0, i32 %1)
  %arrayidx11 = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  store i32 %., ptr %arrayidx11, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %N
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret void
}

define void @vp_umin(ptr noalias %a, ptr noalias %b, ptr noalias %c, i64 %N) {
; IF-EVL: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF={1}' {
; IF-EVL-NEXT: Live-in vp<[[VFUF:%[0-9]+]]> = VF * UF
; IF-EVL-NEXT: Live-in vp<[[VTC:%[0-9]+]]> = vector-trip-count
; IF-EVL-NEXT: Live-in ir<%N> = original trip-count

; IF-EVL: vector.ph:
; IF-EVL-NEXT: Successor(s): vector loop

; IF-EVL: <x1> vector loop: {
; IF-EVL-NEXT:   vector.body:
; IF-EVL-NEXT:     EMIT vp<[[IV:%[0-9]+]]> = CANONICAL-INDUCTION
; IF-EVL-NEXT:     EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<[[EVL_PHI:%[0-9]+]]> = phi ir<0>, vp<[[IV_NEXT:%[0-9]+]]>
; IF-EVL-NEXT:     EMIT vp<[[AVL:%.+]]> = sub ir<%N>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[EVL:%.+]]> = EXPLICIT-VECTOR-LENGTH vp<[[AVL]]>
; IF-EVL-NEXT:     vp<[[ST:%[0-9]+]]> = SCALAR-STEPS vp<[[EVL_PHI]]>, ir<1>
; IF-EVL-NEXT:     CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds ir<%b>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR1:%[0-9]+]]> = vector-pointer ir<[[GEP1]]>
; IF-EVL-NEXT:     WIDEN ir<[[LD1:%.+]]> = vp.load vp<[[PTR1]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%c>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR2:%[0-9]+]]> = vector-pointer ir<[[GEP2]]>
; IF-EVL-NEXT:     WIDEN ir<[[LD2:%.+]]> = vp.load vp<[[PTR2]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     WIDEN-INTRINSIC ir<[[UMIN:%.+]]> = call llvm.umin(ir<[[LD1]]>, ir<[[LD2]]>)
; IF-EVL-NEXT:     CLONE ir<[[GEP3:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; IF-EVL-NEXT:     vp<[[PTR3:%[0-9]+]]> = vector-pointer ir<[[GEP3]]>
; IF-EVL-NEXT:     WIDEN vp.store vp<[[PTR3]]>, ir<[[UMIN]]>, vp<[[EVL]]>
; IF-EVL-NEXT:     SCALAR-CAST vp<[[CAST:%[0-9]+]]> = zext vp<[[EVL]]> to i64
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT]]> = add vp<[[CAST]]>, vp<[[EVL_PHI]]>
; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT_EXIT:%[0-9]+]]> = add vp<[[IV]]>, vp<[[VFUF]]>
; IF-EVL-NEXT:     EMIT branch-on-count  vp<[[IV_NEXT_EXIT]]>, vp<[[VTC]]>
; IF-EVL-NEXT:   No successors
; IF-EVL-NEXT: }

entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx3 = getelementptr inbounds i32, ptr %c, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx3, align 4
  %. = tail call i32 @llvm.umin.i32(i32 %0, i32 %1)
  %arrayidx11 = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  store i32 %., ptr %arrayidx11, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %N
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret void
}

declare i32 @llvm.smax.i32(i32, i32)
declare i32 @llvm.smin.i32(i32, i32)
declare i32 @llvm.umax.i32(i32, i32)
declare i32 @llvm.umin.i32(i32, i32)