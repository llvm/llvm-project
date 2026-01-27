; REQUIRES: asserts

; Verify that `-vplan-print-after-all` option works.

; RUN: opt -passes=loop-vectorize -disable-output -vplan-print-after-all -force-vector-width=4 -vplan-verify-each < %s 2>&1 | FileCheck %s --implicit-check-not "VPlan after"

; CHECK: VPlan after printAfterInitialConstruction
; CHECK: VPlan after VPlanTransforms::clearReductionWrapFlags
; CHECK: VPlan after VPlanTransforms::handleMultiUseReductions
; CHECK: VPlan after VPlanTransforms::handleMaxMinNumReductions
; CHECK: VPlan after VPlanTransforms::handleFindLastReductions
; CHECK: VPlan after VPlanTransforms::convertToAbstractRecipes
; CHECK: VPlan after VPlanTransforms::createInterleaveGroups
; CHECK: VPlan after VPlanTransforms::replaceSymbolicStrides
; CHECK: VPlan after VPlanTransforms::dropPoisonGeneratingRecipes
; CHECK: VPlan after VPlanTransforms::adjustFixedOrderRecurrences
; CHECK: VPlan after VPlanTransforms::truncateToMinimalBitwidths
; CHECK: VPlan after removeRedundantCanonicalIVs
; CHECK: VPlan after removeRedundantInductionCasts
; CHECK: VPlan after simplifyRecipes
; CHECK: VPlan after removeDeadRecipes
; CHECK: VPlan after simplifyBlends
; CHECK: VPlan after legalizeAndOptimizeInductions
; CHECK: VPlan after narrowToSingleScalarRecipes
; CHECK: VPlan after removeRedundantExpandSCEVRecipes
; CHECK: VPlan after simplifyRecipes
; CHECK: VPlan after removeBranchOnConst
; CHECK: VPlan after removeDeadRecipes
; CHECK: VPlan after createAndOptimizeReplicateRegions
; CHECK: VPlan after hoistInvariantLoads
; CHECK: VPlan after mergeBlocksIntoPredecessors
; CHECK: VPlan after licm
; CHECK: VPlan after VPlanTransforms::optimize
; CHECK: VPlan after VPlanTransforms::materializeConstantVectorTripCount
; CHECK: VPlan after VPlanTransforms::unrollByUF
; CHECK: VPlan after VPlanTransforms::materializePacksAndUnpacks
; CHECK: VPlan after VPlanTransforms::materializeBroadcasts
; CHECK: VPlan after VPlanTransforms::replicateByVF
; CHECK: VPlan after printFinalVPlan
; Also verify that VPlan is actually printed:
; CHECK-NEXT: VPlan 'Final VPlan for VF={4},UF={1}' {
; CHECK-NEXT: Live-in ir<%smax> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<entry>:
; CHECK-NEXT:   IR   %smax = call i64 @llvm.smax.i64(i64 %n, i64 1)
; CHECK-NEXT:   EMIT vp<%min.iters.check> = icmp ult ir<%smax>, ir<4>
; CHECK-NEXT:   EMIT branch-on-cond vp<%min.iters.check>
; CHECK-NEXT: Successor(s): ir-bb<scalar.ph>, vector.ph
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT:   EMIT vp<%n.mod.vf> = urem ir<%smax>, ir<4>
; CHECK-NEXT:   EMIT vp<%n.vec> = sub ir<%smax>, vp<%n.mod.vf>
; CHECK-NEXT:   EMIT vp<%3> = step-vector i64
; CHECK-NEXT:   EMIT vp<%4> = broadcast ir<4>
; CHECK-NEXT: Successor(s): vector.body
; CHECK-EMPTY:
; CHECK-NEXT: vector.body:
; CHECK-NEXT:   EMIT-SCALAR vp<%index> = phi [ ir<0>, vector.ph ], [ vp<%index.next>, vector.body ]
; CHECK-NEXT:   WIDEN-PHI ir<%iv> = phi [ vp<%3>, vector.ph ], [ vp<%vec.ind.next>, vector.body ]
; CHECK-NEXT:   CLONE ir<%gep> = getelementptr ir<%ptr>, vp<%index>

define void @foo(ptr dereferenceable(1024) %ptr, i64 %n) {
entry:
  br label %header

header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %header ]
  %gep = getelementptr i64, ptr %ptr, i64 %iv
  store i64 %iv, ptr %gep
  %iv.next = add nsw i64 %iv, 1
  %exitcond = icmp slt i64 %iv.next, %n
  br i1 %exitcond, label %header, label %exit

exit:
  ret void
}
