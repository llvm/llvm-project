; RUN: opt -passes=loop-vectorize -disable-output -vplan-print-after-all -force-vector-width=4 -vplan-verify-each < %s 2>&1 | FileCheck %s --implicit-check-not "VPlan after"
; RUN: opt -passes=loop-vectorize -disable-output -vplan-print-after-all -force-vector-width=4 -vplan-verify-each < %s 2>&1 | FileCheck %s --check-prefix CHECK-DUMP
; REQUIRES: asserts

; Verify that `-vplan-print-after-all` option works.

; CHECK: VPlan after printAfterInitialConstruction
; CHECK: VPlan after VPlanTransforms::clearReductionWrapFlags
; CHECK: VPlan after VPlanTransforms::handleMultiUseReductions
; CHECK: VPlan after VPlanTransforms::handleMaxMinNumReductions
; CHECK: VPlan after VPlanTransforms::handleFindLastReductions
; CHECK: VPlan after VPlanTransforms::createPartialReductions
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
; CHECK: VPlan after mergeBlocksIntoPredecessors
; CHECK: VPlan after licm
; CHECK: VPlan after VPlanTransforms::optimize
; CHECK: VPlan after VPlanTransforms::materializeConstantVectorTripCount
; CHECK: VPlan after VPlanTransforms::unrollByUF
; CHECK: VPlan after VPlanTransforms::materializePacksAndUnpacks
; CHECK: VPlan after VPlanTransforms::materializeBroadcasts
; CHECK: VPlan after VPlanTransforms::replicateByVF
; CHECK: VPlan after printFinalVPlan

; Also verify that VPlans are actually printed (we aren't interested in the
; exact dump content, just that it's performed):

; CHECK-DUMP:      VPlan after printAfterInitialConstruction
; CHECK-DUMP-NEXT: VPlan ' for UF>=1' {
;
; CHECK-DUMP:      VPlan after VPlanTransforms::optimize
; CHECK-DUMP-NEXT: VPlan 'Initial VPlan for VF={4},UF>=1' {
;
; CHECK-DUMP:      VPlan after printFinalVPlan
; CHECK-DUMP-NEXT: VPlan 'Final VPlan for VF={4},UF={1}' {

define void @foo(ptr %ptr, i64 %n) {
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
