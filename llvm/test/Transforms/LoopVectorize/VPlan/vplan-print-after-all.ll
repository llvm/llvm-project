; RUN: opt -passes=loop-vectorize -disable-output -vplan-print-after-all -force-vector-width=4 -vplan-verify-each < %s 2>&1 | FileCheck %s --implicit-check-not "VPlan for loop in 'foo' after"
; RUN: opt -passes=loop-vectorize -disable-output -vplan-print-after-all -force-vector-width=4 -vplan-verify-each < %s 2>&1 | FileCheck %s --check-prefix CHECK-DUMP

; Verify that `-vplan-print-after-all` option works.

; CHECK: VPlan for loop in 'foo' after printAfterInitialConstruction
; CHECK: VPlan for loop in 'foo' after VPlanTransforms::introduceMasksAndLinearize
; CHECK: VPlan for loop in 'foo' after lowerMemoryIdioms
; CHECK: VPlan for loop in 'foo' after scalarizeMemOpsWithIrregularTypes
; CHECK: VPlan for loop in 'foo' after delegateMemOpWideningToLegacyCM
; CHECK: VPlan for loop in 'foo' after VPlanTransforms::makeMemOpWideningDecisions
; CHECK: VPlan for loop in 'foo' after VPlanTransforms::clearReductionWrapFlags
; CHECK: VPlan for loop in 'foo' after VPlanTransforms::optimizeFindIVReductions
; CHECK: VPlan for loop in 'foo' after VPlanTransforms::handleMultiUseReductions
; CHECK: VPlan for loop in 'foo' after VPlanTransforms::handleMaxMinNumReductions
; CHECK: VPlan for loop in 'foo' after VPlanTransforms::handleFindLastReductions
; CHECK: VPlan for loop in 'foo' after VPlanTransforms::createPartialReductions
; CHECK: VPlan for loop in 'foo' after VPlanTransforms::convertToAbstractRecipes
; CHECK: VPlan for loop in 'foo' after VPlanTransforms::createInterleaveGroups
; CHECK: VPlan for loop in 'foo' after VPlanTransforms::replaceSymbolicStrides
; CHECK: VPlan for loop in 'foo' after VPlanTransforms::dropPoisonGeneratingRecipes
; CHECK: VPlan for loop in 'foo' after VPlanTransforms::adjustFixedOrderRecurrences
; CHECK: VPlan for loop in 'foo' after VPlanTransforms::truncateToMinimalBitwidths
; CHECK: VPlan for loop in 'foo' after removeRedundantCanonicalIVs
; CHECK: VPlan for loop in 'foo' after removeRedundantInductionCasts
; CHECK: VPlan for loop in 'foo' after reassociateHeaderMask
; CHECK: VPlan for loop in 'foo' after simplifyRecipes
; CHECK: VPlan for loop in 'foo' after removeDeadRecipes
; CHECK: VPlan for loop in 'foo' after simplifyBlends
; CHECK: VPlan for loop in 'foo' after legalizeAndOptimizeInductions
; CHECK: VPlan for loop in 'foo' after narrowToSingleScalarRecipes
; CHECK: VPlan for loop in 'foo' after removeRedundantExpandSCEVRecipes
; CHECK: VPlan for loop in 'foo' after reassociateHeaderMask
; CHECK: VPlan for loop in 'foo' after simplifyRecipes
; CHECK: VPlan for loop in 'foo' after removeBranchOnConst
; CHECK: VPlan for loop in 'foo' after removeDeadRecipes
; CHECK: VPlan for loop in 'foo' after createAndOptimizeReplicateRegions
; CHECK: VPlan for loop in 'foo' after hoistInvariantLoads
; CHECK: VPlan for loop in 'foo' after mergeBlocksIntoPredecessors
; CHECK: VPlan for loop in 'foo' after licm
; CHECK: VPlan for loop in 'foo' after VPlanTransforms::optimize
; CHECK: VPlan for loop in 'foo' after VPlanTransforms::materializeConstantVectorTripCount
; CHECK: VPlan for loop in 'foo' after VPlanTransforms::unrollByUF
; CHECK: VPlan for loop in 'foo' after VPlanTransforms::materializePacksAndUnpacks
; CHECK: VPlan for loop in 'foo' after VPlanTransforms::materializeBroadcasts
; CHECK: VPlan for loop in 'foo' after VPlanTransforms::replicateByVF
; CHECK: VPlan for loop in 'foo' after printFinalVPlan

; Also verify that VPlans are actually printed (we aren't interested in the
; exact dump content, just that it's performed):

; CHECK-DUMP:      VPlan for loop in 'foo' after printAfterInitialConstruction
; CHECK-DUMP-NEXT: VPlan ' for UF>=1' {
;
; CHECK-DUMP:      VPlan for loop in 'foo' after VPlanTransforms::optimize{{$}}
; CHECK-DUMP-NEXT: VPlan 'Initial VPlan for VF={4},UF>=1' {
;
; CHECK-DUMP:      VPlan for loop in 'foo' after printFinalVPlan
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
