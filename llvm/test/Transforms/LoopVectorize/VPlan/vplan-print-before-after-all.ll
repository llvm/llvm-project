; RUN: opt -passes=loop-vectorize -disable-output -vplan-print-before-all -force-vector-width=4 -vplan-verify-each < %s 2>&1 | FileCheck %s -check-prefixes=CHECK,CHECK-BEFORE -DBEFORE_OR_AFTER=before --implicit-check-not "VPlan for loop in 'foo' before"
; RUN: opt -passes=loop-vectorize -disable-output -vplan-print-before-all -force-vector-width=4 -vplan-verify-each < %s 2>&1 | FileCheck %s -DBEFORE_OR_AFTER=before --check-prefix CHECK-DUMP
; RUN: opt -passes=loop-vectorize -disable-output -vplan-print-after-all -force-vector-width=4 -vplan-verify-each < %s 2>&1 | FileCheck %s -check-prefixes=CHECK,CHECK-AFTER -DBEFORE_OR_AFTER=after --implicit-check-not "VPlan for loop in 'foo' after"
; RUN: opt -passes=loop-vectorize -disable-output -vplan-print-after-all -force-vector-width=4 -vplan-verify-each < %s 2>&1 | FileCheck %s -DBEFORE_OR_AFTER=after --check-prefix CHECK-DUMP

; Verify that `-vplan-print-before/after-all` option works.

; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] printAfterInitialConstruction
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::replaceSymbolicStrides
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::simplifyRecipes
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::removeDeadRecipes
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::createHeaderPhiRecipes
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::replaceSymbolicStrides
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::finalizeSCEVPredicates
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::addMiddleCheck
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::handleEarlyExits
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::createLoopRegions
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::introduceMasksAndLinearize
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::createInLoopReductionRecipes
; CHECK-BEFORE: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::makeMemOpWideningDecisions
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] lowerMemoryIdioms
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] scalarizeMemOpsWithIrregularTypes
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] widenConsecutiveMemOps
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] delegateMemOpWideningToLegacyCM
; CHECK-AFTER: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::makeMemOpWideningDecisions
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::makeScalarizationDecisions
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::makeCallWideningDecisions
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::adjustFirstOrderRecurrenceMiddleUsers
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::clearReductionWrapFlags
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::optimizeFindIVReductions
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::optimizeInductionLiveOutUsers
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::handleMultiUseReductions
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::handleMaxMinNumReductions
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::handleFindLastReductions
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::removeBranchOnConst
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::createPartialReductions
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::convertToAbstractRecipes
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::createInterleaveGroups
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::convertToStridedAccesses
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::dropPoisonGeneratingRecipes
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::hoistPredicatedLoads
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::sinkPredicatedStores
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::truncateToMinimalBitwidths
; CHECK-BEFORE: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::optimize
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] removeRedundantInductionCasts
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] reassociateHeaderMask
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] simplifyRecipes
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] removeDeadRecipes
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] simplifyBlends
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] legalizeAndOptimizeInductions
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] narrowToSingleScalarRecipes
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] removeRedundantExpandSCEVRecipes
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] reassociateHeaderMask
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] simplifyRecipes
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] removeBranchOnConst
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] simplifyReverses
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] removeDeadRecipes
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] createAndOptimizeReplicateRegions
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] mergeBlocksIntoPredecessors
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] licm
; CHECK-AFTER: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::optimize
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::narrowInterleaveGroups
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] printOptimizedVPlan
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::addMinimumIterationCheck
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::replaceWideCanonicalIVWithWideIV
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::unrollByUF
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::materializePacksAndUnpacks
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::materializeBroadcasts
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::replicateByVF
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::materializeConstantVectorTripCount
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::optimizeForVFAndUF
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::simplifyRecipes
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::removeBranchOnConst
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::removeDeadRecipes
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::convertToConcreteRecipes
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::convertEVLExitCond
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::dissolveLoopRegions
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::expandBranchOnTwoConds
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::convertToVariableLengthStep
; CHECK: VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] printFinalVPlan

; Also verify that VPlans are actually printed (we aren't interested in the
; exact dump content, just that it's performed):

; CHECK-DUMP:      VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] printAfterInitialConstruction
; CHECK-DUMP-NEXT: VPlan ' for UF>=1' {
;
; CHECK-DUMP:      VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] VPlanTransforms::optimize{{$}}
; CHECK-DUMP-NEXT: VPlan 'Initial VPlan for VF={4},UF>=1' {
;
; CHECK-DUMP:      VPlan for loop in 'foo' [[BEFORE_OR_AFTER]] printFinalVPlan
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
