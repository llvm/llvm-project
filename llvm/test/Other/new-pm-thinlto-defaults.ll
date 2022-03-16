; The IR below was crafted so as:
; 1) To have a loop, so we create a loop pass manager
; 2) To be "immutable" in the sense that no pass in the standard
;    pipeline will modify it.
; Since no transformations take place, we don't expect any analyses
; to be invalidated.
; Any invalidation that shows up here is a bug, unless we started modifying
; the IR, in which case we need to make it immutable harder.
;
; Prelink pipelines:
; RUN: opt -disable-verify -verify-cfg-preserved=0 -eagerly-invalidate-analyses=0 -debug-pass-manager \
; RUN:     -passes='thinlto-pre-link<O1>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=CHECK-O,CHECK-O1,CHECK-PRELINK-O,CHECK-PRELINK-O-NODIS
; RUN: opt -disable-verify -verify-cfg-preserved=0 -eagerly-invalidate-analyses=0 -debug-pass-manager \
; RUN:     -passes='thinlto-pre-link<O2>' -S  %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=CHECK-O,CHECK-O2,CHECK-O23SZ,CHECK-PRELINK-O,CHECK-PRELINK-O-NODIS
; RUN: opt -disable-verify -verify-cfg-preserved=0 -eagerly-invalidate-analyses=0 -debug-pass-manager \
; RUN:     -passes='thinlto-pre-link<O3>' -S -passes-ep-pipeline-start='no-op-module' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=CHECK-O,CHECK-O3,CHECK-O23SZ,CHECK-PRELINK-O,CHECK-PRELINK-O-NODIS,CHECK-EP-PIPELINE-START
; RUN: opt -disable-verify -verify-cfg-preserved=0 -eagerly-invalidate-analyses=0 -debug-pass-manager \
; RUN:     -passes='thinlto-pre-link<Os>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=CHECK-O,CHECK-Os,CHECK-O23SZ,CHECK-PRELINK-O,CHECK-PRELINK-O-NODIS
; RUN: opt -disable-verify -verify-cfg-preserved=0 -eagerly-invalidate-analyses=0 -debug-pass-manager \
; RUN:     -passes='thinlto-pre-link<Oz>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=CHECK-O,CHECK-Oz,CHECK-O23SZ,CHECK-PRELINK-O,CHECK-PRELINK-O-NODIS
; RUN: opt -disable-verify -verify-cfg-preserved=0 -eagerly-invalidate-analyses=0 -debug-pass-manager -new-pm-debug-info-for-profiling \
; RUN:     -passes='thinlto-pre-link<O2>' -S  %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=CHECK-DIS,CHECK-O,CHECK-O2,CHECK-O23SZ,CHECK-PRELINK-O
;
; Postlink pipelines:
; RUN: opt -disable-verify -verify-cfg-preserved=0 -eagerly-invalidate-analyses=0 -debug-pass-manager \
; RUN:     -passes='thinlto<O1>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=CHECK-O,CHECK-O1,CHECK-POSTLINK-O,%llvmcheckext
; RUN: opt -disable-verify -verify-cfg-preserved=0 -eagerly-invalidate-analyses=0 -debug-pass-manager \
; RUN:     -passes='thinlto<O2>' -S  %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=CHECK-O,CHECK-O2,CHECK-O23SZ,CHECK-POSTLINK-O,%llvmcheckext,CHECK-POSTLINK-O2
; RUN: opt -disable-verify -verify-cfg-preserved=0 -eagerly-invalidate-analyses=0 -debug-pass-manager -passes-ep-pipeline-start='no-op-module' \
; RUN:     -passes='thinlto<O3>' -S  %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=CHECK-O,CHECK-O3,CHECK-O23SZ,CHECK-POSTLINK-O,%llvmcheckext,CHECK-POSTLINK-O3
; RUN: opt -disable-verify -verify-cfg-preserved=0 -eagerly-invalidate-analyses=0 -debug-pass-manager \
; RUN:     -passes='thinlto<Os>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=CHECK-O,CHECK-Os,CHECK-O23SZ,CHECK-POSTLINK-O,%llvmcheckext,CHECK-POSTLINK-Os
; RUN: opt -disable-verify -verify-cfg-preserved=0 -eagerly-invalidate-analyses=0 -debug-pass-manager \
; RUN:     -passes='thinlto<Oz>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=CHECK-O,CHECK-Oz,CHECK-O23SZ,CHECK-POSTLINK-O,%llvmcheckext
; RUN: opt -disable-verify -verify-cfg-preserved=0 -eagerly-invalidate-analyses=0 -debug-pass-manager -new-pm-debug-info-for-profiling \
; RUN:     -passes='thinlto<O2>' -S  %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=CHECK-O,CHECK-O2,CHECK-O23SZ,CHECK-POSTLINK-O,%llvmcheckext,CHECK-POSTLINK-O2

; Suppress FileCheck --allow-unused-prefixes=false diagnostics.
; CHECK-NOEXT: {{^}}

; CHECK-O: Running pass: Annotation2Metadata
; CHECK-O-NEXT: Running pass: ForceFunctionAttrsPass
; CHECK-EP-PIPELINE-START-NEXT: Running pass: NoOpModulePass
; CHECK-DIS-NEXT: Running analysis: InnerAnalysisManagerProxy
; CHECK-DIS-NEXT: Running pass: AddDiscriminatorsPass
; CHECK-POSTLINK-O-NEXT: Running pass: PGOIndirectCallPromotion
; CHECK-POSTLINK-O-NEXT: Running analysis: ProfileSummaryAnalysis
; CHECK-POSTLINK-O-NEXT: Running analysis: InnerAnalysisManagerProxy
; CHECK-POSTLINK-O-NEXT: Running analysis: OptimizationRemarkEmitterAnalysis
; CHECK-O-NEXT: Running pass: InferFunctionAttrsPass
; CHECK-PRELINK-O-NODIS-NEXT: Running analysis: InnerAnalysisManagerProxy
; CHECK-O-NEXT: Running analysis: TargetLibraryAnalysis
; CHECK-O-NEXT: Running pass: LowerExpectIntrinsicPass
; CHECK-O-NEXT: Running pass: SimplifyCFGPass
; CHECK-O-NEXT: Running analysis: TargetIRAnalysis
; CHECK-O-NEXT: Running analysis: AssumptionAnalysis
; CHECK-O-NEXT: Running pass: SROAPass
; CHECK-O-NEXT: Running analysis: DominatorTreeAnalysis
; CHECK-O-NEXT: Running pass: EarlyCSEPass
; CHECK-O-NEXT: Running analysis: TargetLibraryAnalysis
; CHECK-O-NEXT: Running pass: CoroEarlyPass
; CHECK-O3-NEXT: Running pass: CallSiteSplittingPass
; CHECK-O-NEXT: Running pass: OpenMPOptPass
; CHECK-POSTLINK-O-NEXT: Running pass: LowerTypeTestsPass
; CHECK-O-NEXT: Running pass: IPSCCPPass
; CHECK-O-NEXT: Running pass: CalledValuePropagationPass
; CHECK-O-NEXT: Running pass: GlobalOptPass
; CHECK-O-NEXT: Running pass: PromotePass
; CHECK-O-NEXT: Running pass: DeadArgumentEliminationPass
; CHECK-O-NEXT: Running pass: InstCombinePass
; CHECK-PRELINK-O-NEXT: Running analysis: OptimizationRemarkEmitterAnalysis
; CHECK-O-NEXT: Running analysis: AAManager
; CHECK-O-NEXT: Running analysis: BasicAA
; CHECK-O-NEXT: Running analysis: ScopedNoAliasAA
; CHECK-O-NEXT: Running analysis: TypeBasedAA
; CHECK-O-NEXT: Running analysis: OuterAnalysisManagerProxy
; CHECK-O-NEXT: Running pass: SimplifyCFGPass
; CHECK-O-NEXT: Running pass: ModuleInlinerWrapperPass
; CHECK-O-NEXT: Running analysis: InlineAdvisorAnalysis
; CHECK-O-NEXT: Running pass: RequireAnalysisPass<{{.*}}GlobalsAA
; CHECK-O-NEXT: Running analysis: GlobalsAA
; CHECK-O-NEXT: Running analysis: CallGraphAnalysis
; CHECK-O-NEXT: Running pass: InvalidateAnalysisPass<{{.*}}AAManager
; CHECK-O-NEXT: Invalidating analysis: AAManager
; CHECK-O-NEXT: Running pass: RequireAnalysisPass<{{.*}}ProfileSummaryAnalysis
; CHECK-PRELINK-O-NEXT: Running analysis: ProfileSummaryAnalysis
; CHECK-O-NEXT: Running analysis: InnerAnalysisManagerProxy
; CHECK-O-NEXT: Running analysis: LazyCallGraphAnalysis
; CHECK-O-NEXT: Running analysis: FunctionAnalysisManagerCGSCCProxy
; CHECK-O-NEXT: Running analysis: OuterAnalysisManagerProxy
; CHECK-O-NEXT: Running pass: DevirtSCCRepeatedPass
; CHECK-O-NEXT: Running pass: InlinerPass
; CHECK-O-NEXT: Running pass: InlinerPass
; CHECK-O-NEXT: Running pass: PostOrderFunctionAttrsPass
; CHECK-O-NEXT: Running analysis: AAManager
; CHECK-O3-NEXT: Running pass: ArgumentPromotionPass
; CHECK-O2-NEXT: Running pass: OpenMPOptCGSCCPass on (foo)
; CHECK-O3-NEXT: Running pass: OpenMPOptCGSCCPass on (foo)
; CHECK-O-NEXT: Running pass: SROAPass
; CHECK-O-NEXT: Running pass: EarlyCSEPass
; CHECK-O-NEXT: Running analysis: MemorySSAAnalysis
; CHECK-O23SZ-NEXT: Running pass: SpeculativeExecutionPass
; CHECK-O23SZ-NEXT: Running pass: JumpThreadingPass
; CHECK-O23SZ-NEXT: Running analysis: LazyValueAnalysis
; CHECK-O23SZ-NEXT: Running pass: CorrelatedValuePropagationPass
; CHECK-O23SZ-NEXT: Invalidating analysis: LazyValueAnalysis
; CHECK-O-NEXT: Running pass: SimplifyCFGPass
; CHECK-O-NEXT: Running pass: InstCombinePass
; CHECK-O3-NEXT: Running pass: AggressiveInstCombinePass
; CHECK-O1-NEXT: Running pass: LibCallsShrinkWrapPass
; CHECK-O2-NEXT: Running pass: LibCallsShrinkWrapPass
; CHECK-O3-NEXT: Running pass: LibCallsShrinkWrapPass
; CHECK-O23SZ-NEXT: Running pass: TailCallElimPass
; CHECK-O-NEXT: Running pass: SimplifyCFGPass
; CHECK-O-NEXT: Running pass: ReassociatePass
; CHECK-O-NEXT: Running pass: RequireAnalysisPass<{{.*}}OptimizationRemarkEmitterAnalysis
; CHECK-O-NEXT: Running pass: LoopSimplifyPass
; CHECK-O-NEXT: Running analysis: LoopAnalysis
; CHECK-O-NEXT: Running pass: LCSSAPass
; CHECK-O-NEXT: Running analysis: ScalarEvolutionAnalysis
; CHECK-O-NEXT: Running analysis: InnerAnalysisManagerProxy
; CHECK-O-NEXT: Running pass: LoopInstSimplifyPass
; CHECK-O-NEXT: Running pass: LoopSimplifyCFGPass
; CHECK-O-NEXT: Running pass: LICM
; CHECK-O-NEXT: Running pass: LoopRotatePass
; CHECK-O-NEXT: Running pass: LICM
; CHECK-O-NEXT: Running pass: SimpleLoopUnswitchPass
; CHECK-O-NEXT: Running pass: SimplifyCFGPass
; CHECK-O-NEXT: Running pass: InstCombinePass
; CHECK-O-NEXT: Running pass: LoopSimplifyPass
; CHECK-O-NEXT: Running pass: LCSSAPass
; CHECK-O-NEXT: Running pass: LoopIdiomRecognizePass
; CHECK-O-NEXT: Running pass: IndVarSimplifyPass
; CHECK-O-NEXT: Running pass: LoopDeletionPass
; CHECK-O-NEXT: Running pass: LoopFullUnrollPass
; CHECK-O-NEXT: Running pass: SROAPass on foo
; CHECK-Os-NEXT: Running pass: MergedLoadStoreMotionPass
; CHECK-Os-NEXT: Running pass: GVNPass
; CHECK-Os-NEXT: Running analysis: MemoryDependenceAnalysis
; CHECK-Os-NEXT: Running analysis: PhiValuesAnalysis
; CHECK-Oz-NEXT: Running pass: MergedLoadStoreMotionPass
; CHECK-Oz-NEXT: Running pass: GVNPass
; CHECK-Oz-NEXT: Running analysis: MemoryDependenceAnalysis
; CHECK-Oz-NEXT: Running analysis: PhiValuesAnalysis
; CHECK-O2-NEXT: Running pass: MergedLoadStoreMotionPass
; CHECK-O2-NEXT: Running pass: GVNPass
; CHECK-O2-NEXT: Running analysis: MemoryDependenceAnalysis
; CHECK-O2-NEXT: Running analysis: PhiValuesAnalysis
; CHECK-O3-NEXT: Running pass: MergedLoadStoreMotionPass
; CHECK-O3-NEXT: Running pass: GVNPass
; CHECK-O3-NEXT: Running analysis: MemoryDependenceAnalysis
; CHECK-O3-NEXT: Running analysis: PhiValuesAnalysis
; CHECK-O1-NEXT: Running pass: MemCpyOptPass
; CHECK-O-NEXT: Running pass: SCCPPass
; CHECK-O-NEXT: Running pass: BDCEPass
; CHECK-O-NEXT: Running analysis: DemandedBitsAnalysis
; CHECK-O-NEXT: Running pass: InstCombinePass
; CHECK-O23SZ-NEXT: Running pass: JumpThreadingPass
; CHECK-O23SZ-NEXT: Running analysis: LazyValueAnalysis
; CHECK-O23SZ-NEXT: Running pass: CorrelatedValuePropagationPass
; CHECK-O23SZ-NEXT: Invalidating analysis: LazyValueAnalysis
; CHECK-O1-NEXT: Running pass: CoroElidePass
; CHECK-O-NEXT: Running pass: ADCEPass
; CHECK-O-NEXT: Running analysis: PostDominatorTreeAnalysis
; CHECK-O23SZ-NEXT: Running pass: MemCpyOptPass
; CHECK-O23SZ-NEXT: Running pass: DSEPass
; CHECK-O23SZ-NEXT: Running pass: LoopSimplifyPass
; CHECK-O23SZ-NEXT: Running pass: LCSSAPass
; CHECK-O23SZ-NEXT: Running pass: LICMPass on Loop at depth 1 containing: %loop
; CHECK-O23SZ-NEXT: Running pass: CoroElidePass
; CHECK-O-NEXT: Running pass: SimplifyCFGPass
; CHECK-O-NEXT: Running pass: InstCombinePass
; CHECK-O-NEXT: Running pass: CoroSplitPass
; CHECK-O-NEXT: Invalidating analysis: InlineAdvisorAnalysis
; CHECK-PRELINK-O-NEXT: Running pass: GlobalOptPass
; CHECK-POSTLINK-O-NEXT: Running pass: GlobalOptPass
; CHECK-POSTLINK-O-NEXT: Running pass: GlobalDCEPass
; CHECK-POSTLINK-O-NEXT: Running pass: EliminateAvailableExternallyPass
; CHECK-POSTLINK-O-NEXT: Running pass: ReversePostOrderFunctionAttrsPass
; CHECK-POSTLINK-O-NEXT: Running pass: RecomputeGlobalsAAPass
; CHECK-POSTLINK-O-NEXT: Running pass: Float2IntPass
; CHECK-POSTLINK-O-NEXT: Running pass: LowerConstantIntrinsicsPass
; CHECK-EXT: Running pass: {{.*}}::Bye
; CHECK-POSTLINK-O-NEXT: Running pass: LoopSimplifyPass
; CHECK-POSTLINK-O-NEXT: Running pass: LCSSAPass
; CHECK-POSTLINK-O-NEXT: Running pass: LoopRotatePass
; CHECK-POSTLINK-O-NEXT: Running pass: LoopDeletionPass
; CHECK-POSTLINK-O-NEXT: Running pass: LoopDistributePass
; CHECK-POSTLINK-O-NEXT: Running pass: InjectTLIMappings
; CHECK-POSTLINK-O-NEXT: Running pass: LoopVectorizePass
; CHECK-POSTLINK-O-NEXT: Running analysis: BlockFrequencyAnalysis
; CHECK-POSTLINK-O-NEXT: Running analysis: BranchProbabilityAnalysis
; CHECK-POSTLINK-O-NEXT: Running pass: LoopLoadEliminationPass
; CHECK-POSTLINK-O-NEXT: Running analysis: LoopAccessAnalysis
; CHECK-POSTLINK-O-NEXT: Running pass: InstCombinePass
; CHECK-POSTLINK-O-NEXT: Running pass: SimplifyCFGPass
; CHECK-POSTLINK-O2-NEXT: Running pass: SLPVectorizerPass
; CHECK-POSTLINK-O3-NEXT: Running pass: SLPVectorizerPass
; CHECK-POSTLINK-Os-NEXT: Running pass: SLPVectorizerPass
; CHECK-POSTLINK-O-NEXT: Running pass: VectorCombinePass
; CHECK-POSTLINK-O-NEXT: Running pass: InstCombinePass
; CHECK-POSTLINK-O-NEXT: Running pass: LoopUnrollPass
; CHECK-POSTLINK-O-NEXT: Running pass: WarnMissedTransformationsPass
; CHECK-POSTLINK-O-NEXT: Running pass: InstCombinePass
; CHECK-POSTLINK-O-NEXT: Running pass: RequireAnalysisPass<{{.*}}OptimizationRemarkEmitterAnalysis
; CHECK-POSTLINK-O-NEXT: Running pass: LoopSimplifyPass
; CHECK-POSTLINK-O-NEXT: Running pass: LCSSAPass
; CHECK-POSTLINK-O-NEXT: Running pass: LICMPass
; CHECK-POSTLINK-O-NEXT: Running pass: AlignmentFromAssumptionsPass
; CHECK-POSTLINK-O-NEXT: Running pass: LoopSinkPass
; CHECK-POSTLINK-O-NEXT: Running pass: InstSimplifyPass
; CHECK-POSTLINK-O-NEXT: Running pass: DivRemPairsPass
; CHECK-POSTLINK-O-NEXT: Running pass: SimplifyCFGPass
; CHECK-O-NEXT: Running pass: CoroCleanupPass
; CHECK-POSTLINK-O-NEXT: Running pass: CGProfilePass
; CHECK-POSTLINK-O-NEXT: Running pass: GlobalDCEPass
; CHECK-POSTLINK-O-NEXT: Running pass: ConstantMergePass
; CHECK-POSTLINK-O-NEXT: Running pass: RelLookupTableConverterPass
; CHECK-O-NEXT:          Running pass: AnnotationRemarksPass on foo
; CHECK-PRELINK-O-NEXT: Running pass: CanonicalizeAliasesPass
; CHECK-PRELINK-O-NEXT: Running pass: NameAnonGlobalPass
; CHECK-O-NEXT: Running pass: PrintModulePass

; Make sure we get the IR back out without changes when we print the module.
; CHECK-O-LABEL: define void @foo(i32 %n) local_unnamed_addr {
; CHECK-O-NEXT: entry:
; CHECK-O-NEXT:   br label %loop
; CHECK-O:      loop:
; CHECK-O-NEXT:   %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
; CHECK-O-NEXT:   %iv.next = add i32 %iv, 1
; CHECK-O-NEXT:   tail call void @bar()
; CHECK-O-NEXT:   %cmp = icmp eq i32 %iv, %n
; CHECK-O-NEXT:   br i1 %cmp, label %exit, label %loop
; CHECK-O:      exit:
; CHECK-O-NEXT:   ret void
; CHECK-O-NEXT: }
;

declare void @bar() local_unnamed_addr

define void @foo(i32 %n) local_unnamed_addr {
entry:
  br label %loop
loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.next = add i32 %iv, 1
  tail call void @bar()
  %cmp = icmp eq i32 %iv, %n
  br i1 %cmp, label %exit, label %loop
exit:
  ret void
}
