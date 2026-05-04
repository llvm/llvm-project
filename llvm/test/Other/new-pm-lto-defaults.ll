; Basic test for the new LTO pipeline.

; RUN: opt -disable-verify -verify-analysis-invalidation=0 -eagerly-invalidate-analyses=0 -debug-pass-manager \
; RUN:     -passes='lto<O1>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=CHECK-O,CHECK-O1
; RUN: opt -disable-verify -verify-analysis-invalidation=0 -eagerly-invalidate-analyses=0 -debug-pass-manager \
; RUN:     -passes='lto<O1>' -S %s -passes-ep-full-link-time-optimization-early=no-op-module \
; RUN:     -passes-ep-full-link-time-optimization-last=no-op-module 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=CHECK-O,CHECK-O1,CHECK-EP
; RUN: opt -disable-verify -verify-analysis-invalidation=0 -eagerly-invalidate-analyses=0 -debug-pass-manager \
; RUN:     -passes='lto<O2>' -S  %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=CHECK-O,CHECK-O23
; RUN: opt -disable-verify -verify-analysis-invalidation=0 -eagerly-invalidate-analyses=0 -debug-pass-manager \
; RUN:     -passes='lto<O2>' -S %s -passes-ep-full-link-time-optimization-early=no-op-module \
; RUN:     -passes-ep-full-link-time-optimization-last=no-op-module 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=CHECK-O,CHECK-O23,CHECK-EP
; RUN: opt -disable-verify -verify-analysis-invalidation=0 -eagerly-invalidate-analyses=0 -debug-pass-manager \
; RUN:     -passes='lto<O3>' -S  %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=CHECK-O,CHECK-O23
; RUN: opt -disable-verify -verify-analysis-invalidation=0 -eagerly-invalidate-analyses=0 -debug-pass-manager \
; RUN:     -passes='lto<O3>' -S  %s -passes-ep-vectorizer-start='no-op-function' 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=CHECK-O,CHECK-O23,CHECK-EP-VECTORIZER-START
; RUN: opt -disable-verify -verify-analysis-invalidation=0 -eagerly-invalidate-analyses=0 -debug-pass-manager \
; RUN:     -passes='lto<O3>' -S  %s -passes-ep-vectorizer-end='no-op-function' 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=CHECK-O,CHECK-O23,CHECK-EP-VECTORIZER-END
; RUN: opt -disable-verify -verify-analysis-invalidation=0 -eagerly-invalidate-analyses=0 -debug-pass-manager \
; RUN:     -passes='lto<O3>' -S  %s -passes-ep-peephole='no-op-function' 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=CHECK-O,CHECK-O23,CHECK-EP-PEEPHOLE

; CHECK-EP: Running pass: NoOpModulePass
; CHECK-O: Running pass: CrossDSOCFIPass
; CHECK-O-NEXT: Running pass: OpenMPOptPass
; CHECK-O-NEXT: Running pass: GlobalDCEPass
; CHECK-O-NEXT: Running pass: InferFunctionAttrsPass
; CHECK-O-NEXT: Running analysis: InnerAnalysisManagerProxy<{{.*}}Module
; CHECK-O-NEXT: Running analysis: TargetLibraryAnalysis
; CHECK-O23-NEXT: Running pass: CallSiteSplittingPass on foo
; CHECK-O23-NEXT: Running analysis: TargetLibraryAnalysis on foo
; CHECK-O23-NEXT: Running analysis: TargetIRAnalysis on foo
; CHECK-O23-NEXT: Running analysis: DominatorTreeAnalysis on foo
; CHECK-O23-NEXT: PGOIndirectCallPromotion
; CHECK-O23-NEXT: Running analysis: ProfileSummaryAnalysis
; CHECK-O23-NEXT: Running analysis: OptimizationRemarkEmitterAnalysis
; CHECK-O23-NEXT: Running analysis: InnerAnalysisManagerProxy<{{.*}}SCC
; CHECK-O23-NEXT: Running analysis: LazyCallGraphAnalysis
; CHECK-O23-NEXT: Running analysis: FunctionAnalysisManagerCGSCCProxy
; CHECK-O23-NEXT: Running analysis: OuterAnalysisManagerProxy<{{.*}}LazyCallGraph{{.*}}>
; CHECK-O23-NEXT: Running pass: PostOrderFunctionAttrsPass
; CHECK-O23-NEXT: Running analysis: AAManager
; CHECK-O23-NEXT: Running analysis: BasicAA
; CHECK-O23-NEXT: Running analysis: AssumptionAnalysis on foo
; CHECK-O23-NEXT: Running analysis: ScopedNoAliasAA
; CHECK-O23-NEXT: Running analysis: TypeBasedAA
; CHECK-O23-NEXT: Running analysis: OuterAnalysisManagerProxy
; CHECK-O23-NEXT: Running pass: ArgumentPromotionPass
; CHECK-O23-NEXT: Running pass: SROAPass
; CHECK-O23-NEXT: Running pass: IPSCCPPass
; CHECK-O23-NEXT: Running pass: CalledValuePropagationPass
; CHECK-O-NEXT: Running pass: ReversePostOrderFunctionAttrsPass
; CHECK-O1-NEXT: Running analysis: LazyCallGraphAnalysis
; CHECK-O1-NEXT: Running analysis: TargetLibraryAnalysis
; CHECK-O-NEXT: Running pass: GlobalSplitPass
; CHECK-O-NEXT: Running pass: WholeProgramDevirtPass
; CHECK-O-NEXT: Running pass: NoRecurseLTOInferencePass
; CHECK-O23-NEXT: Running pass: CoroEarlyPass
; CHECK-O1-NEXT: Running pass: LowerTypeTestsPass
; CHECK-O23-NEXT: Running pass: GlobalOptPass
; CHECK-O23-NEXT: Running pass: PromotePass
; CHECK-O23-NEXT: Running pass: ConstantMergePass
; CHECK-O23-NEXT: Running pass: DeadArgumentEliminationPass
; CHECK-O23-NEXT: Running pass: InstCombinePass
; CHECK-O23-NEXT: Running analysis: LastRunTrackingAnalysis
; CHECK-O23-NEXT: Running pass: AggressiveInstCombinePass
; CHECK-EP-PEEPHOLE-NEXT: Running pass: NoOpFunctionPass
; CHECK-O23-NEXT: Running pass: ExpandVariadicsPass
; CHECK-O23-NEXT: Running pass: ModuleInlinerWrapperPass
; CHECK-O23-NEXT: Running analysis: InlineAdvisorAnalysis
; CHECK-O23-NEXT: Running pass: InlinerPass
; CHECK-O23-NEXT: Running pass: InlinerPass
; CHECK-O23-NEXT: Invalidating analysis: InlineAdvisorAnalysis
; CHECK-O23-NEXT: Running pass: GlobalOptPass
; CHECK-O23-NEXT: Running pass: OpenMPOptPass
; CHECK-O23-NEXT: Running pass: GlobalDCEPass
; CHECK-O23-NEXT: Running pass: ArgumentPromotionPass on (foo)
; CHECK-O23-NEXT: CoroSplitPass on (foo)
; CHECK-O23-NEXT: CoroAnnotationElidePass on (foo)
; CHECK-O23-NEXT: Running pass: InstCombinePass
; CHECK-EP-PEEPHOLE-NEXT: Running pass: NoOpFunctionPass
; CHECK-O23-NEXT: Running pass: ConstraintEliminationPass
; CHECK-O23-NEXT: Running analysis: LoopAnalysis on foo
; CHECK-O23-NEXT: Running analysis: ScalarEvolutionAnalysis on foo
; CHECK-O23-NEXT: Running pass: JumpThreadingPass
; CHECK-O23-NEXT: Running analysis: LazyValueAnalysis
; CHECK-O23-NEXT: Running pass: SROAPass on foo
; CHECK-O23-NEXT: Running pass: TailCallElimPass on foo
; CHECK-O23-NEXT: Running pass: PostOrderFunctionAttrsPass on (foo)
; CHECK-O23-NEXT: Running pass: RequireAnalysisPass<{{.*}}GlobalsAA
; CHECK-O23-NEXT: Running analysis: GlobalsAA on [module]
; CHECK-O23-NEXT: Running analysis: CallGraphAnalysis on [module]
; CHECK-O23-NEXT: Running pass: InvalidateAnalysisPass<{{.*}}AAManager
; CHECK-O23-NEXT: Invalidating analysis: AAManager on foo
; CHECK-O23-NEXT: Running pass: OpenMPOptCGSCCPass on (foo)
; CHECK-O23-NEXT: Running pass: LoopSimplifyPass on foo
; CHECK-O23-NEXT: Running pass: LCSSAPass on foo
; CHECK-O23-NEXT: Running analysis: MemorySSAAnalysis on foo
; CHECK-O23-NEXT: Running analysis: AAManager on foo
; CHECK-O23-NEXT: Running analysis: InnerAnalysisManagerProxy
; CHECK-O23-NEXT: Running pass: LICMPass on loop
; CHECK-O23-NEXT: Running pass: GVNPass on foo
; CHECK-O23-NEXT: Running analysis: MemoryDependenceAnalysis on foo
; CHECK-O23-NEXT: Running pass: MemCpyOptPass on foo
; CHECK-O23-NEXT: Running analysis: PostDominatorTreeAnalysis on foo
; CHECK-O23-NEXT: Running pass: DSEPass on foo
; CHECK-O23-NEXT: Running analysis: CycleAnalysis
; CHECK-O23-NEXT: Running pass: MoveAutoInitPass on foo
; CHECK-O23-NEXT: Running pass: MergedLoadStoreMotionPass on foo
; CHECK-EP-VECTORIZER-START-NEXT: Running pass: NoOpFunctionPass on foo
; CHECK-O23-NEXT: Running pass: LoopSimplifyPass on foo
; CHECK-O23-NEXT: Running pass: LCSSAPass on foo
; CHECK-O23-NEXT: Running pass: IndVarSimplifyPass on loop
; CHECK-O23-NEXT: Running pass: LoopDeletionPass on loop
; CHECK-O23-NEXT: Running pass: LoopFullUnrollPass on loop
; CHECK-O23-NEXT: Running pass: LoopDistributePass on foo
; CHECK-O23-NEXT: Running analysis: LoopAccessAnalysis on foo
; CHECK-O23-NEXT: Running pass: LoopVectorizePass on foo
; CHECK-O23-NEXT: Running analysis: DemandedBitsAnalysis on foo
; CHECK-O23-NEXT: Running pass: DropUnnecessaryAssumesPass on foo
; CHECK-O23-NEXT: Running pass: InferAlignmentPass on foo
; CHECK-O23-NEXT: Running pass: LoopUnrollPass on foo
; CHECK-O23-NEXT: WarnMissedTransformationsPass on foo
; CHECK-O23-NEXT: Running pass: SROAPass on foo
; CHECK-O23-NEXT: Running pass: InstCombinePass on foo
; CHECK-O23-NEXT: Running pass: SimplifyCFGPass on foo
; CHECK-O23-NEXT: Running pass: SCCPPass on foo
; CHECK-O23-NEXT: Running pass: InstCombinePass on foo
; CHECK-O23-NEXT: Running pass: BDCEPass on foo
; CHECK-O23-NEXT: Running pass: SLPVectorizerPass on foo
; CHECK-O23-NEXT: Running pass: VectorCombinePass on foo
; CHECK-O23-NEXT: Running pass: InferAlignmentPass on foo
; CHECK-O23-NEXT: Running pass: InstCombinePass on foo
; CHECK-O23-NEXT: Running pass: LoopSimplifyPass
; CHECK-O23-NEXT: Running pass: LCSSAPass
; CHECK-O23-NEXT: Running pass: LICMPass
; CHECK-O23-NEXT: Running pass: AlignmentFromAssumptionsPass on foo
; CHECK-EP-VECTORIZER-END-NEXT: Running pass: NoOpFunctionPass on foo
; CHECK-EP-PEEPHOLE-NEXT: Running pass: NoOpFunctionPass on foo
; CHECK-O23-NEXT: Running pass: JumpThreadingPass on foo
; CHECK-O23-NEXT: Running pass: LowerTypeTestsPass
; CHECK-O-NEXT: Running pass: DropTypeTestsPass
; CHECK-O23-NEXT: Running pass: LoopSink
; CHECK-O23-NEXT: Running pass: DivRemPairs
; CHECK-O23-NEXT: Running pass: SimplifyCFGPass
; CHECK-O23-NEXT: Running pass: EliminateAvailableExternallyPass
; CHECK-O23-NEXT: Running pass: GlobalDCEPass
; CHECK-O23-NEXT: Running pass: RelLookupTableConverterPass
; CHECK-O23-NEXT: Running pass: CGProfilePass
; CHECK-O1-NEXT: Running pass: CoroConditionalWrapper
; CHECK-O23-NEXT: Running pass: CoroCleanupPass
; CHECK-O-NEXT: Running pass: AllocTokenPass
; CHECK-EP-NEXT: Running pass: NoOpModulePass
; CHECK-O-NEXT: Running pass: AnnotationRemarksPass on foo
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
