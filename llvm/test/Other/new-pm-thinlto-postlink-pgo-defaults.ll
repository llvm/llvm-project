; Validate ThinLTO post link pipeline when we have instrumentation PGO
;
; Postlink pipelines:
; RUN: opt -disable-verify -verify-cfg-preserved=0 -eagerly-invalidate-analyses=0 -debug-pass-manager \
; RUN:     -passes='thinlto<O1>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=CHECK-O,CHECK-O1,%llvmcheckext
; RUN: opt -disable-verify -verify-cfg-preserved=0 -eagerly-invalidate-analyses=0 -debug-pass-manager \
; RUN:     -passes='thinlto<O2>' -S  %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=CHECK-O,CHECK-O2,CHECK-O23SZ,%llvmcheckext
; RUN: opt -disable-verify -verify-cfg-preserved=0 -eagerly-invalidate-analyses=0 -debug-pass-manager -passes-ep-pipeline-start='no-op-module' \
; RUN:     -passes='thinlto<O3>' -S  %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=CHECK-O,CHECK-O3,CHECK-O23SZ,%llvmcheckext
; RUN: opt -disable-verify -verify-cfg-preserved=0 -eagerly-invalidate-analyses=0 -debug-pass-manager \
; RUN:     -passes='thinlto<Os>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=CHECK-O,CHECK-Os,CHECK-O23SZ,%llvmcheckext
; RUN: opt -disable-verify -verify-cfg-preserved=0 -eagerly-invalidate-analyses=0 -debug-pass-manager \
; RUN:     -passes='thinlto<Oz>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=CHECK-O,CHECK-O23SZ,%llvmcheckext
; RUN: opt -disable-verify -verify-cfg-preserved=0 -eagerly-invalidate-analyses=0 -debug-pass-manager -debug-info-for-profiling \
; RUN:     -passes='thinlto<O2>' -S  %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=CHECK-O,CHECK-O2,CHECK-O23SZ,%llvmcheckext

; Suppress FileCheck --allow-unused-prefixes=false diagnostics.
; CHECK-NOEXT: {{^}}

; CHECK-O: Running pass: Annotation2Metadata
; CHECK-O-NEXT: Running pass: ForceFunctionAttrsPass
; CHECK-EP-PIPELINE-START-NEXT: Running pass: NoOpModulePass
; CHECK-O-NEXT: Running pass: PGOIndirectCallPromotion
; CHECK-O-NEXT: Running analysis: ProfileSummaryAnalysis
; CHECK-O-NEXT: Running analysis: InnerAnalysisManagerProxy
; CHECK-O-NEXT: Running analysis: OptimizationRemarkEmitterAnalysis
; CHECK-O-NEXT: Running pass: InferFunctionAttrsPass
; CHECK-O-NEXT: Running analysis: TargetLibraryAnalysis
; CHECK-O-NEXT: Running pass: CoroEarlyPass
; CHECK-O-NEXT: Running pass: LowerExpectIntrinsicPass
; CHECK-O-NEXT: Running pass: SimplifyCFGPass
; CHECK-O-NEXT: Running analysis: TargetIRAnalysis
; CHECK-O-NEXT: Running analysis: AssumptionAnalysis
; CHECK-O-NEXT: Running pass: SROAPass
; CHECK-O-NEXT: Running analysis: DominatorTreeAnalysis
; CHECK-O-NEXT: Running pass: EarlyCSEPass
; CHECK-O-NEXT: Running analysis: TargetLibraryAnalysis
; CHECK-O3-NEXT: Running pass: CallSiteSplittingPass
; CHECK-O-NEXT: Running pass: OpenMPOptPass
; CHECK-O-NEXT: Running pass: LowerTypeTestsPass
; CHECK-O-NEXT: Running pass: IPSCCPPass
; CHECK-O-NEXT: Running pass: CalledValuePropagationPass
; CHECK-O-NEXT: Running pass: GlobalOptPass
; CHECK-O-NEXT: Running pass: PromotePass
; CHECK-O-NEXT: Running pass: InstCombinePass
; CHECK-O-NEXT: Running analysis: AAManager
; CHECK-O-NEXT: Running analysis: BasicAA
; CHECK-O-NEXT: Running analysis: ScopedNoAliasAA
; CHECK-O-NEXT: Running analysis: TypeBasedAA
; CHECK-O-NEXT: Running analysis: OuterAnalysisManagerProxy
; CHECK-O-NEXT: Running analysis: BlockFrequencyAnalysis on foo
; These next two can appear in any order since they are accessed as parameters
; on the same call to BlockFrequencyInfo::calculate.
; CHECK-O-DAG: Running analysis: LoopAnalysis on foo
; CHECK-O-DAG: Running analysis: BranchProbabilityAnalysis on foo
; CHECK-O-NEXT: Running analysis: PostDominatorTreeAnalysis on foo
; CHECK-O-NEXT: Running pass: SimplifyCFGPass
; CHECK-O-NEXT: Running pass: ModuleInlinerWrapperPass
; CHECK-O-NEXT: Running analysis: InlineAdvisorAnalysis
; CHECK-O-NEXT: Running pass: RequireAnalysisPass<{{.*}}GlobalsAA
; CHECK-O-NEXT: Running analysis: GlobalsAA
; CHECK-O-NEXT: Running analysis: CallGraphAnalysis
; CHECK-O-NEXT: Running pass: InvalidateAnalysisPass<{{.*}}AAManager
; CHECK-O-NEXT: Invalidating analysis: AAManager
; CHECK-O-NEXT: Running pass: RequireAnalysisPass<{{.*}}ProfileSummaryAnalysis
; CHECK-O-NEXT: Running analysis: InnerAnalysisManagerProxy
; CHECK-O-NEXT: Running analysis: LazyCallGraphAnalysis
; CHECK-O-NEXT: Running analysis: FunctionAnalysisManagerCGSCCProxy
; CHECK-O-NEXT: Running analysis: OuterAnalysisManagerProxy<{{.*}}LazyCallGraph::SCC{{.*}}>
; CHECK-O-NEXT: Running pass: DevirtSCCRepeatedPass
; CHECK-O-NEXT: Running pass: InlinerPass
; CHECK-O-NEXT: Running pass: InlinerPass
; CHECK-O-NEXT: Running pass: PostOrderFunctionAttrsPass
; CHECK-O-NEXT: Running analysis: AAManager
; CHECK-O3-NEXT: Running pass: ArgumentPromotionPass
; CHECK-O2-NEXT: Running pass: OpenMPOptCGSCCPass
; CHECK-O3-NEXT: Running pass: OpenMPOptCGSCCPass
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
; CHECK-O-NEXT: Running pass: LCSSAPass
; CHECK-O-NEXT: Running analysis: ScalarEvolutionAnalysis
; CHECK-O-NEXT: Running analysis: InnerAnalysisManagerProxy
; CHECK-O-NEXT: Running pass: LoopInstSimplifyPass
; CHECK-O-NEXT: Running pass: LoopSimplifyCFGPass
; CHECK-O-NEXT: Running pass: LICM
; CHECK-O-NEXT: Running pass: LoopRotatePass
; CHECK-O-NEXT: Running pass: LICM
; CHECK-O-NEXT: Running pass: SimpleLoopUnswitchPass
; CHECK-O-NEXT: Running analysis: OuterAnalysisManagerProxy
; CHECK-O-NEXT: Running pass: LoopFlattenPass
; CHECK-O-NEXT: Running pass: SimplifyCFGPass
; CHECK-O-NEXT: Running pass: InstCombinePass
; CHECK-O-NEXT: Running pass: LoopSimplifyPass
; CHECK-O-NEXT: Running pass: LCSSAPass
; CHECK-O-NEXT: Running pass: LoopIdiomRecognizePass
; CHECK-O-NEXT: Running pass: IndVarSimplifyPass
; CHECK-O-NEXT: Running pass: LoopDeletionPass
; CHECK-O-NEXT: Running pass: LoopFullUnrollPass
; CHECK-O-NEXT: Running pass: SROAPass on foo
; CHECK-O23SZ-NEXT: Running pass: VectorCombinePass
; CHECK-O23SZ-NEXT: Running pass: MergedLoadStoreMotionPass
; CHECK-O23SZ-NEXT: Running pass: GVNPass
; CHECK-O23SZ-NEXT: Running analysis: MemoryDependenceAnalysis
; CHECK-O23SZ-NEXT: Running analysis: PhiValuesAnalysis
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
; CHECK-O23SZ-NEXT: Running pass: MemCpyOptPass
; CHECK-O23SZ-NEXT: Running pass: DSEPass
; CHECK-O23SZ-NEXT: Running pass: LoopSimplifyPass
; CHECK-O23SZ-NEXT: Running pass: LCSSAPass
; CHECK-O23SZ-NEXT: Running pass: LICMPass
; CHECK-O23SZ-NEXT: Running pass: CoroElidePass
; CHECK-O-NEXT: Running pass: SimplifyCFGPass
; CHECK-O-NEXT: Running pass: InstCombinePass
; CHECK-O-NEXT: Running analysis: ShouldNotRunFunctionPassesAnalysis
; CHECK-O-NEXT: Running pass: CoroSplitPass
; CHECK-O-NEXT: Running pass: InvalidateAnalysisPass<{{.*}}ShouldNotRunFunctionPassesAnalysis
; CHECK-O-NEXT: Invalidating analysis: ShouldNotRunFunctionPassesAnalysis
; CHECK-O-NEXT: Invalidating analysis: InlineAdvisorAnalysis
; CHECK-O-NEXT: Running pass: DeadArgumentEliminationPass
; CHECK-O-NEXT: Running pass: CoroCleanupPass
; CHECK-O-NEXT: Running pass: GlobalOptPass
; CHECK-O-NEXT: Running pass: GlobalDCEPass
; CHECK-O-NEXT: Running pass: EliminateAvailableExternallyPass
; CHECK-O-NEXT: Running pass: ReversePostOrderFunctionAttrsPass
; CHECK-O-NEXT: Running pass: RecomputeGlobalsAAPass
; CHECK-O-NEXT: Running pass: Float2IntPass
; CHECK-O-NEXT: Running pass: LowerConstantIntrinsicsPass
; CHECK-EXT: Running pass: {{.*}}::Bye
; CHECK-O-NEXT: Running pass: LoopSimplifyPass on foo
; CHECK-O-NEXT: Running pass: LCSSAPass on foo
; CHECK-O-NEXT: Running pass: LoopRotatePass
; CHECK-O-NEXT: Running pass: LoopDeletionPass
; CHECK-O-NEXT: Running pass: LoopDistributePass
; CHECK-O-NEXT: Running analysis: LoopAccessAnalysis on foo
; CHECK-O-NEXT: Running pass: InjectTLIMappings
; CHECK-O-NEXT: Running pass: LoopVectorizePass
; CHECK-O-NEXT: Running pass: LoopLoadEliminationPass
; CHECK-O-NEXT: Running pass: InstCombinePass
; CHECK-O-NEXT: Running pass: SimplifyCFGPass
; CHECK-O2-NEXT: Running pass: SLPVectorizerPass
; CHECK-O3-NEXT: Running pass: SLPVectorizerPass
; CHECK-Os-NEXT: Running pass: SLPVectorizerPass
; CHECK-O-NEXT: Running pass: VectorCombinePass
; CHECK-O-NEXT: Running pass: InstCombinePass
; CHECK-O-NEXT: Running pass: LoopUnrollPass
; CHECK-O-NEXT: Running pass: WarnMissedTransformationsPass
; CHECK-O-NEXT: Running pass: SROAPass
; CHECK-O-NEXT: Running pass: InstCombinePass
; CHECK-O-NEXT: Running pass: RequireAnalysisPass<{{.*}}OptimizationRemarkEmitterAnalysis
; CHECK-O-NEXT: Running pass: LoopSimplifyPass
; CHECK-O-NEXT: Running pass: LCSSAPass
; CHECK-O-NEXT: Running pass: LICMPass
; CHECK-O-NEXT: Running pass: AlignmentFromAssumptionsPass
; CHECK-O-NEXT: Running pass: LoopSinkPass
; CHECK-O-NEXT: Running pass: InstSimplifyPass
; CHECK-O-NEXT: Running pass: DivRemPairsPass
; CHECK-O-NEXT: Running pass: TailCallElimPass
; CHECK-O-NEXT: Running pass: SimplifyCFGPass
; CHECK-O-NEXT: Running pass: GlobalDCEPass
; CHECK-O-NEXT: Running pass: ConstantMergePass
; CHECK-O-NEXT: Running pass: CGProfilePass
; CHECK-O-NEXT: Running pass: RelLookupTableConverterPass
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
; Ignore a bunch of intervening metadata containing profile data.
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

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 0}
!4 = !{!"MaxCount", i64 0}
!5 = !{!"MaxInternalCount", i64 0}
!6 = !{!"MaxFunctionCount", i64 0}
!7 = !{!"NumCounts", i64 0}
!8 = !{!"NumFunctions", i64 0}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26}
!11 = !{i32 10000, i64 0, i32 0}
!12 = !{i32 100000, i64 0, i32 0}
!13 = !{i32 200000, i64 0, i32 0}
!14 = !{i32 300000, i64 0, i32 0}
!15 = !{i32 400000, i64 0, i32 0}
!16 = !{i32 500000, i64 0, i32 0}
!17 = !{i32 600000, i64 0, i32 0}
!18 = !{i32 700000, i64 0, i32 0}
!19 = !{i32 800000, i64 0, i32 0}
!20 = !{i32 900000, i64 0, i32 0}
!21 = !{i32 950000, i64 0, i32 0}
!22 = !{i32 990000, i64 0, i32 0}
!23 = !{i32 999000, i64 0, i32 0}
!24 = !{i32 999900, i64 0, i32 0}
!25 = !{i32 999990, i64 0, i32 0}
!26 = !{i32 999999, i64 0, i32 0}
