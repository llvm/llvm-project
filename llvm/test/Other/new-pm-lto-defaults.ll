; Basic test for the new LTO pipeline.
; For now the only difference is between -O1 and everything else, so
; -O2, -O3, -Os, -Oz are the same.

; RUN: opt -disable-verify -verify-cfg-preserved=0 -eagerly-invalidate-analyses=0 -debug-pass-manager \
; RUN:     -passes='lto<O1>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-O --check-prefix=CHECK-O1
; RUN: opt -disable-verify -verify-cfg-preserved=0 -eagerly-invalidate-analyses=0 -debug-pass-manager \
; RUN:     -passes='lto<O1>' -S %s -passes-ep-full-link-time-optimization-early=no-op-module \
; RUN:     -passes-ep-full-link-time-optimization-last=no-op-module 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-O --check-prefix=CHECK-O1 --check-prefix=CHECK-EP
; RUN: opt -disable-verify -verify-cfg-preserved=0 -eagerly-invalidate-analyses=0 -debug-pass-manager \
; RUN:     -passes='lto<O2>' -S  %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-O --check-prefix=CHECK-O23SZ \
; RUN:     --check-prefix=CHECK-O2
; RUN: opt -disable-verify -verify-cfg-preserved=0 -eagerly-invalidate-analyses=0 -debug-pass-manager \
; RUN:     -passes='lto<O2>' -S %s -passes-ep-full-link-time-optimization-early=no-op-module \
; RUN:     -passes-ep-full-link-time-optimization-last=no-op-module 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-O --check-prefix=CHECK-O23SZ \
; RUN:     --check-prefix=CHECK-O2 --check-prefix=CHECK-EP
; RUN: opt -disable-verify -verify-cfg-preserved=0 -eagerly-invalidate-analyses=0 -debug-pass-manager \
; RUN:     -passes='lto<O3>' -S  %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-O --check-prefix=CHECK-O23SZ \
; RUN:     --check-prefix=CHECK-O3
; RUN: opt -disable-verify -verify-cfg-preserved=0 -eagerly-invalidate-analyses=0 -debug-pass-manager \
; RUN:     -passes='lto<Os>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-O --check-prefix=CHECK-O23SZ \
; RUN:     --check-prefix=CHECK-OS
; RUN: opt -disable-verify -verify-cfg-preserved=0 -eagerly-invalidate-analyses=0 -debug-pass-manager \
; RUN:     -passes='lto<Oz>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-O --check-prefix=CHECK-O23SZ
; RUN: opt -disable-verify -verify-cfg-preserved=0 -eagerly-invalidate-analyses=0 -debug-pass-manager \
; RUN:     -passes='lto<O3>' -S  %s -passes-ep-peephole='no-op-function' 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-O --check-prefix=CHECK-O23SZ \
; RUN:     --check-prefix=CHECK-O3 --check-prefix=CHECK-EP-Peephole

; CHECK-O: Running pass: Annotation2Metadata
; CHECK-EP-NEXT: Running pass: NoOpModulePass
; CHECK-O-NEXT: Running pass: CrossDSOCFIPass
; CHECK-O-NEXT: Running pass: OpenMPOptPass
; CHECK-O-NEXT: Running pass: GlobalDCEPass
; CHECK-O-NEXT: Running pass: ForceFunctionAttrsPass
; CHECK-O-NEXT: Running pass: InferFunctionAttrsPass
; CHECK-O-NEXT: Running analysis: InnerAnalysisManagerProxy<{{.*}}Module
; CHECK-O-NEXT: Running analysis: TargetLibraryAnalysis
; CHECK-O23SZ-NEXT: Running pass: CallSiteSplittingPass on foo
; CHECK-O23SZ-NEXT: Running analysis: TargetLibraryAnalysis on foo
; CHECK-O23SZ-NEXT: Running analysis: TargetIRAnalysis on foo
; CHECK-O23SZ-NEXT: Running analysis: DominatorTreeAnalysis on foo
; CHECK-O23SZ-NEXT: PGOIndirectCallPromotion
; CHECK-O23SZ-NEXT: Running analysis: ProfileSummaryAnalysis
; CHECK-O23SZ-NEXT: Running analysis: OptimizationRemarkEmitterAnalysis
; CHECK-O23SZ-NEXT: Running pass: IPSCCPPass
; CHECK-O23SZ-NEXT: Running analysis: AssumptionAnalysis on foo
; CHECK-O23SZ-NEXT: Running pass: CalledValuePropagationPass
; CHECK-O-NEXT: Running analysis: InnerAnalysisManagerProxy<{{.*}}SCC
; CHECK-O-NEXT: Running analysis: LazyCallGraphAnalysis
; CHECK-O1-NEXT: Running analysis: TargetLibraryAnalysis
; CHECK-O-NEXT: Running analysis: FunctionAnalysisManagerCGSCCProxy
; CHECK-O-NEXT: Running analysis: OuterAnalysisManagerProxy<{{.*}}LazyCallGraph{{.*}}>
; CHECK-O-NEXT: Running pass: PostOrderFunctionAttrsPass
; CHECK-O-NEXT: Running analysis: AAManager
; CHECK-O-NEXT: Running analysis: BasicAA
; CHECK-O1-NEXT: Running analysis: AssumptionAnalysis on foo
; CHECK-O1-NEXT: Running analysis: TargetIRAnalysis
; CHECK-O1-NEXT: Running analysis: DominatorTreeAnalysis
; CHECK-O-NEXT: Running analysis: ScopedNoAliasAA
; CHECK-O-NEXT: Running analysis: TypeBasedAA
; CHECK-O-NEXT: Running analysis: OuterAnalysisManagerProxy
; CHECK-O-NEXT: Running pass: ReversePostOrderFunctionAttrsPass
; CHECK-O-NEXT: Running analysis: CallGraphAnalysis
; CHECK-O-NEXT: Running pass: GlobalSplitPass
; CHECK-O-NEXT: Running pass: WholeProgramDevirtPass
; CHECK-O1-NEXT: Running pass: LowerTypeTestsPass
; CHECK-O23SZ-NEXT: Running pass: GlobalOptPass
; CHECK-O23SZ-NEXT: Running pass: PromotePass
; CHECK-O23SZ-NEXT: Running pass: ConstantMergePass
; CHECK-O23SZ-NEXT: Running pass: InstCombinePass
; CHECK-O3-NEXT: Running pass: AggressiveInstCombinePass
; CHECK-EP-Peephole-NEXT: Running pass: NoOpFunctionPass
; CHECK-O23SZ-NEXT: Running pass: ModuleInlinerWrapperPass
; CHECK-O23SZ-NEXT: Running analysis: InlineAdvisorAnalysis
; CHECK-O23SZ-NEXT: Running pass: InlinerPass
; CHECK-O23SZ-NEXT: Running pass: InlinerPass
; CHECK-O23SZ-NEXT: Invalidating analysis: InlineAdvisorAnalysis
; CHECK-O23SZ-NEXT: Running pass: GlobalOptPass
; CHECK-O23SZ-NEXT: Running pass: GlobalDCEPass
; CHECK-O23SZ-NEXT: Running pass: ArgumentPromotionPass
; CHECK-O23SZ-NEXT: Running pass: DeadArgumentEliminationPass
; CHECK-O23SZ-NEXT: Running pass: InstCombinePass
; CHECK-EP-Peephole-NEXT: Running pass: NoOpFunctionPass
; CHECK-O23SZ-NEXT: Running pass: JumpThreadingPass
; CHECK-O23SZ-NEXT: Running analysis: LazyValueAnalysis
; CHECK-O23SZ-NEXT: Running pass: SROAPass on foo
; CHECK-O23SZ-NEXT: Running pass: TailCallElimPass on foo
; CHECK-O23SZ-NEXT: Running pass: PostOrderFunctionAttrsPass on (foo)
; CHECK-O23SZ-NEXT: Running pass: RequireAnalysisPass<{{.*}}GlobalsAA
; CHECK-O23SZ-NEXT: Running analysis: GlobalsAA on [module]
; CHECK-O23SZ-NEXT: Running pass: InvalidateAnalysisPass<{{.*}}AAManager
; CHECK-O23SZ-NEXT: Invalidating analysis: AAManager on foo
; CHECK-O23SZ-NEXT: Running pass: OpenMPOptCGSCCPass on (foo)
; CHECK-O23SZ-NEXT: Running pass: LoopSimplifyPass on foo
; CHECK-O23SZ-NEXT: Running analysis: LoopAnalysis on foo
; CHECK-O23SZ-NEXT: Running pass: LCSSAPass on foo
; CHECK-O23SZ-NEXT: Running analysis: MemorySSAAnalysis on foo
; CHECK-O23SZ-NEXT: Running analysis: AAManager on foo
; CHECK-O23SZ-NEXT: Running analysis: ScalarEvolutionAnalysis on foo
; CHECK-O23SZ-NEXT: Running analysis: InnerAnalysisManagerProxy
; CHECK-O23SZ-NEXT: Running pass: LICMPass on loop
; CHECK-O23SZ-NEXT: Running pass: GVNPass on foo
; CHECK-O23SZ-NEXT: Running analysis: MemoryDependenceAnalysis on foo
; CHECK-O23SZ-NEXT: Running analysis: PhiValuesAnalysis on foo
; CHECK-O23SZ-NEXT: Running pass: MemCpyOptPass on foo
; CHECK-O23SZ-NEXT: Running pass: DSEPass on foo
; CHECK-O23SZ-NEXT: Running analysis: PostDominatorTreeAnalysis on foo
; CHECK-O23SZ-NEXT: Running pass: MergedLoadStoreMotionPass on foo
; CHECK-O23SZ-NEXT: Running pass: LoopSimplifyPass on foo
; CHECK-O23SZ-NEXT: Running pass: LCSSAPass on foo
; CHECK-O23SZ-NEXT: Running pass: LoopFlattenPass on loop
; CHECK-O23SZ-NEXT: Running pass: IndVarSimplifyPass on loop
; CHECK-O23SZ-NEXT: Running pass: LoopDeletionPass on loop
; CHECK-O23SZ-NEXT: Running pass: LoopFullUnrollPass on loop
; CHECK-O23SZ-NEXT: Running pass: LoopDistributePass on foo
; CHECK-O23SZ-NEXT: Running analysis: LoopAccessAnalysis on foo
; CHECK-O23SZ-NEXT: Running pass: LoopVectorizePass on foo
; CHECK-O23SZ-NEXT: Running analysis: BlockFrequencyAnalysis on foo
; CHECK-O23SZ-NEXT: Running analysis: BranchProbabilityAnalysis on foo
; CHECK-O23SZ-NEXT: Running analysis: DemandedBitsAnalysis on foo
; CHECK-O23SZ-NEXT: Running pass: LoopUnrollPass on foo
; CHECK-O23SZ-NEXT: WarnMissedTransformationsPass on foo
; CHECK-O23SZ-NEXT: Running pass: SROAPass on foo
; CHECK-O23SZ-NEXT: Running pass: InstCombinePass on foo
; CHECK-O23SZ-NEXT: Running pass: SimplifyCFGPass on foo
; CHECK-O23SZ-NEXT: Running pass: SCCPPass on foo
; CHECK-O23SZ-NEXT: Running pass: InstCombinePass on foo
; CHECK-O23SZ-NEXT: Running pass: BDCEPass on foo
; CHECK-O2-NEXT: Running pass: SLPVectorizerPass on foo
; CHECK-O3-NEXT: Running pass: SLPVectorizerPass on foo
; CHECK-OS-NEXT: Running pass: SLPVectorizerPass on foo
; CHECK-O23SZ-NEXT: Running pass: VectorCombinePass on foo
; CHECK-O23SZ-NEXT: Running pass: AlignmentFromAssumptionsPass on foo
; CHECK-O23SZ-NEXT: Running pass: InstCombinePass on foo
; CHECK-EP-Peephole-NEXT: Running pass: NoOpFunctionPass on foo
; CHECK-O23SZ-NEXT: Running pass: JumpThreadingPass on foo
; CHECK-O23SZ-NEXT: Running pass: LowerTypeTestsPass
; CHECK-O-NEXT: Running pass: LowerTypeTestsPass
; CHECK-O23SZ-NEXT: Running pass: SimplifyCFGPass
; CHECK-O23SZ-NEXT: Running pass: EliminateAvailableExternallyPass
; CHECK-O23SZ-NEXT: Running pass: GlobalDCEPass
; CHECK-O23SZ-NEXT: Running pass: CGProfilePass
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
