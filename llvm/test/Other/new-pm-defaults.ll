; The IR below was crafted so as:
; 1) To have a loop, so we create a loop pass manager
; 2) To be "immutable" in the sense that no pass in the standard
;    pipeline will modify it.
; Since no transformations take place, we don't expect any analyses
; to be invalidated.
; Any invalidation that shows up here is a bug, unless we started modifying
; the IR, in which case we need to make it immutable harder.

; RUN: opt -disable-verify -debug-pass-manager \
; RUN:     -passes='default<O1>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-O --check-prefix=CHECK-O1
; RUN: opt -disable-verify -debug-pass-manager \
; RUN:     -passes='default<O2>' -S  %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-O --check-prefix=CHECK-O2
; RUN: opt -disable-verify -debug-pass-manager \
; RUN:     -passes='default<O3>' -S  %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-O --check-prefix=CHECK-O3
; RUN: opt -disable-verify -debug-pass-manager \
; RUN:     -passes='default<Os>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-O --check-prefix=CHECK-Os
; RUN: opt -disable-verify -debug-pass-manager \
; RUN:     -passes='default<Oz>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-O --check-prefix=CHECK-Oz
; RUN: opt -disable-verify -debug-pass-manager \
; RUN:     -passes='lto-pre-link<O2>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-O --check-prefix=CHECK-O2

; RUN: opt -disable-verify -debug-pass-manager \
; RUN:     -passes-ep-peephole='no-op-function' \
; RUN:     -passes='default<O3>' -S  %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-O --check-prefix=CHECK-O3 \
; RUN:     --check-prefix=CHECK-EP-PEEPHOLE
; RUN: opt -disable-verify -debug-pass-manager \
; RUN:     -passes-ep-late-loop-optimizations='no-op-loop' \
; RUN:     -passes='default<O3>' -S  %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-O --check-prefix=CHECK-O3 \
; RUN:     --check-prefix=CHECK-EP-LOOP-LATE
; RUN: opt -disable-verify -debug-pass-manager \
; RUN:     -passes-ep-loop-optimizer-end='no-op-loop' \
; RUN:     -passes='default<O3>' -S  %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-O --check-prefix=CHECK-O3 \
; RUN:     --check-prefix=CHECK-EP-LOOP-END
; RUN: opt -disable-verify -debug-pass-manager \
; RUN:     -passes-ep-scalar-optimizer-late='no-op-function' \
; RUN:     -passes='default<O3>' -S  %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-O --check-prefix=CHECK-O3 \
; RUN:     --check-prefix=CHECK-EP-SCALAR-LATE
; RUN: opt -disable-verify -debug-pass-manager \
; RUN:     -passes-ep-cgscc-optimizer-late='no-op-cgscc' \
; RUN:     -passes='default<O3>' -S  %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-O --check-prefix=CHECK-O3 \
; RUN:     --check-prefix=CHECK-EP-CGSCC-LATE
; RUN: opt -disable-verify -debug-pass-manager \
; RUN:     -passes-ep-vectorizer-start='no-op-function' \
; RUN:     -passes='default<O3>' -S  %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-O --check-prefix=CHECK-O3 \
; RUN:     --check-prefix=CHECK-EP-VECTORIZER-START

; CHECK-O: Starting llvm::Module pass manager run.
; CHECK-O-NEXT: Running pass: PassManager<{{.*}}Module{{.*}}>
; CHECK-O-NEXT: Starting llvm::Module pass manager run.
; CHECK-O-NEXT: Running pass: ForceFunctionAttrsPass
; CHECK-O-NEXT: Running pass: PassManager<{{.*}}Module{{.*}}>
; CHECK-O-NEXT: Starting llvm::Module pass manager run.
; CHECK-O-NEXT: Running pass: InferFunctionAttrsPass
; CHECK-O-NEXT: Running analysis: TargetLibraryAnalysis
; CHECK-O-NEXT: Running pass: ModuleToFunctionPassAdaptor<{{.*}}PassManager{{.*}}>
; CHECK-O-NEXT: Running analysis: InnerAnalysisManagerProxy
; CHECK-O-NEXT: Starting llvm::Function pass manager run.
; CHECK-O-NEXT: Running pass: SimplifyCFGPass
; CHECK-O-NEXT: Running analysis: TargetIRAnalysis
; CHECK-O-NEXT: Running analysis: AssumptionAnalysis
; CHECK-O-NEXT: Running pass: SROA
; CHECK-O-NEXT: Running analysis: DominatorTreeAnalysis
; CHECK-O-NEXT: Running pass: EarlyCSEPass
; CHECK-O-NEXT: Running analysis: TargetLibraryAnalysis
; CHECK-O-NEXT: Running pass: LowerExpectIntrinsicPass
; CHECK-O-NEXT: Finished llvm::Function pass manager run.
; CHECK-O-NEXT: Running pass: IPSCCPPass
; CHECK-O-NEXT: Running pass: GlobalOptPass
; CHECK-O-NEXT: Running pass: ModuleToFunctionPassAdaptor<{{.*}}PromotePass>
; CHECK-O-NEXT: Running pass: DeadArgumentEliminationPass
; CHECK-O-NEXT: Running pass: ModuleToFunctionPassAdaptor<{{.*}}PassManager{{.*}}>
; CHECK-O-NEXT: Starting llvm::Function pass manager run.
; CHECK-O-NEXT: Running pass: InstCombinePass
; CHECK-O-NEXT: Running analysis: OptimizationRemarkEmitterAnalysis
; CHECK-EP-PEEPHOLE-NEXT: Running pass: NoOpFunctionPass
; CHECK-O-NEXT: Running pass: SimplifyCFGPass
; CHECK-O-NEXT: Finished llvm::Function pass manager run.
; CHECK-O-NEXT: Running pass: RequireAnalysisPass<{{.*}}GlobalsAA
; CHECK-O-NEXT: Running analysis: GlobalsAA
; CHECK-O-NEXT: Running analysis: CallGraphAnalysis
; CHECK-O-NEXT: Running pass: RequireAnalysisPass<{{.*}}ProfileSummaryAnalysis
; CHECK-O-NEXT: Running analysis: ProfileSummaryAnalysis
; CHECK-O-NEXT: Running pass: ModuleToPostOrderCGSCCPassAdaptor<{{.*}}LazyCallGraph{{.*}}>
; CHECK-O-NEXT: Running analysis: InnerAnalysisManagerProxy
; CHECK-O-NEXT: Running analysis: LazyCallGraphAnalysis
; CHECK-O-NEXT: Starting CGSCC pass manager run.
; CHECK-O-NEXT: Running pass: InlinerPass
; CHECK-O-NEXT: Running analysis: OuterAnalysisManagerProxy<{{.*}}LazyCallGraph{{.*}}>
; CHECK-O-NEXT: Running pass: PostOrderFunctionAttrsPass
; CHECK-O-NEXT: Running analysis: FunctionAnalysisManagerCGSCCProxy
; CHECK-O-NEXT: Running analysis: AAManager
; CHECK-O3-NEXT: Running pass: ArgumentPromotionPass
; CHECK-O-NEXT: Running pass: CGSCCToFunctionPassAdaptor<{{.*}}PassManager{{.*}}>
; CHECK-O-NEXT: Starting llvm::Function pass manager run.
; CHECK-O-NEXT: Running pass: SROA
; CHECK-O-NEXT: Running pass: EarlyCSEPass
; CHECK-O-NEXT: Running analysis: MemorySSAAnalysis
; CHECK-O-NEXT: Running pass: SpeculativeExecutionPass
; CHECK-O-NEXT: Running pass: JumpThreadingPass
; CHECK-O-NEXT: Running analysis: LazyValueAnalysis
; CHECK-O-NEXT: Running pass: CorrelatedValuePropagationPass
; CHECK-O-NEXT: Running pass: SimplifyCFGPass
; CHECK-O-NEXT: Running pass: InstCombinePass
; CHECK-O1-NEXT: Running pass: LibCallsShrinkWrapPass
; CHECK-O2-NEXT: Running pass: LibCallsShrinkWrapPass
; CHECK-O3-NEXT: Running pass: LibCallsShrinkWrapPass
; CHECK-EP-PEEPHOLE-NEXT: Running pass: NoOpFunctionPass
; CHECK-O-NEXT: Running pass: TailCallElimPass
; CHECK-O-NEXT: Running pass: SimplifyCFGPass
; CHECK-O-NEXT: Running pass: ReassociatePass
; CHECK-O-NEXT: Running pass: RequireAnalysisPass<{{.*}}OptimizationRemarkEmitterAnalysis
; CHECK-O-NEXT: Running pass: FunctionToLoopPassAdaptor<{{.*}}LoopStandardAnalysisResults{{.*}}>
; CHECK-O-NEXT: Running analysis: LoopAnalysis
; CHECK-O-NEXT: Running analysis: ScalarEvolutionAnalysis
; CHECK-O-NEXT: Running analysis: InnerAnalysisManagerProxy
; CHECK-O-NEXT: Starting Loop pass manager run.
; CHECK-O-NEXT: Running pass: LoopRotatePass
; CHECK-O-NEXT: Running pass: LICM
; CHECK-O-NEXT: Running analysis: OuterAnalysisManagerProxy
; CHECK-O-NEXT: Running pass: SimpleLoopUnswitchPass
; CHECK-O-NEXT: Finished Loop pass manager run.
; CHECK-O-NEXT: Running pass: SimplifyCFGPass
; CHECK-O-NEXT: Running pass: InstCombinePass
; CHECK-O-NEXT: Running pass: FunctionToLoopPassAdaptor<{{.*}}LoopStandardAnalysisResults{{.*}}>
; CHECK-O-NEXT: Starting Loop pass manager run.
; CHECK-O-NEXT: Running pass: IndVarSimplifyPass
; CHECK-O-NEXT: Running pass: LoopIdiomRecognizePass
; CHECK-EP-LOOP-LATE-NEXT: Running pass: NoOpLoopPass
; CHECK-O-NEXT: Running pass: LoopDeletionPass
; CHECK-O-NEXT: Running pass: LoopFullUnrollPass
; CHECK-EP-LOOP-END-NEXT: Running pass: NoOpLoopPass
; CHECK-O-NEXT: Finished Loop pass manager run.
; CHECK-Os-NEXT: Running pass: MergedLoadStoreMotionPass
; CHECK-Os-NEXT: Running pass: GVN
; CHECK-Os-NEXT: Running analysis: MemoryDependenceAnalysis
; CHECK-Oz-NEXT: Running pass: MergedLoadStoreMotionPass
; CHECK-Oz-NEXT: Running pass: GVN
; CHECK-Oz-NEXT: Running analysis: MemoryDependenceAnalysis
; CHECK-O2-NEXT: Running pass: MergedLoadStoreMotionPass
; CHECK-O2-NEXT: Running pass: GVN
; CHECK-O2-NEXT: Running analysis: MemoryDependenceAnalysis
; CHECK-O3-NEXT: Running pass: MergedLoadStoreMotionPass
; CHECK-O3-NEXT: Running pass: GVN
; CHECK-O3-NEXT: Running analysis: MemoryDependenceAnalysis
; CHECK-O-NEXT: Running pass: MemCpyOptPass
; CHECK-O1-NEXT: Running analysis: MemoryDependenceAnalysis
; CHECK-O-NEXT: Running pass: SCCPPass
; CHECK-O-NEXT: Running pass: BDCEPass
; CHECK-O-NEXT: Running analysis: DemandedBitsAnalysis
; CHECK-O-NEXT: Running pass: InstCombinePass
; CHECK-EP-PEEPHOLE-NEXT: Running pass: NoOpFunctionPass
; CHECK-O-NEXT: Running pass: JumpThreadingPass
; CHECK-O-NEXT: Running pass: CorrelatedValuePropagationPass
; CHECK-O-NEXT: Running pass: DSEPass
; CHECK-O-NEXT: Running pass: FunctionToLoopPassAdaptor<{{.*}}LICMPass{{.*}}>
; CHECK-EP-SCALAR-LATE-NEXT: Running pass: NoOpFunctionPass
; CHECK-O-NEXT: Running pass: ADCEPass
; CHECK-O-NEXT: Running analysis: PostDominatorTreeAnalysis
; CHECK-O-NEXT: Running pass: SimplifyCFGPass
; CHECK-O-NEXT: Running pass: InstCombinePass
; CHECK-EP-PEEPHOLE-NEXT: Running pass: NoOpFunctionPass
; CHECK-O-NEXT: Finished llvm::Function pass manager run.
; CHECK-EP-CGSCC-LATE-NEXT: Running pass: NoOpCGSCCPass
; CHECK-O-NEXT: Finished CGSCC pass manager run.
; CHECK-O-NEXT: Finished llvm::Module pass manager run.
; CHECK-O-NEXT: Running pass: PassManager<{{.*}}Module{{.*}}>
; CHECK-O-NEXT: Starting llvm::Module pass manager run.
; CHECK-O-NEXT: Running pass: GlobalOptPass
; CHECK-O-NEXT: Running pass: EliminateAvailableExternallyPass
; CHECK-O-NEXT: Running pass: ReversePostOrderFunctionAttrsPass
; CHECK-O-NEXT: Running pass: RequireAnalysisPass<{{.*}}GlobalsAA
; CHECK-O-NEXT: Running pass: ModuleToFunctionPassAdaptor<{{.*}}PassManager{{.*}}>
; CHECK-O-NEXT: Starting llvm::Function pass manager run.
; CHECK-O-NEXT: Running pass: Float2IntPass
; CHECK-EP-VECTORIZER-START-NEXT: Running pass: NoOpFunctionPass
; CHECK-O-NEXT: Running pass: FunctionToLoopPassAdaptor<{{.*}}LoopRotatePass
; CHECK-O-NEXT: Running pass: LoopDistributePass
; CHECK-O-NEXT: Running pass: LoopVectorizePass
; CHECK-O-NEXT: Running analysis: BlockFrequencyAnalysis
; CHECK-O-NEXT: Running analysis: BranchProbabilityAnalysis
; CHECK-O-NEXT: Running pass: LoopLoadEliminationPass
; CHECK-O-NEXT: Running analysis: LoopAccessAnalysis
; CHECK-O-NEXT: Running pass: InstCombinePass
; CHECK-O-NEXT: Running pass: SLPVectorizerPass
; CHECK-O-NEXT: Running pass: SimplifyCFGPass
; CHECK-O-NEXT: Running pass: InstCombinePass
; CHECK-O-NEXT: Running pass: LoopUnrollPass
; CHECK-O-NEXT: Running analysis: OuterAnalysisManagerProxy
; CHECK-O-NEXT: Running pass: InstCombinePass
; CHECK-O-NEXT: Running pass: RequireAnalysisPass<{{.*}}OptimizationRemarkEmitterAnalysis
; CHECK-O-NEXT: Running pass: FunctionToLoopPassAdaptor<{{.*}}LICMPass
; CHECK-O-NEXT: Running pass: AlignmentFromAssumptionsPass
; CHECK-O-NEXT: Running pass: LoopSinkPass
; CHECK-O-NEXT: Running pass: InstSimplifierPass
; CHECK-O-NEXT: Running pass: DivRemPairsPass
; CHECK-O-NEXT: Running pass: SimplifyCFGPass
; CHECK-O-NEXT: Finished llvm::Function pass manager run.
; CHECK-O-NEXT: Running pass: GlobalDCEPass
; CHECK-O-NEXT: Running pass: ConstantMergePass
; CHECK-O-NEXT: Finished llvm::Module pass manager run.
; CHECK-O-NEXT: Finished llvm::Module pass manager run.
; CHECK-O-NEXT: Running pass: PrintModulePass
;
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
; CHECK-O-NEXT: Finished llvm::Module pass manager run.

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
