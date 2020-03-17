; Basic test for the new LTO pipeline.
; For now the only difference is between -O1 and everything else, so
; -O2, -O3, -Os, -Oz are the same.

; RUN: opt -disable-verify -debug-pass-manager \
; RUN:     -passes='lto<O1>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-O --check-prefix=CHECK-O1
; RUN: opt -disable-verify -debug-pass-manager \
; RUN:     -passes='lto<O2>' -S  %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-O --check-prefix=CHECK-O2
; RUN: opt -disable-verify -debug-pass-manager \
; RUN:     -passes='lto<O3>' -S  %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-O --check-prefix=CHECK-O2 \
; RUN:     --check-prefix=CHECK-O3
; RUN: opt -disable-verify -debug-pass-manager \
; RUN:     -passes='lto<Os>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-O --check-prefix=CHECK-O2
; RUN: opt -disable-verify -debug-pass-manager \
; RUN:     -passes='lto<Oz>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-O --check-prefix=CHECK-O2
; RUN: opt -disable-verify -debug-pass-manager \
; RUN:     -passes='lto<O3>' -S  %s -passes-ep-peephole='no-op-function' 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-O --check-prefix=CHECK-O2 \
; RUN:     --check-prefix=CHECK-O3 --check-prefix=CHECK-EP-Peephole

; CHECK-O: Running analysis: PassInstrumentationAnalysis
; CHECK-O-NEXT: Starting llvm::Module pass manager run.
; CHECK-O-NEXT: Running pass: PassManager<{{.*}}Module
; CHECK-O-NEXT: Starting llvm::Module pass manager run.
; CHECK-O-NEXT: Running pass: GlobalDCEPass
; CHECK-O-NEXT: Running pass: ForceFunctionAttrsPass
; CHECK-O-NEXT: Running pass: InferFunctionAttrsPass
; CHECK-O-NEXT: Running analysis: InnerAnalysisManagerProxy<{{.*}}Module
; CHECK-O-NEXT: Running analysis: TargetLibraryAnalysis
; CHECK-O-NEXT: Running analysis: PassInstrumentationAnalysis
; CHECK-O1-NEXT: Running pass: ModuleToPostOrderCGSCCPassAdaptor<{{.*}}PostOrderFunctionAttrsPass>
; CHECK-O2-NEXT: Running pass: ModuleToFunctionPassAdaptor<{{.*}}PassManager{{.*}}>
; CHECK-O2-NEXT: Running analysis: PassInstrumentationAnalysis
; CHECK-O2-NEXT: Starting llvm::Function pass manager run.
; CHECK-O2-NEXT: Running pass: CallSiteSplittingPass on foo
; CHECK-O2-NEXT: Running analysis: TargetLibraryAnalysis on foo
; CHECK-O2-NEXT: Running analysis: TargetIRAnalysis on foo
; CHECK-O2-NEXT: Running analysis: DominatorTreeAnalysis on foo
; CHECK-O2-NEXT: Finished llvm::Function pass manager run.
; CHECK-O2-NEXT: PGOIndirectCallPromotion
; CHECK-O2-NEXT: Running analysis: ProfileSummaryAnalysis
; CHECK-O2-NEXT: Running analysis: OptimizationRemarkEmitterAnalysis
; CHECK-O2-NEXT: Running pass: IPSCCPPass
; CHECK-O2-NEXT: Running analysis: AssumptionAnalysis on foo
; CHECK-O2-NEXT: Running pass: CalledValuePropagationPass
; CHECK-O2-NEXT: Running pass: ModuleToPostOrderCGSCCPassAdaptor<{{.*}}PostOrderFunctionAttrsPass>
; CHECK-O-NEXT: Running analysis: InnerAnalysisManagerProxy<{{.*}}SCC
; CHECK-O-NEXT: Running analysis: LazyCallGraphAnalysis
; CHECK-O1-NEXT: Running analysis: TargetLibraryAnalysis
; CHECK-O1-NEXT: Running analysis: PassInstrumentationAnalysis
; CHECK-O-NEXT: Running analysis: FunctionAnalysisManagerCGSCCProxy
; CHECK-O-NEXT: Running analysis: PassInstrumentationAnalysis
; CHECK-O-NEXT: Running analysis: OuterAnalysisManagerProxy<{{.*}}LazyCallGraph{{.*}}>
; CHECK-O-NEXT: Running analysis: AAManager
; CHECK-O-NEXT: Running pass: ReversePostOrderFunctionAttrsPass
; CHECK-O-NEXT: Running analysis: CallGraphAnalysis
; CHECK-O-NEXT: Running pass: GlobalSplitPass
; CHECK-O-NEXT: Running pass: WholeProgramDevirtPass
; CHECK-O1-NEXT: Running pass: LowerTypeTestsPass
; CHECK-O2-NEXT: Running pass: GlobalOptPass
; CHECK-O2-NEXT: Running pass: ModuleToFunctionPassAdaptor<{{.*}}PromotePass>
; CHECK-O2-NEXT: Running pass: ConstantMergePass
; CHECK-O2-NEXT: Running pass: DeadArgumentEliminationPass
; CHECK-O2-NEXT: Running pass: ModuleToFunctionPassAdaptor<{{.*}}PassManager{{.*}}>
; CHECK-O2-NEXT: Starting llvm::Function pass manager run.
; CHECK-O3-NEXT: Running pass: AggressiveInstCombinePass
; CHECK-O2-NEXT: Running pass: InstCombinePass
; CHECK-O2-NEXT: Running analysis: OuterAnalysisManagerProxy
; CHECK-EP-Peephole-NEXT: Running pass: NoOpFunctionPass
; CHECK-O2-NEXT: Finished llvm::Function pass manager run.
; CHECK-O2-NEXT: Running pass: ModuleInlinerWrapperPass
; CHECK-O2-NEXT: Running analysis: InlineAdvisorAnalysis
; CHECK-O2-NEXT: Starting llvm::Module pass manager run.
; CHECK-O2-NEXT: Running pass: ModuleToPostOrderCGSCCPassAdaptor<{{.*}}PassManager{{.*}}>
; CHECK-O2-NEXT: Starting CGSCC pass manager run.
; CHECK-O2-NEXT: Running pass: InlinerPass
; CHECK-O2-NEXT: Finished CGSCC pass manager run.
; CHECK-O2-NEXT: Finished llvm::Module pass manager run.
; CHECK-O2-NEXT: Running pass: GlobalOptPass
; CHECK-O2-NEXT: Running pass: GlobalDCEPass
; CHECK-O2-NEXT: Running pass: ModuleToFunctionPassAdaptor<{{.*}}PassManager{{.*}}>
; CHECK-O2-NEXT: Starting llvm::Function pass manager run.
; CHECK-O2-NEXT: Running pass: InstCombinePass
; CHECK-EP-Peephole-NEXT: Running pass: NoOpFunctionPass
; CHECK-O2-NEXT: Running pass: JumpThreadingPass
; CHECK-O2-NEXT: Running analysis: LazyValueAnalysis
; CHECK-O2-NEXT: Running pass: SROA on foo
; CHECK-O2-NEXT: Running pass: TailCallElimPass on foo
; CHECK-O2-NEXT: Finished llvm::Function pass manager run.
; CHECK-O2-NEXT: Running pass: ModuleToPostOrderCGSCCPassAdaptor<{{.*}}PostOrderFunctionAttrsPass>
; CHECK-O2-NEXT: Running pass: ModuleToFunctionPassAdaptor<{{.*}}PassManager{{.*}}>
; CHECK-O2-NEXT: Running analysis: MemoryDependenceAnalysis
; CHECK-O2-NEXT: Running analysis: PhiValuesAnalysis
; CHECK-O2-NEXT: Running analysis: DemandedBitsAnalysis
; CHECK-O2-NEXT: Running pass: CrossDSOCFIPass
; CHECK-O2-NEXT: Running pass: LowerTypeTestsPass
; CHECK-O-NEXT: Running pass: LowerTypeTestsPass
; CHECK-O2-NEXT: Running pass: ModuleToFunctionPassAdaptor<{{.*}}SimplifyCFGPass>
; CHECK-O2-NEXT: Running pass: EliminateAvailableExternallyPass
; CHECK-O2-NEXT: Running pass: GlobalDCEPass
; CHECK-O-NEXT: Finished llvm::Module pass manager run.
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
