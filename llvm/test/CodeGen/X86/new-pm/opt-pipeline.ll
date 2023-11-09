; When EXPENSIVE_CHECKS are enabled, the machine verifier appears between each
; pass. Ignore it with 'grep -v'.
; RUN: llc -mtriple=x86_64-- -O1 -debug-pass-manager -enable-new-pm < %s \
; RUN:    -o /dev/null 2>&1 | FileCheck %s
; RUN: llc -mtriple=x86_64-- -O2 -debug-pass-manager -enable-new-pm < %s \
; RUN:    -o /dev/null 2>&1 | FileCheck %s
; RUN: llc -mtriple=x86_64-- -O3 -debug-pass-manager -enable-new-pm < %s \
; RUN:    -o /dev/null 2>&1 | FileCheck %s

; REQUIRES: asserts

; CHECK-LABEL: Running pass: RequireAnalysisPass{{.*}}
; CHECK-NEXT: Running analysis: ProfileSummaryAnalysis on [module]
; CHECK-NEXT: Running pass: PreISelIntrinsicLoweringPass on [module]
; CHECK-NEXT: Running analysis: {{.*InnerAnalysisManagerProxy.*}} on [module]
; CHECK-NEXT: Running pass: ExpandLargeDivRemPass on f (3 instructions)
; CHECK-NEXT: Running pass: ExpandLargeFpConvertPass on f (3 instructions)
; CHECK-NEXT: Running pass: AtomicExpandPass on f (3 instructions)
; CHECK-NEXT: Running pass: VerifierPass on f (3 instructions)
; CHECK-NEXT: Running analysis: VerifierAnalysis on f
; CHECK-NEXT: Running pass: LoopSimplifyPass on f (3 instructions)
; CHECK-NEXT: Running analysis: LoopAnalysis on f
; CHECK-NEXT: Running analysis: DominatorTreeAnalysis on f
; CHECK-NEXT: Running analysis: AssumptionAnalysis on f
; CHECK-NEXT: Running analysis: TargetIRAnalysis on f
; CHECK-NEXT: Running pass: LCSSAPass on f (3 instructions)
; CHECK-NEXT: Running analysis: MemorySSAAnalysis on f
; CHECK-NEXT: Running analysis: AAManager on f
; CHECK-NEXT: Running analysis: TargetLibraryAnalysis on f
; CHECK-NEXT: Running analysis: BasicAA on f
; CHECK-NEXT: Running analysis: ScopedNoAliasAA on f
; CHECK-NEXT: Running analysis: TypeBasedAA on f
; CHECK-NEXT: Running analysis: {{.*OuterAnalysisManagerProxy.*}} on f
; CHECK-NEXT: Running analysis: ScalarEvolutionAnalysis on f
; CHECK-NEXT: Running analysis: {{.*InnerAnalysisManagerProxy.*}} on f
; CHECK-NEXT: Running pass: CanonicalizeFreezeInLoopsPass on b
; CHECK-NEXT: Running pass: LoopSimplifyPass on f (3 instructions)
; CHECK-NEXT: Running pass: LCSSAPass on f (3 instructions)
; CHECK-NEXT: Running pass: LoopStrengthReducePass on b
; CHECK-NEXT: Running analysis: IVUsersAnalysis on b
; CHECK-NEXT: Running pass: MergeICmpsPass on f (3 instructions)
; CHECK-NEXT: Running pass: ExpandMemCmpPass on f (3 instructions)
; CHECK-NEXT: Running pass: GCLoweringPass on f (3 instructions)
; CHECK-NEXT: Running pass: ShadowStackGCLoweringPass on f (3 instructions)
; CHECK-NEXT: Running pass: LowerConstantIntrinsicsPass on f (3 instructions)
; CHECK-NEXT: Running pass: UnreachableBlockElimPass on f (3 instructions)
; CHECK-NEXT: Clearing all analysis results for: <possibly invalidated loop>
; CHECK-NEXT: Invalidating analysis: VerifierAnalysis on f
; CHECK-NEXT: Invalidating analysis: LoopAnalysis on f
; CHECK-NEXT: Invalidating analysis: MemorySSAAnalysis on f
; CHECK-NEXT: Invalidating analysis: ScalarEvolutionAnalysis on f
; CHECK-NEXT: Invalidating analysis: {{.*InnerAnalysisManagerProxy.*}} on f
; CHECK-NEXT: Running pass: ConstantHoistingPass on f (2 instructions)
; CHECK-NEXT: Running analysis: BlockFrequencyAnalysis on f
; CHECK-NEXT: Running analysis: BranchProbabilityAnalysis on f
; CHECK-NEXT: Running analysis: LoopAnalysis on f
; CHECK-NEXT: Running analysis: PostDominatorTreeAnalysis on f
; CHECK-NEXT: Running pass: ReplaceWithVeclib on f (2 instructions)
; CHECK-NEXT: Running pass: PartiallyInlineLibCallsPass on f (2 instructions)
; CHECK-NEXT: Running pass: ExpandVectorPredicationPass on f (2 instructions)
; CHECK-NEXT: Running pass: ScalarizeMaskedMemIntrinPass on f (2 instructions)
; CHECK-NEXT: Running pass: ExpandReductionsPass on f (2 instructions)
; CHECK-NEXT: Running pass: TLSVariableHoistPass on f (2 instructions)
; CHECK-NEXT: Running pass: InterleavedAccessPass on f (2 instructions)
; CHECK-NEXT: Running pass: IndirectBrExpandPass on f (2 instructions)
; CHECK-NEXT: Running pass: CodeGenPreparePass on f (2 instructions)
; CHECK-NEXT: Running pass: DwarfEHPreparePass on f (2 instructions)
; CHECK-NEXT: Running pass: CallBrPreparePass on f (2 instructions)
; CHECK-NEXT: Running pass: SafeStackPass on f (2 instructions)
; CHECK-NEXT: Running pass: StackProtectorPass on f (2 instructions)
; CHECK-NEXT: Running pass: VerifierPass on f (2 instructions)
; CHECK-NEXT: Running analysis: VerifierAnalysis on f
; CHECK-NEXT: Running analysis: MachineModuleAnalysis on [module]
; CHECK-NEXT: Running pass: {{.*}}::X86ISelDagPass on f
; CHECK-NEXT: Running pass: {{.*}}::CleanupLocalDynamicTLSPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86GlobalBaseRegPass on f
; CHECK-NEXT: Running pass: FinalizeISelPass on f
; CHECK-NEXT: Running pass: MachineVerifierPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86DomainReassignmentPass on f
; CHECK-NEXT: Running pass: EarlyTailDuplicatePass on f
; CHECK-NEXT: Running pass: OptimizePHIsPass on f
; CHECK-NEXT: Running pass: StackColoringPass on f
; CHECK-NEXT: Running pass: LocalStackSlotPass on f
; CHECK-NEXT: Running pass: DeadMachineInstructionElimPass on f
; CHECK-NEXT: Running pass: EarlyIfConverterPass on f
; CHECK-NEXT: Running pass: MachineCombinerPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86CmovConverterDummyPass on f
; CHECK-NEXT: Running pass: EarlyMachineLICMPass on f
; CHECK-NEXT: Running pass: MachineCSEPass on f
; CHECK-NEXT: Running pass: MachineSinkingPass on f
; CHECK-NEXT: Running pass: PeepholeOptimizerPass on f
; CHECK-NEXT: Running pass: DeadMachineInstructionElimPass on f
; CHECK-NEXT: Running pass: LiveRangeShrinkPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86FixupSetCCPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86OptimizeLEAsPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86CallFrameOptimizationPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86AvoidStoreForwardingBlocksPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86SpeculativeLoadHardeningPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86FlagsCopyLoweringDummyPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86DynAllocaExpanderPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86PreTileConfigPass on f
; CHECK-NEXT: Running pass: DetectDeadLanesPass on f
; CHECK-NEXT: Running pass: ProcessImplicitDefsPass on f
; CHECK-NEXT: Running pass: PHIEliminationPass on f
; CHECK-NEXT: Running pass: TwoAddressInstructionPass on f
; CHECK-NEXT: Running pass: RegisterCoalescerPass on f
; CHECK-NEXT: Running pass: RenameIndependentSubregsPass on f
; CHECK-NEXT: Running pass: MachineSchedulerPass on f
; CHECK-NEXT: Running pass: RegAllocPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86FastTileConfigPass on f
; CHECK-NEXT: Running pass: MachineCopyPropagationPass on f
; CHECK-NEXT: Running pass: MachineLICMPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86LowerTileCopyPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86FloatingPointStackifierPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86LoadValueInjectionRetHardeningPass on f
; CHECK-NEXT: Running pass: RemoveRedundantDebugValuesPass on f
; CHECK-NEXT: Running pass: FixupStatepointCallerSavedPass on f
; CHECK-NEXT: Running pass: PostRAMachineSinkingPass on f
; CHECK-NEXT: Running pass: ShrinkWrapPass on f
; CHECK-NEXT: Running pass: PrologEpilogInserterPass on f
; CHECK-NEXT: Running pass: BranchFolderPass on f
; CHECK-NEXT: Running pass: TailDuplicatePass on f
; CHECK-NEXT: Running pass: MachineLateInstrsCleanupPass on f
; CHECK-NEXT: Running pass: MachineCopyPropagationPass on f
; CHECK-NEXT: Running pass: ExpandPostRAPseudosPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86ExpandPseudoPass on f
; CHECK-NEXT: Running pass: MachineKCFIPass on f
; CHECK-NEXT: Running pass: PostRASchedulerPass on f
; CHECK-NEXT: Running pass: MachineBlockPlacementPass on f
; CHECK-NEXT: Running pass: FEntryInserterPass on f
; CHECK-NEXT: Running pass: XRayInstrumentationPass on f
; CHECK-NEXT: Running pass: PatchableFunctionPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86ExecutionDomainFixPass on f
; CHECK-NEXT: Running pass: BreakFalseDepsPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86IndirectBranchTrackingPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86IssueVZeroUpperPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86FixupBWInstsPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86PadShortFunctionsPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86FixupLEAsPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86FixupInstTuningPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86FixupVectorConstantsPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86EvexToVexInstsPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86DiscriminateMemOpsPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86InsertPrefetchPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86InsertX87waitPass on f
; CHECK-NEXT: Running pass: FuncletLayoutPass on f
; CHECK-NEXT: Running pass: StackMapLivenessPass on f
; CHECK-NEXT: Running pass: LiveDebugValuesPass on f
; CHECK-NEXT: Running pass: MachineSanitizerBinaryMetadata on f
; CHECK-NEXT: Running pass: StackFrameLayoutAnalysisPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86SpeculativeExecutionSideEffectSuppressionPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86IndirectThunksPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86ReturnThunksPass on f
; CHECK-NEXT: Running pass: CFIInstrInserterPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86LoadValueInjectionRetHardeningPass on f
; CHECK-NEXT: Running pass: PseudoProbeInserterPass on [module]
; CHECK-NEXT: Running pass: UnpackMachineBundlesPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86AsmPrinterPass on f
; CHECK-NEXT: Running pass: FreeMachineFunctionPass on f

define void @f() {
  br label %b
b:
  br label %b
  ret void
}
