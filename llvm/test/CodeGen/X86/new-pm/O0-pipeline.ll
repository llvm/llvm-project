; When EXPENSIVE_CHECKS are enabled, the machine verifier appears between each
; pass. Ignore it with 'grep -v'.
; RUN: llc -mtriple=x86_64-- -O0 -debug-pass-manager -enable-new-pm < %s \
; RUN:    -o /dev/null 2>&1 | grep -v 'Verify generated machine code' | FileCheck %s

; REQUIRES: asserts

; CHECK-LABEL: Running pass: RequireAnalysisPass{{.*}}
; CHECK-NEXT: Running analysis: ProfileSummaryAnalysis on [module]
; CHECK-NEXT: Running pass: PreISelIntrinsicLoweringPass on [module]
; CHECK-NEXT: Running analysis: {{.*InnerAnalysisManagerProxy.*}} on [module]
; CHECK-NEXT: Running pass: ExpandLargeDivRemPass on f (1 instruction)
; CHECK-NEXT: Running pass: ExpandLargeFpConvertPass on f (1 instruction)
; CHECK-NEXT: Running pass: AtomicExpandPass on f (1 instruction)
; CHECK-NEXT: Running pass: VerifierPass on f (1 instruction)
; CHECK-NEXT: Running analysis: VerifierAnalysis on f
; CHECK-NEXT: Running pass: GCLoweringPass on f (1 instruction)
; CHECK-NEXT: Running pass: ShadowStackGCLoweringPass on f (1 instruction)
; CHECK-NEXT: Running pass: LowerConstantIntrinsicsPass on f (1 instruction)
; CHECK-NEXT: Running analysis: TargetLibraryAnalysis on f
; CHECK-NEXT: Running pass: UnreachableBlockElimPass on f (1 instruction)
; CHECK-NEXT: Running pass: ExpandVectorPredicationPass on f (1 instruction)
; CHECK-NEXT: Running analysis: TargetIRAnalysis on f
; CHECK-NEXT: Running pass: ScalarizeMaskedMemIntrinPass on f (1 instruction)
; CHECK-NEXT: Running pass: ExpandReductionsPass on f (1 instruction)
; CHECK-NEXT: Running pass: IndirectBrExpandPass on f (1 instruction)
; CHECK-NEXT: Running pass: DwarfEHPreparePass on f (1 instruction)
; CHECK-NEXT: Running pass: CallBrPreparePass on f (1 instruction)
; CHECK-NEXT: Running pass: SafeStackPass on f (1 instruction)
; CHECK-NEXT: Running pass: StackProtectorPass on f (1 instruction)
; CHECK-NEXT: Running pass: VerifierPass on f (1 instruction)
; CHECK-NEXT: Running analysis: MachineModuleAnalysis on [module]
; CHECK-NEXT: Running pass: {{.*}}::X86ISelDagPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86GlobalBaseRegPass on f
; CHECK-NEXT: Running pass: FinalizeISelPass on f
; CHECK-NEXT: Running pass: MachineVerifierPass on f
; CHECK-NEXT: Running pass: LocalStackSlotPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86SpeculativeLoadHardeningPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86FlagsCopyLoweringDummyPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86DynAllocaExpanderPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86FastPreTileConfigPass on f
; CHECK-NEXT: Running pass: PHIEliminationPass on f
; CHECK-NEXT: Running pass: TwoAddressInstructionPass on f
; CHECK-NEXT: Running pass: RegAllocPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86FastTileConfigPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86LowerTileCopyPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86FloatingPointStackifierPass on f
; CHECK-NEXT: Running pass: RemoveRedundantDebugValuesPass on f
; CHECK-NEXT: Running pass: FixupStatepointCallerSavedPass on f
; CHECK-NEXT: Running pass: PrologEpilogInserterPass on f
; CHECK-NEXT: Running pass: ExpandPostRAPseudosPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86ExpandPseudoPass on f
; CHECK-NEXT: Running pass: MachineKCFIPass on f
; CHECK-NEXT: Running pass: FEntryInserterPass on f
; CHECK-NEXT: Running pass: XRayInstrumentationPass on f
; CHECK-NEXT: Running pass: PatchableFunctionPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86IndirectBranchTrackingPass on f
; CHECK-NEXT: Running pass: {{.*}}::X86IssueVZeroUpperPass on f
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
  ret void
}
