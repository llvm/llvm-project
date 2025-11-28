//===- Construction of CodeGen pass pipelines -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Passes/TargetPassBuilder.h"
#include "llvm/CodeGen/CallBrPrepare.h"
#include "llvm/CodeGen/CodeGenPrepare.h"
#include "llvm/CodeGen/DwarfEHPrepare.h"
#include "llvm/CodeGen/ExpandFp.h"
#include "llvm/CodeGen/ExpandLargeDivRem.h"
#include "llvm/CodeGen/ExpandMemCmp.h"
#include "llvm/CodeGen/ExpandReductions.h"
#include "llvm/CodeGen/FinalizeISel.h"
#include "llvm/CodeGen/GCMetadata.h"
#include "llvm/CodeGen/GlobalMergeFunctions.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/LocalStackSlotAllocation.h"
#include "llvm/CodeGen/LowerEmuTLS.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/PreISelIntrinsicLowering.h"
#include "llvm/CodeGen/ReplaceWithVeclib.h"
#include "llvm/CodeGen/SafeStack.h"
#include "llvm/CodeGen/SelectOptimize.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/ShadowStackGCLowering.h"
#include "llvm/CodeGen/SjLjEHPrepare.h"
#include "llvm/CodeGen/StackProtector.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/UnreachableBlockElim.h"
#include "llvm/CodeGen/WasmEHPrepare.h"
#include "llvm/CodeGen/WinEHPrepare.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Passes/CodeGenPassBuilder.h" // Dummy passes only!
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/ObjCARC.h"
#include "llvm/Transforms/Scalar/ConstantHoisting.h"
#include "llvm/Transforms/Scalar/LoopStrengthReduce.h"
#include "llvm/Transforms/Scalar/LoopTermFold.h"
#include "llvm/Transforms/Scalar/MergeICmps.h"
#include "llvm/Transforms/Scalar/PartiallyInlineLibCalls.h"
#include "llvm/Transforms/Scalar/ScalarizeMaskedMemIntrin.h"
#include "llvm/Transforms/Utils/CanonicalizeFreezeInLoops.h"
#include "llvm/Transforms/Utils/LowerGlobalDtors.h"
#include "llvm/Transforms/Utils/LowerInvoke.h"
#include <stack>

using namespace llvm;

TargetPassBuilder::TargetPassBuilder(PassBuilder &PB)
    : PB(PB), TM(PB.TM), OptLevel(TM->getOptLevel()) {}

ModulePassManager TargetPassBuilder::buildPipeline(raw_pwrite_stream &Out,
                                                   raw_pwrite_stream *DwoOut,
                                                   CodeGenFileType FileType,
                                                   MCContext &Ctx) {
  TargetModulePassManager MPM;
  buildCoreCodeGenPipeline(MPM);
  invokeInjectionCallbacks(MPM);
  filterPassList(MPM);
  addPrinterPassesAndFreeMachineFunction(MPM, Out, DwoOut, FileType, Ctx);
  return constructRealPassManager(MPM);
}

void TargetPassBuilder::buildCoreCodeGenPipeline(TargetModulePassManager &MPM) {
  MPM.addPass(buildCodeGenIRPipeline());
  TargetMachineFunctionPassManager MFPM;
  TargetFunctionPassManager FPM;
  addISelPasses(MFPM);
  FPM.addMachineFunctionPass(std::move(MFPM));
  MPM.addFunctionPass(std::move(FPM));
  MPM.addPass(buildCodeGenMIRPipeline());
}

TargetModulePassManager TargetPassBuilder::buildCodeGenIRPipeline() {
  TargetModulePassManager MPM;
  if (TM->useEmulatedTLS())
    MPM.addPass(LowerEmuTLSPass());
  MPM.addPass(PreISelIntrinsicLoweringPass(TM));
  TargetFunctionPassManager FPM;
  FPM.addPass(ExpandLargeDivRemPass(*TM));
  FPM.addPass(ExpandFpPass(*TM, TM->getOptLevel()));

  // Run loop strength reduction before anything else.
  if (TM->getOptLevel() == CodeGenOptLevel::None) {
    // Basic AliasAnalysis support.
    // Add TypeBasedAliasAnalysis before BasicAliasAnalysis so that
    // BasicAliasAnalysis wins if they disagree. This is intended to help
    // support "obvious" type-punning idioms.
    FPM.addPass(RequireAnalysisPass<TypeBasedAA, Function>());
    FPM.addPass(RequireAnalysisPass<ScopedNoAliasAA, Function>());
    FPM.addPass(RequireAnalysisPass<BasicAA, Function>());

    if (!CGPBO.DisableLSR) {
      TargetLoopPassManager LPM;
      LPM.addPass(CanonicalizeFreezeInLoopsPass());
      LPM.addPass(LoopStrengthReducePass());
      if (CGPBO.EnableLoopTermFold)
        LPM.addPass(LoopTermFoldPass());
      FPM.addLoopPass(std::move(LPM));
    }

    // The MergeICmpsPass tries to create memcmp calls by grouping sequences
    // of loads and compares. ExpandMemCmpPass then tries to expand those
    // calls into optimally-sized loads and compares. The transforms are
    // enabled by a target lowering hook.
    if (!CGPBO.DisableMergeICmps)
      FPM.addPass(MergeICmpsPass());
    FPM.addPass(ExpandMemCmpPass(*TM));
  }

  // Run GC lowering passes for builtin collectors
  FPM.addPass(GCLoweringPass());
  MPM.addFunctionPass(std::move(FPM));
  FPM = TargetFunctionPassManager();
  MPM.addPass(ShadowStackGCLoweringPass());
  // PB.invokeGCLoweringEPCallbacks();

  if (TM->getTargetTriple().isOSBinFormatMachO() &&
      !CGPBO.DisableAtExitBasedGlobalDtorLowering) {
    MPM.addFunctionPass(std::move(FPM));
    FPM = TargetFunctionPassManager();
    MPM.addFunctionPass(std::move(FPM));
    MPM.addPass(LowerGlobalDtorsPass());
  }

  // Make sure that no unreachable blocks are instruction selected.
  FPM.addPass(UnreachableBlockElimPass());

  if (OptLevel != CodeGenOptLevel::None) {
    if (!CGPBO.DisableConstantHoisting)
      FPM.addPass(ConstantHoistingPass());
    if (!CGPBO.DisableReplaceWithVecLib)
      FPM.addPass(ReplaceWithVeclib());
    if (!CGPBO.DisablePartialLibcallInlining)
      FPM.addPass(PartiallyInlineLibCallsPass());
  }

  // Instrument function entry after all inlining.
  FPM.addPass(EntryExitInstrumenterPass(/*PostInlining=*/true));

  // Add scalarization of target's unsupported masked memory intrinsics pass.
  // the unsupported intrinsic will be replaced with a chain of basic blocks,
  // that stores/loads element one-by-one if the appropriate mask bit is set.
  FPM.addPass(ScalarizeMaskedMemIntrinPass());

  // Expand reduction intrinsics into shuffle sequences if the target wants
  // to. Allow disabling it for testing purposes.
  if (!CGPBO.DisableExpandReductions)
    FPM.addPass(ExpandReductionsPass());

  // Convert conditional moves to conditional jumps when profitable.
  if (OptLevel != CodeGenOptLevel::None && !CGPBO.DisableSelectOptimize)
    FPM.addPass(SelectOptimizePass(*TM));

  MPM.addFunctionPass(std::move(FPM));
  FPM = TargetFunctionPassManager();
  if (CGPBO.EnableGlobalMergeFunc)
    MPM.addPass(GlobalMergeFuncPass());

  if (OptLevel != CodeGenOptLevel::None && !CGPBO.DisableCGP)
    FPM.addPass(CodeGenPreparePass(*TM));
  addExceptionHandlingPasses(FPM);

  // Add common passes that perform LLVM IR to IR transforms in preparation for
  // instruction selection.
  if (CGPBO.RequiresCodeGenSCCOrder) {
    MPM.addFunctionPass(std::move(FPM));
    FPM = TargetFunctionPassManager();
  }

  // Now we need to force codegen to run according to the callgraph if target
  // requires it.
  FPM.addPass(PreISelInjectionPoint());
  if (OptLevel != CodeGenOptLevel::None)
    FPM.addPass(ObjCARCContractPass());

  FPM.addPass(CallBrPreparePass());

  // Add both the safe stack and the stack protection passes: each of them will
  // only protect functions that have corresponding attributes.
  FPM.addPass(SafeStackPass(*TM));
  FPM.addPass(StackProtectorPass(*TM));

  // All passes which modify the LLVM IR are now complete; run the verifier
  // to ensure that the IR is valid.
  if (!CGPBO.DisableVerify)
    FPM.addPass(VerifierPass());
  if (CGPBO.RequiresCodeGenSCCOrder)
    MPM.addFunctionPassWithPostOrderCGSCC(std::move(FPM));
  else
    MPM.addFunctionPass(std::move(FPM));

  return MPM;
}

void TargetPassBuilder::addISelPasses(TargetMachineFunctionPassManager &MFPM) {
  // Add core instruction selection passes.
  // Enable FastISel with -fast-isel, but allow that to be overridden.
  TM->setO0WantsFastISel(CGPBO.EnableFastISelOption.value_or(true));

  // Determine an instruction selector.
  enum class SelectorType { SelectionDAG, FastISel, GlobalISel };
  SelectorType Selector;

  if (CGPBO.EnableFastISelOption.value_or(false))
    Selector = SelectorType::FastISel;
  else if (CGPBO.EnableGlobalISelOption.value_or(false) ||
           (TM->Options.EnableGlobalISel &&
            CGPBO.EnableGlobalISelOption.value_or(true)))
    Selector = SelectorType::GlobalISel;
  else if (TM->getOptLevel() == CodeGenOptLevel::None &&
           TM->getO0WantsFastISel())
    Selector = SelectorType::FastISel;
  else
    Selector = SelectorType::SelectionDAG;

  // Set consistently TM->Options.EnableFastISel and EnableGlobalISel.
  if (Selector == SelectorType::FastISel) {
    TM->setFastISel(true);
    TM->setGlobalISel(false);
  } else if (Selector == SelectorType::GlobalISel) {
    TM->setFastISel(false);
    TM->setGlobalISel(true);
  }

  // FIXME: Currently debugify is not support in new pass manager,
  // because it inserts module passes between each pass.

  // Add instruction selector passes for global isel if enabled.
  if (Selector == SelectorType::GlobalISel) {
    MFPM.addPass(IRTranslatorPass());
    MFPM.addPass(RegBankSelectPass());
    MFPM.addPass(InstructionSelectPass(OptLevel));
  }

  // Pass to reset the MachineFunction if the ISel failed. Outside of the
  // above if so that the verifier is not added to it.
  MFPM.addPass(ResetMachineFunctionPass());

  // Run the SDAG InstSelector, providing a fallback path when we do not want
  // to abort on not-yet-supported input.
  if (Selector != SelectorType::GlobalISel ||
      TM->Options.GlobalISelAbort != GlobalISelAbortMode::Enable) {
    AddSelectionDAGISelPass(MFPM);
  }

  // Expand pseudo-instructions emitted by ISel. Don't run the verifier
  // before FinalizeISel.
  MFPM.addPass(FinalizeISelPass());
}

void TargetPassBuilder::addRegAllocPipeline(
    TargetMachineFunctionPassManager &MFPM) {
  if (CGPBO.OptimizeRegAlloc.value_or(OptLevel != CodeGenOptLevel::None)) {
    MFPM.addPass(DetectDeadLanesPass());

    MFPM.addPass(InitUndefPass());

    MFPM.addPass(ProcessImplicitDefsPass());

    // LiveVariables currently requires pure SSA form.
    //
    // FIXME: Once TwoAddressInstruction pass no longer uses kill flags,
    // LiveVariables can be removed completely, and LiveIntervals can be
    // directly computed. (We still either need to regenerate kill flags after
    // regalloc, or preferably fix the scavenger to not depend on them).
    // FIXME: UnreachableMachineBlockElim is a dependant pass of LiveVariables.
    // When LiveVariables is removed this has to be removed/moved either.
    // Explicit addition of UnreachableMachineBlockElim allows stopping before
    // or after it with -stop-before/-stop-after.
    MFPM.addPass(UnreachableMachineBlockElimPass());
    MFPM.addPass(RequireAnalysisPass<LiveVariablesAnalysis, MachineFunction>());

    // Edge splitting is smarter with machine loop info.
    MFPM.addPass(RequireAnalysisPass<MachineLoopAnalysis, MachineFunction>());
    MFPM.addPass(PHIEliminationPass());

    // Eventually, we want to run LiveIntervals before PHI elimination.
    if (CGPBO.EarlyLiveIntervals)
      MFPM.addPass(
          RequireAnalysisPass<LiveIntervalsAnalysis, MachineFunction>());

    MFPM.addPass(TwoAddressInstructionPass());
    MFPM.addPass(RegisterCoalescerPass());

    // The machine scheduler may accidentally create disconnected components
    // when moving subregister definitions around, avoid this by splitting them
    // to separate vregs before. Splitting can also improve reg. allocation
    // quality.
    MFPM.addPass(RenameIndependentSubregsPass());

    // PreRA instruction scheduling.
    MFPM.addPass(MachineSchedulerPass(TM));

    /// Add register assign and rewrite passes.
    // Add the selected register allocation pass.
    addRegAllocPass(MFPM, true);

    // Finally rewrite virtual registers.
    MFPM.addPass(VirtRegRewriterPass());

    // Regalloc scoring for ML-driven eviction - noop except when learning a new
    // eviction policy.
    MFPM.addPass(RegAllocScoringPass());

    // Perform stack slot coloring and post-ra machine LICM.
    MFPM.addPass(StackSlotColoringPass());

    // Allow targets to expand pseudo instructions depending on the choice of
    // registers before MachineCopyPropagation.

    // Copy propagate to forward register uses and try to eliminate COPYs that
    // were not coalesced.
    MFPM.addPass(MachineCopyPropagationPass());

    // Run post-ra machine LICM to hoist reloads / remats.
    //
    // FIXME: can this move into MachineLateOptimization?
    MFPM.addPass(MachineLICMPass());
  } else {
    MFPM.addPass(PHIEliminationPass());
    MFPM.addPass(TwoAddressInstructionPass());

    /// Add register assign and rewrite passes.
    if (CGPBO.RegAlloc != RegAllocType::Default &&
        CGPBO.RegAlloc != RegAllocType::Fast)
      report_fatal_error("Must use fast (default) register allocator for "
                         "unoptimized regalloc.");

    addRegAllocPass(MFPM, false);
  }
}

void TargetPassBuilder::addRegAllocPass(TargetMachineFunctionPassManager &MFPM,
                                        bool Optimized) {
  // TODO: Handle register allocator
}

void TargetPassBuilder::addMachineSSAOptimizationPasses(
    TargetMachineFunctionPassManager &MFPM) {
  // Pre-ra tail duplication.
  MFPM.addPass(EarlyTailDuplicatePass());

  // Optimize PHIs before DCE: removing dead PHI cycles may make more
  // instructions dead.
  MFPM.addPass(OptimizePHIsPass());

  // This pass merges large allocas. StackSlotColoring is a different pass
  // which merges spill slots.
  MFPM.addPass(StackColoringPass());

  // If the target requests it, assign local variables to stack slots relative
  // to one another and simplify frame index references where possible.
  MFPM.addPass(LocalStackSlotAllocationPass());

  // With optimization, dead code should already be eliminated. However
  // there is one known exception: lowered code for arguments that are only
  // used by tail calls, where the tail calls reuse the incoming stack
  // arguments directly (see t11 in test/CodeGen/X86/sibcall.ll).
  MFPM.addPass(DeadMachineInstructionElimPass());

  // Allow targets to insert passes that improve instruction level parallelism,
  // like if-conversion. Such passes will typically need dominator trees and
  // loop info, just like LICM and CSE below.
  MFPM.addPass(ILPOptsInjectionPoint());

  // Target can insert passes here that improve instruction level parallelism,
  // like if-conversion. Such passes will typically need dominator trees and
  // loop info, just like LICM and CSE below.
  // Just use injectBefore<EarlyMachineLICMPass>([](){ Your pass builder... });

  MFPM.addPass(EarlyMachineLICMPass());
  MFPM.addPass(MachineCSEPass());

  MFPM.addPass(MachineSinkingPass(CGPBO.EnableSinkAndFold));

  MFPM.addPass(PeepholeOptimizerPass());
  // Clean-up the dead code that may have been generated by peephole
  // rewriting.
  MFPM.addPass(DeadMachineInstructionElimPass());
}

// Find the FSProfile file name. The internal option takes the precedence
// before getting from TargetMachine.
static std::string getFSProfileFile(const CGPassBuilderOption &CGPBO,
                                    const TargetMachine *TM) {
  if (!CGPBO.FSProfileFile.empty())
    return CGPBO.FSProfileFile;
  const std::optional<PGOOptions> &PGOOpt = TM->getPGOOption();
  if (!PGOOpt || PGOOpt->Action != PGOOptions::SampleUse)
    return std::string();
  return PGOOpt->ProfileFile;
}

// Find the Profile remapping file name. The internal option takes the
// precedence before getting from TargetMachine.
static std::string getFSRemappingFile(const CGPassBuilderOption &CGPBO,
                                      const TargetMachine *TM) {
  if (!CGPBO.FSRemappingFile.empty())
    return CGPBO.FSRemappingFile;
  const std::optional<PGOOptions> &PGOOpt = TM->getPGOOption();
  if (!PGOOpt || PGOOpt->Action != PGOOptions::SampleUse)
    return std::string();
  return PGOOpt->ProfileRemappingFile;
}

TargetModulePassManager TargetPassBuilder::buildCodeGenMIRPipeline() {
  TargetModulePassManager MPM;
  TargetMachineFunctionPassManager MFPM;
  if (OptLevel != CodeGenOptLevel::None) {
    addMachineSSAOptimizationPasses(MFPM);
  } else {
    // If the target requests it, assign local variables to stack slots relative
    // to one another and simplify frame index references where possible.
    MFPM.addPass(LocalStackSlotAllocationPass());
  }

  if (TM->Options.EnableIPRA)
    MFPM.addPass(RegUsageInfoPropagationPass());

  // Add a FSDiscriminator pass right before RA, so that we could get
  // more precise SampleFDO profile for RA.

  if (EnableFSDiscriminator) {
    MFPM.addPass(
        MIRAddFSDiscriminatorsPass(sampleprof::FSDiscriminatorPass::Pass1));
    const std::string ProfileFile = getFSProfileFile(CGPBO, TM);
    if (!ProfileFile.empty() && !CGPBO.DisableRAFSProfileLoader)
      MFPM.addPass(MIRProfileLoaderNewPass(
          ProfileFile, getFSRemappingFile(CGPBO, TM),
          sampleprof::FSDiscriminatorPass::Pass1, nullptr));
  }

  // Run register allocation and passes that are tightly coupled with it,
  // including phi elimination and scheduling.
  addRegAllocPipeline(MFPM);

  MFPM.addPass(RemoveRedundantDebugValuesPass());

  MFPM.addPass(FixupStatepointCallerSavedPass());

  // Insert prolog/epilog code.  Eliminate abstract frame index references...
  if (OptLevel != CodeGenOptLevel::None) {
    MFPM.addPass(PostRAMachineSinkingPass());
    MFPM.addPass(ShrinkWrapPass());
  }

  // Prolog/Epilog inserter needs a TargetMachine to instantiate. But only
  // do so if it hasn't been disabled, substituted, or overridden.
  if (!CGPBO.DisablePrologEpilogInserterPass)
    MFPM.addPass(PrologEpilogCodeInserterPass());

  /// Add passes that optimize machine instructions after register
  /// allocation.
  if (OptLevel != CodeGenOptLevel::None) {
    // Cleanup of redundant immediate/address loads.
    MFPM.addPass(MachineLateInstrsCleanupPass());

    // Branch folding must be run after regalloc and prolog/epilog insertion.
    MFPM.addPass(BranchFolderPass(CGPBO.EnableTailMerge));

    // Tail duplication.
    // Note that duplicating tail just increases code size and degrades
    // performance for targets that require Structured Control Flow.
    // In addition it can also make CFG irreducible. Thus we disable it.
    if (!TM->requiresStructuredCFG())
      MFPM.addPass(TailDuplicatePass());

    // Copy propagation.
    MFPM.addPass(MachineCopyPropagationPass());
  }

  // Expand pseudo instructions before second scheduling pass.
  MFPM.addPass(ExpandPostRAPseudosPass());

  if (CGPBO.EnableImplicitNullChecks)
    MFPM.addPass(ImplicitNullChecksPass());

  // Second pass scheduler.
  // Let Target optionally insert this pass by itself at some other
  // point.
  if (OptLevel != CodeGenOptLevel::None &&
      !TM->targetSchedulesPostRAScheduling()) {
    if (CGPBO.MISchedPostRA)
      MFPM.addPass(PostMachineSchedulerPass(TM));
    else
      MFPM.addPass(PostRASchedulerPass(TM));
  }

  // GC
  // MFPM.addPass(GCMachineCodeAnalysis());

  // Basic block placement.
  if (OptLevel != CodeGenOptLevel::None) {
    if (EnableFSDiscriminator) {
      MFPM.addPass(
          MIRAddFSDiscriminatorsPass(sampleprof::FSDiscriminatorPass::Pass2));
      const std::string ProfileFile = getFSProfileFile(CGPBO, TM);
      if (!ProfileFile.empty() && !CGPBO.DisableLayoutFSProfileLoader)
        MFPM.addPass(MIRProfileLoaderNewPass(
            ProfileFile, getFSRemappingFile(CGPBO, TM),
            sampleprof::FSDiscriminatorPass::Pass2, nullptr));
      MFPM.addPass(MachineBlockPlacementPass(CGPBO.EnableTailMerge));
      // Run a separate pass to collect block placement statistics.
      if (CGPBO.EnableBlockPlacementStats)
        MFPM.addPass(MachineBlockPlacementStatsPass());
    }
  }

  // Insert before XRay Instrumentation.
  MFPM.addPass(FEntryInserterPass());

  MFPM.addPass(XRayInstrumentationPass());
  MFPM.addPass(PatchableFunctionPass());

  if (TM->Options.EnableIPRA)
    // Collect register usage information and produce a register mask of
    // clobbered registers, to be used to optimize call sites.
    MFPM.addPass(RegUsageInfoCollectorPass());

  // FIXME: Some backends are incompatible with running the verifier after
  // addPreEmitPass.  Maybe only pass "false" here for those targets?
  MFPM.addPass(FuncletLayoutPass());

  MFPM.addPass(RemoveLoadsIntoFakeUsesPass());
  MFPM.addPass(StackMapLivenessPass());
  MFPM.addPass(LiveDebugValuesPass(TM->Options.ShouldEmitDebugEntryValues()));
  MFPM.addPass(MachineSanitizerBinaryMetadataPass());

  if (TM->Options.EnableMachineOutliner && OptLevel != CodeGenOptLevel::None &&
      CGPBO.EnableMachineOutliner != RunOutliner::NeverOutline) {
    bool RunOnAllFunctions =
        (CGPBO.EnableMachineOutliner == RunOutliner::AlwaysOutline);
    bool AddOutliner =
        RunOnAllFunctions || TM->Options.SupportsDefaultOutlining;
    if (AddOutliner) {
      if (CGPBO.RequiresCodeGenSCCOrder) {
        MPM.addFunctionPassWithPostOrderCGSCC(
            TargetFunctionPassManager().addMachineFunctionPass(
                std::move(MFPM)));
      } else {
        MPM.addFunctionPass(TargetFunctionPassManager().addMachineFunctionPass(
            std::move(MFPM)));
      }

      MPM.addPass(MachineOutlinerPass(RunOnAllFunctions));
      MFPM = TargetMachineFunctionPassManager();
    }
  }

  if (CGPBO.GCEmptyBlocks)
    MFPM.addPass(GCEmptyBasicBlocksPass());

  if (EnableFSDiscriminator)
    MFPM.addPass(
        MIRAddFSDiscriminatorsPass(sampleprof::FSDiscriminatorPass::PassLast));

  // Machine function splitter uses the basic block sections feature.
  // When used along with `-basic-block-sections=`, the basic-block-sections
  // feature takes precedence. This means functions eligible for
  // basic-block-sections optimizations (`=all`, or `=list=` with function
  // included in the list profile) will get that optimization instead.
  if (TM->Options.EnableMachineFunctionSplitter ||
      CGPBO.EnableMachineFunctionSplitter) {
    const std::string ProfileFile = getFSProfileFile(CGPBO, TM);
    if (!ProfileFile.empty()) {
      if (EnableFSDiscriminator) {
        MFPM.addPass(MIRProfileLoaderNewPass(
            ProfileFile, getFSRemappingFile(CGPBO, TM),
            sampleprof::FSDiscriminatorPass::PassLast, nullptr));
      } else {
        // Sample profile is given, but FSDiscriminator is not
        // enabled, this may result in performance regression.
        WithColor::warning()
            << "Using AutoFDO without FSDiscriminator for MFS may regress "
               "performance.\n";
      }
    }
    MFPM.addPass(MachineFunctionSplitterPass());
    if (CGPBO.SplitStaticData || TM->Options.EnableStaticDataPartitioning) {
      // The static data splitter pass is a machine function pass. and
      // static data annotator pass is a module-wide pass. See the file comment
      // in StaticDataAnnotator.cpp for the motivation.
      MFPM.addPass(StaticDataSplitterPass());
      if (CGPBO.RequiresCodeGenSCCOrder) {
        MPM.addFunctionPassWithPostOrderCGSCC(
            TargetFunctionPassManager().addMachineFunctionPass(
                std::move(MFPM)));
      } else {
        MPM.addFunctionPass(TargetFunctionPassManager().addMachineFunctionPass(
            std::move(MFPM)));
      }
      MFPM = TargetMachineFunctionPassManager();
      MPM.addPass(StaticDataAnnotatorPass());
    }
  }
  // We run the BasicBlockSections pass if either we need BB sections or BB
  // address map (or both).
  if (TM->getBBSectionsType() != llvm::BasicBlockSection::None ||
      TM->Options.BBAddrMap) {
    if (TM->getBBSectionsType() == llvm::BasicBlockSection::List) {
      MFPM.addPass(
          BasicBlockSectionsProfileReaderPass(TM->getBBSectionsFuncListBuf()));
      MFPM.addPass(BasicBlockPathCloningPass());
    }
    MFPM.addPass(BasicBlockSectionsPass());
  }

  MFPM.addPass(PostBBSectionsInjectionPoint());

  if (!CGPBO.DisableCFIFixup && TM->Options.EnableCFIFixup)
    MFPM.addPass(CFIFixupPass());

  MFPM.addPass(StackFrameLayoutAnalysisPass());
  MFPM.addPass(PreEmitInjectionPoint());
  if (CGPBO.RequiresCodeGenSCCOrder)
    MPM.addFunctionPassWithPostOrderCGSCC(
        TargetFunctionPassManager().addMachineFunctionPass(std::move(MFPM)));
  else
    MPM.addFunctionPass(
        TargetFunctionPassManager().addMachineFunctionPass(std::move(MFPM)));
  return MPM;
}

void TargetPassBuilder::addExceptionHandlingPasses(
    TargetFunctionPassManager &FPM) {
  const MCAsmInfo *MCAI = TM->getMCAsmInfo();

  switch (MCAI->getExceptionHandlingType()) {
  case ExceptionHandling::SjLj:
    // SjLj piggy-backs on dwarf for this bit. The cleanups done apply to both
    // Dwarf EH prepare needs to be run after SjLj prepare. Otherwise,
    // catch info can get misplaced when a selector ends up more than one block
    // removed from the parent invoke(s). This could happen when a landing
    // pad is shared by multiple invokes and is also a target of a normal
    // edge from elsewhere.
    FPM.addPass(SjLjEHPreparePass(TM));
    [[fallthrough]];
  case ExceptionHandling::DwarfCFI:
  case ExceptionHandling::ARM:
  case ExceptionHandling::AIX:
  case ExceptionHandling::ZOS:
    FPM.addPass(DwarfEHPreparePass(*TM));
    break;
  case ExceptionHandling::WinEH:
    // We support using both GCC-style and MSVC-style exceptions on Windows, so
    // add both preparation passes. Each pass will only actually run if it
    // recognizes the personality function.
    FPM.addPass(WinEHPreparePass());
    FPM.addPass(DwarfEHPreparePass(*TM));
    break;
  case ExceptionHandling::Wasm:
    // Wasm EH uses Windows EH instructions, but it does not need to demote PHIs
    // on catchpads and cleanuppads because it does not outline them into
    // funclets. Catchswitch blocks are not lowered in SelectionDAG, so we
    // should remove PHIs there.
    FPM.addPass(WinEHPreparePass(/*DemoteCatchSwitchPHIOnly=*/true));
    FPM.addPass(WasmEHPreparePass());
    break;
  case ExceptionHandling::None:
    FPM.addPass(LowerInvokePass());
    // The lower invoke pass may create unreachable code. Remove it.
    FPM.addPass(UnreachableBlockElimPass());
    break;
  }
}

void TargetPassBuilder::filterPassList(TargetModulePassManager &MPM) const {
  PassList &Passes = MPM.Passes;
  auto *PIC = PB.getPassInstrumentationCallbacks();
  auto ESSI = TargetPassConfig::getStartStopInfo(*PIC);
  if (!ESSI)
    report_fatal_error(ESSI.takeError());
  auto SSI = *ESSI;
  size_t StartCnt = 0, StopCnt = 0;
  bool HandledStartAfter = !SSI.StartAfter, HandledStopAfter = !SSI.StopAfter;
  bool ShouldRemove = !SSI.StartPass.empty();
  for (auto I = Passes.begin(), E = Passes.end(); I != E;) {
    // Handle disabled pass firstly.
    if (isPassDisabled(I->Name) || I->IsInjectionPoint) {
      I = Passes.erase(I);
      continue;
    }

    auto PassName = PIC->getPassNameForClassName(I->Name);
    if (!SSI.StartPass.empty() && PassName == SSI.StartPass) {
      ++StartCnt;
      if (StartCnt == SSI.StartInstanceNum)
        ShouldRemove = SSI.StartAfter;
    }
    if (!SSI.StopPass.empty() && PassName == SSI.StopPass) {
      ++StopCnt;
      if (StopCnt == SSI.StopInstanceNum)
        ShouldRemove = !SSI.StopAfter;
    }
    bool Inc = true;
    if (ShouldRemove && !PassName.starts_with("RequireAnalysisPass")) {
      Inc = false;
      I = Passes.erase(I);
    }
    if (!HandledStartAfter && !SSI.StartPass.empty() &&
        PassName == SSI.StartPass && SSI.StartAfter &&
        StartCnt == SSI.StartInstanceNum) {
      HandledStartAfter = true;
      ShouldRemove = false;
    }
    if (!HandledStopAfter && !SSI.StopPass.empty() &&
        PassName == SSI.StopPass && SSI.StopAfter &&
        StopCnt == SSI.StopInstanceNum) {
      HandledStopAfter = true;
      ShouldRemove = true;
    }
    if (Inc)
      ++I;
  }
}

void TargetPassBuilder::addPrinterPassesAndFreeMachineFunction(
    TargetModulePassManager &MPM, raw_pwrite_stream &Out,
    raw_pwrite_stream *DwoOut, CodeGenFileType FileType, MCContext &Ctx) {

  PassList &Passes = MPM.Passes;
  PassList::iterator LastModulePassInsertPoint =
      llvm::find_if(llvm::reverse(Passes), [](detail::PassWrapper &PW) {
        return PW.Ctor.index() == 0; // The last module pass
      }).base();

  // FIXME: CodeGenFileType here is not enough, we need an output type for MC
  if (TargetPassConfig::willCompleteCodeGenPipeline()) {
    TargetModulePassManager AsmPrinterPM;
    TargetFunctionPassManager FPM;
    // MachineFunctionPassManager MFPM;
    // TODO: Insert asm printer initialization pass at LastModulePassInsertPoint
    // TODO: Insert asm printer
    FPM.addPass(InvalidateAnalysisPass<MachineFunctionAnalysis>());
    // TODO: Add asm printer finalization pass.
    if (CGPBO.RequiresCodeGenSCCOrder)
      AsmPrinterPM.addFunctionPassWithPostOrderCGSCC((std::move(FPM)));
    else
      AsmPrinterPM.addFunctionPass(std::move(FPM));
    MPM.addPass(std::move(AsmPrinterPM));
  } else {
    Passes.insert(LastModulePassInsertPoint,
                  detail::PassWrapper(PrintMIRPreparePass(Out)));
    TargetFunctionPassManager FPM;
    TargetMachineFunctionPassManager MFPM;
    MFPM.addPass(PrintMIRPass(Out));
    FPM.addMachineFunctionPass(std::move(MFPM));
    FPM.addPass(InvalidateAnalysisPass<MachineFunctionAnalysis>());
    if (CGPBO.RequiresCodeGenSCCOrder)
      MPM.addFunctionPassWithPostOrderCGSCC(std::move(FPM));
    else
      MPM.addFunctionPass(std::move(FPM));
  }
}

ModulePassManager TargetPassBuilder::constructRealPassManager(
    TargetModulePassManager &TMPM) const {
  ModulePassManager MPM;
  FunctionPassManager FPM;
  LoopPassManager LPM;
  MachineFunctionPassManager MFPM;

  std::stack<size_t> S({0});
  bool InCGSCC = false, LastInCGSCC = false;
  for (auto &P : TMPM.Passes) {
    std::visit(
        [&](auto &Ctor) {
          if (P.InCGSCC)
            InCGSCC = true;
          size_t VarIdx = P.Ctor.index();
          while (VarIdx < S.top()) {
            switch (S.top()) {
            case 3:
              if (!MFPM.isEmpty())
                FPM.addPass(createFunctionToMachineFunctionPassAdaptor(
                    std::move(MFPM)));
              MFPM = MachineFunctionPassManager();
              S.pop();
              break;
            case 2:
              if (!LPM.isEmpty())
                FPM.addPass(llvm::createFunctionToLoopPassAdaptor(
                    std::move(LPM), /*UseMemorySSA=*/true));
              LPM = LoopPassManager();
              S.pop();
              break;
            case 1:
              if (!FPM.isEmpty()) {
                if (CGPBO.RequiresCodeGenSCCOrder && LastInCGSCC) {
                  InCGSCC = false;
                  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(
                      createCGSCCToFunctionPassAdaptor(std::move(FPM))));
                } else {
                  MPM.addPass(
                      createModuleToFunctionPassAdaptor(std::move(FPM)));
                }
              }
              FPM = FunctionPassManager();
              S.pop();
              break;
            case 0:
              break;
            default:
              llvm_unreachable("Invalid pass manager type!");
            }
          }
          while (VarIdx > S.top())
            S.push(S.top() + 1);

          if constexpr (std::is_same_v<
                            std::remove_reference_t<decltype(Ctor)>,
                            llvm::unique_function<void(ModulePassManager &)>>) {
            Ctor(MPM);
          } else if constexpr (std::is_same_v<
                                   std::remove_reference_t<decltype(Ctor)>,
                                   llvm::unique_function<void(
                                       FunctionPassManager &)>>) {
            if (!LastInCGSCC && P.InCGSCC) {
              MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
              FPM = FunctionPassManager();
            }
            Ctor(FPM);
          } else if constexpr (std::is_same_v<
                                   std::remove_reference_t<decltype(Ctor)>,
                                   llvm::unique_function<void(
                                       LoopPassManager &)>>) {
            Ctor(LPM);
          } else if constexpr (std::is_same_v<
                                   decltype(Ctor),
                                   llvm::unique_function<void(
                                       MachineFunctionPassManager &)>>) {
            Ctor(MFPM);
          }

          LastInCGSCC = P.InCGSCC;
        },
        P.Ctor);
  }

  if (!MFPM.isEmpty())
    FPM.addPass(createFunctionToMachineFunctionPassAdaptor(std::move(MFPM)));
  if (!LPM.isEmpty())
    FPM.addPass(createFunctionToLoopPassAdaptor(std::move(LPM),
                                                /*UseMemorySSA=*/true));
  if (!FPM.isEmpty()) {
    if (CGPBO.RequiresCodeGenSCCOrder)
      MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(
          createCGSCCToFunctionPassAdaptor(std::move(FPM))));
    else
      MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
  }

  return MPM;
}

void TargetPassBuilder::invokeInjectionCallbacks(
    TargetModulePassManager &MPM) const {
  PassList &Passes = MPM.Passes;
  for (auto I = Passes.begin(), E = Passes.end(); I != E; ++I)
    for (auto &C : InjectionCallbacks)
      I = C(Passes, I);
}

void TargetPassBuilder::anchor() {}
