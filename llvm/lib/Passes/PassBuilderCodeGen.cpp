//===- Construction of code generation pass pipelines ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file provides the implementation of the PassBuilder based on our
/// static pass registry as well as related functionality.
///
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/CallBrPrepare.h"
#include "llvm/CodeGen/CodeGenPrepare.h"
#include "llvm/CodeGen/DwarfEHPrepare.h"
#include "llvm/CodeGen/ExpandLargeDivRem.h"
#include "llvm/CodeGen/ExpandLargeFpConvert.h"
#include "llvm/CodeGen/ExpandMemCmp.h"
#include "llvm/CodeGen/ExpandReductions.h"
#include "llvm/CodeGen/FinalizeISel.h"
#include "llvm/CodeGen/GCMetadata.h"
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
#include "llvm/CodeGen/ShadowStackGCLowering.h"
#include "llvm/CodeGen/SjLjEHPrepare.h"
#include "llvm/CodeGen/StackProtector.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/UnreachableBlockElim.h"
#include "llvm/CodeGen/WasmEHPrepare.h"
#include "llvm/CodeGen/WinEHPrepare.h"
#include "llvm/MC/MCAsmInfo.h"
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
#include "llvm/Transforms/Scalar/MergeICmps.h"
#include "llvm/Transforms/Scalar/PartiallyInlineLibCalls.h"
#include "llvm/Transforms/Scalar/ScalarizeMaskedMemIntrin.h"
#include "llvm/Transforms/Scalar/TLSVariableHoist.h"
#include "llvm/Transforms/Utils/LowerGlobalDtors.h"
#include "llvm/Transforms/Utils/LowerInvoke.h"

namespace llvm {
extern cl::opt<std::string> FSRemappingFile;
}

using namespace llvm;

void PassBuilder::invokeCodeGenIREarlyEPCallbacks(ModulePassManager &MPM) {
  for (auto &C : CodeGenIREarlyEPCallbacks)
    C(MPM);
}

void PassBuilder::invokeGCLoweringEPCallbacks(FunctionPassManager &FPM) {
  for (auto &C : GCLoweringEPCallbacks)
    C(FPM);
}

void PassBuilder::invokeISelPrepareEPCallbacks(ModulePassManager &MPM) {
  for (auto &C : ISelPrepareEPCallbacks)
    C(MPM);
}

void PassBuilder::invokeMachineSSAOptimizationEarlyEPCallbacks(
    MachineFunctionPassManager &MFPM) {
  for (auto &C : MachineSSAOptimizationEarlyEPCallbacks)
    C(MFPM);
}

void PassBuilder::invokeMachineSSAOptimizationLastEPCallbacks(
    MachineFunctionPassManager &MFPM) {
  for (auto &C : MachineSSAOptimizationLastEPCallbacks)
    C(MFPM);
}

void PassBuilder::invokePreRegAllocEPCallbacks(
    MachineFunctionPassManager &MFPM) {
  for (auto &C : PreRegAllocEPCallbacks)
    C(MFPM);
}

void PassBuilder::invokePreRegBankSelectEPCallbacks(
    MachineFunctionPassManager &MFPM) {
  for (auto &C : PreRegBankSelectEPCallbacks)
    C(MFPM);
}

void PassBuilder::invokePreGlobalInstructionSelectEPCallbacks(
    MachineFunctionPassManager &MFPM) {
  for (auto &C : PreGlobalInstructionSelectEPCallbacks)
    C(MFPM);
}

void PassBuilder::invokePostGlobalInstructionSelectEPCallbacks(
    MachineFunctionPassManager &MFPM) {
  for (auto &C : PostGlobalInstructionSelectEPCallbacks)
    C(MFPM);
}

void PassBuilder::invokeILPOptsEPCallbacks(MachineFunctionPassManager &MFPM) {
  for (auto &C : ILPOptsEPCallbacks)
    C(MFPM);
}

void PassBuilder::invokeMachineLateOptimizationEPCallbacks(
    MachineFunctionPassManager &MFPM) {
  for (auto &C : MachineLateOptimizationEPCallbacks)
    C(MFPM);
}

void PassBuilder::invokeMIEmitEPCallbacks(MachineFunctionPassManager &MFPM) {
  for (auto &C : MIEmitEPCallbacks)
    C(MFPM);
}

void PassBuilder::invokePreEmitEPCallbacks(MachineFunctionPassManager &MFPM) {
  for (auto &C : PreEmitEPCallbacks)
    C(MFPM);
}

void PassBuilder::invokePostRegAllocEPCallbacks(
    MachineFunctionPassManager &MFPM) {
  for (auto &C : PostRegAllocEPCallbacks)
    C(MFPM);
}

void PassBuilder::invokePreSched2EPCallbacks(MachineFunctionPassManager &MFPM) {
  for (auto &C : PreSched2EPCallbacks)
    C(MFPM);
}

void PassBuilder::invokePostBBSectionsEPCallbacks(
    MachineFunctionPassManager &MFPM) {
  for (auto &C : PostBBSectionsEPCallbacks)
    C(MFPM);
}

void PassBuilder::addDefaultCodeGenPreparePasses(ModulePassManager &MPM) {
  FunctionPassManager FPM;
  // CodeGen prepare
  if (TM->getOptLevel() != CodeGenOptLevel::None && !CGPBO.DisableCGP)
    FPM.addPass(CodeGenPreparePass(TM));
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
}

Error PassBuilder::addDefaultRegAllocFastPasses(
    MachineFunctionPassManager &MFPM) {
  MFPM.addPass(PHIEliminationPass());
  MFPM.addPass(TwoAddressInstructionPass());
  if (auto Err = addRegAllocPass(MFPM))
    return Err;
  return Error::success();
}

Error PassBuilder::addDefaultRegAllocOptimizedPasses(
    MachineFunctionPassManager &MFPM) {
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
  // FIXME: Some X86 tests failed because of incomplete pipeline.
  // MFPM.addPass(RequireAnalysisPass<LiveVariablesAnalysis,
  // MachineFunction>());

  // Edge splitting is smarter with machine loop info.
  MFPM.addPass(RequireAnalysisPass<MachineLoopAnalysis, MachineFunction>());
  MFPM.addPass(PHIEliminationPass());

  if (CGPBO.EarlyLiveIntervals)
    MFPM.addPass(RequireAnalysisPass<LiveIntervalsAnalysis, MachineFunction>());

  MFPM.addPass(TwoAddressInstructionPass());
  MFPM.addPass(RegisterCoalescerPass());

  // The machine scheduler may accidentally create disconnected components
  // when moving subregister definitions around, avoid this by splitting them
  // to separate vregs before. Splitting can also improve reg. allocation
  // quality.
  MFPM.addPass(RenameIndependentSubregsPass());

  // PreRA instruction scheduling.
  MFPM.addPass(MachineSchedulerPass());

  if (auto Err = addRegAllocPass(MFPM))
    return Err;
  // Finally rewrite virtual registers.
  MFPM.addPass(VirtRegRewriterPass());

  // Regalloc scoring for ML-driven eviction - noop except when learning a new
  // eviction policy.
  MFPM.addPass(RegAllocScoringPass());

  return Error::success();
}

bool PassBuilder::isOptimizedRegAlloc() const {
  return CGPBO.OptimizeRegAlloc.value_or(TM->getOptLevel() !=
                                         CodeGenOptLevel::None);
}

// Find the Profile remapping file name. The internal option takes the
// precedence before getting from TargetMachine.
static std::string getFSRemappingFile(const TargetMachine *TM,
                                      const CGPassBuilderOption &Options) {
  if (!Options.FSRemappingFile.empty())
    return Options.FSRemappingFile;
  const std::optional<PGOOptions> &PGOOpt = TM->getPGOOption();
  if (PGOOpt == std::nullopt || PGOOpt->Action != PGOOptions::SampleUse)
    return std::string();
  return PGOOpt->ProfileRemappingFile;
}

// Find the FSProfile file name. The internal option takes the precedence
// before getting from TargetMachine.
static std::string getFSProfileFile(const TargetMachine *TM,
                                    const CGPassBuilderOption &Options) {
  if (!Options.FSProfileFile.empty())
    return Options.FSProfileFile;
  const std::optional<PGOOptions> &PGOOpt = TM->getPGOOption();
  if (PGOOpt == std::nullopt || PGOOpt->Action != PGOOptions::SampleUse)
    return std::string();
  return PGOOpt->ProfileFile;
}

Error PassBuilder::addExceptionHandlingPasses(FunctionPassManager &FPM) {
  const MCAsmInfo *MCAI = TM->getMCAsmInfo();
  if (!MCAI)
    return make_error<StringError>("No MCAsmInfo!", inconvertibleErrorCode());
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
    FPM.addPass(DwarfEHPreparePass(TM));
    break;
  case ExceptionHandling::WinEH:
    // We support using both GCC-style and MSVC-style exceptions on Windows, so
    // add both preparation passes. Each pass will only actually run if it
    // recognizes the personality function.
    FPM.addPass(WinEHPreparePass());
    FPM.addPass(DwarfEHPreparePass(TM));
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
  return Error::success();
}

Error PassBuilder::addInstructionSelectorPasses(
    MachineFunctionPassManager &MFPM) {
  CodeGenOptLevel OptLevel = TM->getOptLevel();

  // Core ISel
  // Enable FastISel with -fast-isel, but allow that to be overridden.
  TM->setO0WantsFastISel(CGPBO.EnableFastISelOption.value_or(true));
  // Determine an instruction selector.
  enum class SelectorType { SelectionDAG, FastISel, GlobalISel };
  SelectorType Selector;

  CGPBO.EnableFastISelOption.value_or(false);
  if (CGPBO.EnableFastISelOption.value_or(false))
    Selector = SelectorType::FastISel;

  else if (CGPBO.EnableGlobalISelOption.value_or(false) ||
           (TM->Options.EnableGlobalISel &&
            !CGPBO.EnableGlobalISelOption.value_or(false)))
    Selector = SelectorType::GlobalISel;
  else if (OptLevel == CodeGenOptLevel::None && TM->getO0WantsFastISel())
    Selector = SelectorType::FastISel;
  else
    Selector = SelectorType::SelectionDAG;

  // Set consistently TM.Options.EnableFastISel and EnableGlobalISel.
  if (Selector == SelectorType::FastISel) {
    TM->setFastISel(true);
    TM->setGlobalISel(false);
  } else if (Selector == SelectorType::GlobalISel) {
    TM->setFastISel(false);
    TM->setGlobalISel(true);
  }

  // Add instruction selector passes.
  if (Selector == SelectorType::GlobalISel) {
    MFPM.addPass(IRTranslatorPass());
    MFPM.addPass(LegalizerPass());

    // Before running the register bank selector, ask the target if it
    // wants to run some passes.
    invokePreRegBankSelectEPCallbacks(MFPM);
    MFPM.addPass(RegBankSelectPass());

    invokePreGlobalInstructionSelectEPCallbacks(MFPM);
    MFPM.addPass(InstructionSelectPass());
    invokePostGlobalInstructionSelectEPCallbacks(MFPM);

    // Pass to reset the MachineFunction if the ISel failed.
    MFPM.addPass(ResetMachineFunctionPass(
        TM->Options.GlobalISelAbort == GlobalISelAbortMode::DisableWithDiag,
        TM->Options.GlobalISelAbort == GlobalISelAbortMode::Enable));

    // Provide a fallback path when we do not want to abort on
    // not-yet-supported input.
    if (TM->Options.GlobalISelAbort != GlobalISelAbortMode::Enable) {
      if (!AddInstSelectorCallback)
        return make_error<StringError>("No InstSelectorCallback!",
                                       inconvertibleErrorCode());
      AddInstSelectorCallback(MFPM);
    }
  } else {
    if (!AddInstSelectorCallback)
      return make_error<StringError>("No InstSelectorCallback!",
                                     inconvertibleErrorCode());
    AddInstSelectorCallback(MFPM);
  }
  return Error::success();
}

void PassBuilder::addMachineSSAOptimizationPasses(
    MachineFunctionPassManager &MFPM) {
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
  invokeILPOptsEPCallbacks(MFPM);

  MFPM.addPass(EarlyMachineLICMPass());
  MFPM.addPass(MachineCSEPass());
  MFPM.addPass(MachineSinkingPass());
  MFPM.addPass(PeepholeOptimizerPass());
  // Clean-up the dead code that may have been generated by peephole
  // rewriting.
  MFPM.addPass(DeadMachineInstructionElimPass());
}

static Error setStartStop(ModulePassManager &MPM,
                          PassInstrumentationCallbacks *PIC) {
  auto StartStopInfoOrErr = TargetPassConfig::getStartStopInfo();
  if (!StartStopInfoOrErr)
    return StartStopInfoOrErr.takeError();
  auto &SSI = *StartStopInfoOrErr;

  if (SSI.StartPass.empty() && SSI.StopPass.empty())
    return Error::success();
  if (!PIC) {
    return make_error<StringError>("Need PassInstrumentationCallbacks!",
                                   inconvertibleErrorCode());
  }

  static const std::vector<StringRef> SpecialPasses = {
      "InvalidateAnalysisPass", "MachineVerifierPass", "PrintMIRPass",
      "PrintMIRPreparePass",    "RequireAnalysisPass", "VerifierPass"};

  bool Started = SSI.StartPass.empty(), Stopped = false;
  // Return true if pass is skipped.
  auto Filter = [&, StartInstanceNum = 0u,
                 StopInstanceNum = 0u](StringRef Name) mutable {
    if (isSpecialPass(Name, SpecialPasses))
      return false;

    bool ShouldDrop = true;
    StringRef CurPassName = PIC->getPassNameForClassName(Name);

    // Set instance counters correctly.
    if (!SSI.StartPass.empty() && CurPassName == SSI.StartPass) {
      ++StartInstanceNum;
      if (StartInstanceNum == SSI.StartInstanceNum)
        Started = true;
    }
    if (!SSI.StopPass.empty() && CurPassName == SSI.StopPass) {
      ++StopInstanceNum;
      if (StopInstanceNum == SSI.StopInstanceNum)
        Stopped = true;
    }

    // Border case.
    const bool AtStartBorder = !SSI.StartPass.empty() && Started &&
                               CurPassName == SSI.StartPass &&
                               StartInstanceNum == SSI.StartInstanceNum;
    const bool AtStopBorder = !SSI.StopPass.empty() && Stopped &&
                              CurPassName == SSI.StopPass &&
                              StopInstanceNum == SSI.StopInstanceNum;
    if (AtStartBorder)
      ShouldDrop = SSI.StartAfter;
    if (AtStopBorder)
      ShouldDrop = !SSI.StopAfter;
    if (!AtStartBorder && !AtStopBorder)
      ShouldDrop = !Started || Stopped;

    return ShouldDrop;
  };

  MPM.eraseIf(Filter);
  if (!Started) {
    return make_error<StringError>(
        "Can't find start pass \"" + SSI.StartPass + "\".",
        std::make_error_code(std::errc::invalid_argument));
  }
  if (!Stopped && !SSI.StopPass.empty()) {
    return make_error<StringError>(
        "Can't find stop pass \"" + SSI.StopPass + "\".",
        std::make_error_code(std::errc::invalid_argument));
  }
  return Error::success();
}

Error PassBuilder::addRegisterAllocatorPasses(
    MachineFunctionPassManager &MFPM) {
  return isOptimizedRegAlloc() ? AddRegAllocOptimizedCallback(MFPM)
                               : AddRegAllocFastCallback(MFPM);
}

Error PassBuilder::addRegAllocPass(MachineFunctionPassManager &MFPM,
                                   StringRef FilterName) {
  std::optional<RegAllocFilterFunc> FilterFunc =
      parseRegAllocFilter(FilterName);
  if (!FilterFunc) {
    return make_error<StringError>(
        formatv("Unknown register filter name: {0}", FilterName).str(),
        std::make_error_code(std::errc::invalid_argument));
  }

  if (RegAllocPasses.contains(FilterName)) {
    MFPM.addPass(std::move(RegAllocPasses[FilterName]));
    return Error::success();
  }

  // Add default register allocator.
  if (isOptimizedRegAlloc()) {
    MFPM.addPass(RAGreedyPass());
  } else {
    RegAllocFastPassOptions Opts;
    Opts.Filter = *FilterFunc;
    Opts.FilterName = FilterName;
    MFPM.addPass(RegAllocFastPass(Opts));
  }
  return Error::success();
}

Error PassBuilder::addMachinePasses(ModulePassManager &MPM,
                                    FunctionPassManager &FPM,
                                    MachineFunctionPassManager &MFPM) {
  CodeGenOptLevel OptLevel = TM->getOptLevel();

  // Expand pseudo-instructions emitted by ISel. Don't run the verifier before
  // FinalizeISel.
  MFPM.addPass(FinalizeISelPass());

  // Add passes that optimize machine instructions in SSA form.
  if (OptLevel != CodeGenOptLevel::None) {
    invokeMachineSSAOptimizationEarlyEPCallbacks(MFPM);
    addMachineSSAOptimizationPasses(MFPM);
    invokeMachineSSAOptimizationLastEPCallbacks(MFPM);
  } else {
    MFPM.addPass(LocalStackSlotAllocationPass());
  }

  if (TM->Options.EnableIPRA)
    MFPM.addPass(RegUsageInfoPropagationPass());

  // Run pre-ra passes.
  invokePreRegAllocEPCallbacks(MFPM);

  if (EnableFSDiscriminator) {
    MFPM.addPass(
        MIRAddFSDiscriminatorsPass(sampleprof::FSDiscriminatorPass::Pass1));
    const std::string ProfileFile = getFSProfileFile(TM, CGPBO);
    if (!ProfileFile.empty() && !CGPBO.DisableRAFSProfileLoader)
      MFPM.addPass(MIRProfileLoaderNewPass(
          ProfileFile, getFSRemappingFile(TM, CGPBO),
          sampleprof::FSDiscriminatorPass::Pass1, nullptr));
  }

  if (auto Err = addRegisterAllocatorPasses(MFPM))
    return Err;

  invokePostRegAllocEPCallbacks(MFPM);

  MFPM.addPass(RemoveRedundantDebugValuesPass());
  MFPM.addPass(FixupStatepointCallerSavedPass());

  // Insert prolog/epilog code.  Eliminate abstract frame index references...
  if (OptLevel != CodeGenOptLevel::None) {
    MFPM.addPass(PostRAMachineSinkingPass());
    MFPM.addPass(ShrinkWrapPass());
  }

  if (!CGPBO.DisablePrologEpilogInserterPass)
    MFPM.addPass(PrologEpilogInserterPass());
  /// Add passes that optimize machine instructions after register allocation.
  if (OptLevel != CodeGenOptLevel::None)
    invokeMachineLateOptimizationEPCallbacks(MFPM);

  // Expand pseudo instructions before second scheduling pass.
  MFPM.addPass(ExpandPostRAPseudosPass());

  // Run pre-sched2 passes.
  invokePreSched2EPCallbacks(MFPM);

  if (CGPBO.EnableImplicitNullChecks)
    MFPM.addPass(ImplicitNullChecksPass());

  // Second pass scheduler.
  // Let Target optionally insert this pass by itself at some other
  // point.
  if (OptLevel != CodeGenOptLevel::None &&
      !TM->targetSchedulesPostRAScheduling()) {
    if (CGPBO.MISchedPostRA)
      MFPM.addPass(PostMachineSchedulerPass());
    else
      MFPM.addPass(PostRASchedulerPass());
  }

  // GC, replacement for GCMachineCodeAnalysis
  MFPM.addPass(GCMachineCodeInsertionPass());

  // Basic block placement.
  if (OptLevel != CodeGenOptLevel::None) {
    if (EnableFSDiscriminator) {
      MFPM.addPass(
          MIRAddFSDiscriminatorsPass(sampleprof::FSDiscriminatorPass::Pass2));
      const std::string ProfileFile = getFSProfileFile(TM, CGPBO);
      if (!ProfileFile.empty() && !CGPBO.DisableLayoutFSProfileLoader)
        MFPM.addPass(MIRProfileLoaderNewPass(
            ProfileFile, getFSRemappingFile(TM, CGPBO),
            sampleprof::FSDiscriminatorPass::Pass2, nullptr));
    }
    MFPM.addPass(MachineBlockPlacementPass());
    // Run a separate pass to collect block placement statistics.
    if (CGPBO.EnableBlockPlacementStats)
      MFPM.addPass(MachineBlockPlacementStatsPass());
  }

  // Insert before XRay Instrumentation.
  MFPM.addPass(FEntryInserterPass());
  MFPM.addPass(XRayInstrumentationPass());
  MFPM.addPass(PatchableFunctionPass());

  invokePreEmitEPCallbacks(MFPM);

  if (TM->Options.EnableIPRA)
    // Collect register usage information and produce a register mask of
    // clobbered registers, to be used to optimize call sites.
    MFPM.addPass(RegUsageInfoCollectorPass());

  // FIXME: Some backends are incompatible with running the verifier after
  // addPreEmitPass.  Maybe only pass "false" here for those targets?
  MFPM.addPass(FuncletLayoutPass());

  MFPM.addPass(StackMapLivenessPass());
  MFPM.addPass(LiveDebugValuesPass());
  MFPM.addPass(MachineSanitizerBinaryMetadata());

  if (TM->Options.EnableMachineOutliner && OptLevel != CodeGenOptLevel::None &&
      CGPBO.EnableMachineOutliner != RunOutliner::NeverOutline) {
    bool RunOnAllFunctions =
        (CGPBO.EnableMachineOutliner == RunOutliner::AlwaysOutline);
    bool AddOutliner =
        RunOnAllFunctions || TM->Options.SupportsDefaultOutlining;
    if (AddOutliner) {
      FPM.addPass(createFunctionToMachineFunctionPassAdaptor(std::move(MFPM)));
      if (CGPBO.RequiresCodeGenSCCOrder)
        MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(
            createCGSCCToFunctionPassAdaptor(std::move(FPM))));
      else
        MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
      MPM.addPass(MachineOutlinerPass(RunOnAllFunctions));
      FPM = FunctionPassManager();
      MFPM = MachineFunctionPassManager();
    }
  }

  if (CGPBO.GCEmptyBlocks)
    MFPM.addPass(GCEmptyBasicBlocksPass());

  if (EnableFSDiscriminator)
    MFPM.addPass(
        MIRAddFSDiscriminatorsPass(sampleprof::FSDiscriminatorPass::PassLast));

  bool NeedsBBSections =
      TM->getBBSectionsType() != llvm::BasicBlockSection::None;
  // Machine function splitter uses the basic block sections feature. Both
  // cannot be enabled at the same time. We do not apply machine function
  // splitter if -basic-block-sections is requested.
  if (!NeedsBBSections && (TM->Options.EnableMachineFunctionSplitter ||
                           CGPBO.EnableMachineFunctionSplitter)) {
    const std::string ProfileFile = getFSProfileFile(TM, CGPBO);
    if (!ProfileFile.empty()) {
      if (EnableFSDiscriminator) {
        MFPM.addPass(MIRProfileLoaderNewPass(
            ProfileFile, getFSRemappingFile(TM, CGPBO),
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
  }

  // We run the BasicBlockSections pass if either we need BB sections or BB
  // address map (or both).
  if (NeedsBBSections || TM->Options.BBAddrMap) {
    if (TM->getBBSectionsType() == llvm::BasicBlockSection::List)
      MFPM.addPass(BasicBlockPathCloningPass());
    MFPM.addPass(BasicBlockSectionsPass());
  }

  invokePostBBSectionsEPCallbacks(MFPM);

  if (!CGPBO.DisableCFIFixup && TM->Options.EnableCFIFixup)
    MFPM.addPass(CFIFixupPass());

  MFPM.addPass(StackFrameLayoutAnalysisPass());

  // Add passes that directly emit MI after all other MI passes.
  invokeMIEmitEPCallbacks(MFPM);

  return Error::success();
}

Error PassBuilder::buildDefaultCodeGenPipeline(ModulePassManager &TopLevelMPM,
                                               raw_pwrite_stream &Out,
                                               raw_pwrite_stream *DwoOut,
                                               CodeGenFileType FileType,
                                               MCContext &Ctx) {
  if (!TM)
    return make_error<StringError>("Need a TargetMachine instance!",
                                   inconvertibleErrorCode());

  if (CustomCodeGenPipelineBuilderCallback) {
    ModulePassManager MPM;
    if (auto Err = CustomCodeGenPipelineBuilderCallback(MPM, Out, DwoOut,
                                                        FileType, Ctx))
      return Err;
    MPM.eraseIf([&](StringRef Name) { return DisabledPasses.contains(Name); });
    if (auto Err = setStartStop(MPM, PIC))
      return Err;
    TopLevelMPM.addPass(std::move(MPM));
    return Error::success();
  }

  CodeGenOptLevel OptLevel = TM->getOptLevel();
  if (auto Err = parseRegAllocOption(CGPBO.RegAlloc))
    return Err;

  bool PrintAsm = TargetPassConfig::willCompleteCodeGenPipeline();
  bool PrintMIR = !PrintAsm && FileType != CodeGenFileType::Null;

  ModulePassManager MPM;
  FunctionPassManager FPM;
  MachineFunctionPassManager MFPM;

  if (!CGPBO.DisableVerify)
    MPM.addPass(VerifierPass());

  // IR part
  // TODO: Remove RequireAnalysisPass<MachineModuleAnalysis, Module>()
  // when we port AsmPrinter.
  MPM.addPass(RequireAnalysisPass<MachineModuleAnalysis, Module>());
  MPM.addPass(RequireAnalysisPass<ProfileSummaryAnalysis, Module>());
  MPM.addPass(RequireAnalysisPass<CollectorMetadataAnalysis, Module>());

  invokeCodeGenIREarlyEPCallbacks(MPM);

  if (TM->useEmulatedTLS())
    MPM.addPass(LowerEmuTLSPass());
  MPM.addPass(PreISelIntrinsicLoweringPass(TM));

  // For MachO, lower @llvm.global_dtors into @llvm.global_ctors with
  // __cxa_atexit() calls to avoid emitting the deprecated __mod_term_func.
  if (TM->getTargetTriple().isOSBinFormatMachO() &&
      !CGPBO.DisableAtExitBasedGlobalDtorLowering)
    MPM.addPass(LowerGlobalDtorsPass());

  FPM.addPass(ExpandLargeDivRemPass(TM));
  FPM.addPass(ExpandLargeFpConvertPass(TM));

  // Run loop strength reduction before anything else.
  if (OptLevel != CodeGenOptLevel::None) {
    if (!CGPBO.DisableLSR) {
      LoopPassManager LPM;
      LPM.addPass(LoopStrengthReducePass());
      FPM.addPass(createFunctionToLoopPassAdaptor(LoopStrengthReducePass(),
                                                  /*UseMemorySSA=*/true));
    }
    // The MergeICmpsPass tries to create memcmp calls by grouping sequences of
    // loads and compares. ExpandMemCmpPass then tries to expand those calls
    // into optimally-sized loads and compares. The transforms are enabled by a
    // target lowering hook.
    if (!CGPBO.DisableMergeICmps)
      FPM.addPass(MergeICmpsPass());
    FPM.addPass(ExpandMemCmpPass(TM));
  }

  // Run GC lowering passes for builtin collectors
  FPM.addPass(GCLoweringPass());
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
  MPM.addPass(ShadowStackGCLoweringPass());
  FPM = FunctionPassManager();
  invokeGCLoweringEPCallbacks(FPM);

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

  // Expand reduction intrinsics into shuffle sequences if the target wants to.
  // Allow disabling it for testing purposes.
  if (!CGPBO.DisableExpandReductions)
    FPM.addPass(ExpandReductionsPass());

  if (OptLevel != CodeGenOptLevel::None) {
    FPM.addPass(TLSVariableHoistPass());

    // Convert conditional moves to conditional jumps when profitable.
    if (!CGPBO.DisableSelectOptimize)
      FPM.addPass(SelectOptimizePass(TM));
  }

  {
    ModulePassManager CGMPM;
    AddCodeGenPreparePassesCallback(CGMPM);
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
    MPM.addPass(std::move(CGMPM));
    FPM = FunctionPassManager();
  }

  // Turn exception handling constructs into something the code generators can
  // handle.
  if (auto Err = addExceptionHandlingPasses(FPM))
    return Err;

  // All passes after this point need to handle cgscc.

  { // Pre isel extension
    ModulePassManager ISelPreparePasses;
    invokeISelPrepareEPCallbacks(ISelPreparePasses);
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
    MPM.addPass(std::move(ISelPreparePasses));
  }

  if (OptLevel != CodeGenOptLevel::None)
    FPM.addPass(ObjCARCContractPass());
  FPM.addPass(CallBrPreparePass());
  // Add both the safe stack and the stack protection passes: each of them will
  // only protect functions that have corresponding attributes.
  FPM.addPass(SafeStackPass(TM));
  FPM.addPass(StackProtectorPass(TM));

  // All passes which modify the LLVM IR are now complete; run the verifier
  // to ensure that the IR is valid.
  if (!CGPBO.DisableVerify)
    FPM.addPass(VerifierPass());

  if (PrintMIR) {
    if (CGPBO.RequiresCodeGenSCCOrder)
      MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(
          createCGSCCToFunctionPassAdaptor(std::move(FPM))));
    else
      MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
    MPM.addPass(PrintMIRPreparePass(Out));
    FPM = FunctionPassManager();
  }

  if (auto Err = addInstructionSelectorPasses(MFPM))
    return Err;

  if (auto Err = addMachinePasses(MPM, FPM, MFPM))
    return Err;

  if (!CGPBO.DisableVerify)
    MFPM.addPass(MachineVerifierPass());

  if (PrintMIR)
    MFPM.addPass(PrintMIRPass(Out));

  // TODO: Add AsmPrinter.
  (void)Ctx;

  FPM.addPass(createFunctionToMachineFunctionPassAdaptor(std::move(MFPM)));
  FPM.addPass(InvalidateAnalysisPass<MachineFunctionAnalysis>());
  if (CGPBO.RequiresCodeGenSCCOrder)
    MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(
        createCGSCCToFunctionPassAdaptor(std::move(FPM))));
  else
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));

  MPM.eraseIf([&](StringRef Name) { return DisabledPasses.contains(Name); });
  if (auto Err = setStartStop(MPM, PIC))
    return Err;
  TopLevelMPM.addPass(std::move(MPM));
  return Error::success();
}

Expected<ModulePassManager> PassBuilder::buildDefaultCodeGenPipeline(
    raw_pwrite_stream &Out, raw_pwrite_stream *DwoOut, CodeGenFileType FileType,
    MCContext &Ctx) {
  ModulePassManager MPM;
  Error Err = buildDefaultCodeGenPipeline(MPM, Out, DwoOut, FileType, Ctx);
  if (Err)
    return std::move(Err);
  return std::move(MPM);
}

static bool isRegAllocPass(StringRef Name) {
  // TODO: Add all register allocator names.
  return Name.starts_with("regallocfast");
}

Error PassBuilder::parseRegAllocOption(StringRef Text) {
  if (Text == "default")
    return Error::success();

  RegAllocPasses.clear();
  while (!Text.empty()) {
    StringRef SinglePass;
    std::tie(SinglePass, Text) = Text.split(',');

    if (!isRegAllocPass(SinglePass)) {
      return make_error<StringError>(
          formatv("{0} is not a register allocator!", SinglePass).str(),
          std::make_error_code(std::errc::invalid_argument));
    }
    if (!isOptimizedRegAlloc() && !SinglePass.starts_with("regallocfast")) {
      return make_error<StringError>(
          "Must use fast (default) register allocator for unoptimized "
          "regalloc.",
          std::make_error_code(std::errc::invalid_argument));
    }

    MachineFunctionPassManager MFPM;
    if (auto Err = parsePassPipeline(MFPM, SinglePass))
      return Err;

    auto FilterPos = SinglePass.find("filter=");
    if (FilterPos == std::string::npos) {
      bool Success = RegAllocPasses.try_emplace("all", std::move(MFPM)).second;
      if (!Success) {
        return make_error<StringError>(
            formatv("Already set register allocator '{0}' for all registers!",
                    SinglePass)
                .str(),
            std::make_error_code(std::errc::invalid_argument));
      }
      continue;
    }

    StringRef FilterName = SinglePass.drop_front(FilterPos);
    FilterName.consume_front("filter=");
    FilterName =
        FilterName.take_until([](char C) { return C == ';' || C == '>'; });
    bool Success =
        RegAllocPasses.try_emplace(FilterName, std::move(MFPM)).second;
    if (!Success) {
      return make_error<StringError>(
          formatv("Already set register allocator '{0}' for filter {1}!",
                  SinglePass, FilterName)
              .str(),
          std::make_error_code(std::errc::invalid_argument));
    }
  }
  return Error::success();
}
