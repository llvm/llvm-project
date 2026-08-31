//===--- CodeGenPassBuilder.cpp --------------------------------------- ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines interfaces to access the target independent code
// generation passes provided by the LLVM backend.
//
//===----------------------------------------------------------------------===//

#include "llvm/Passes/CodeGenPassBuilder.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/RuntimeLibcallInfo.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/AsmPrinterAnalysis.h"
#include "llvm/CodeGen/BranchFoldingPass.h"
#include "llvm/CodeGen/CodeGenPrepare.h"
#include "llvm/CodeGen/DeadMachineInstructionElim.h"
#include "llvm/CodeGen/DetectDeadLanes.h"
#include "llvm/CodeGen/DwarfEHPrepare.h"
#include "llvm/CodeGen/ExpandIRInsts.h"
#include "llvm/CodeGen/ExpandPostRAPseudos.h"
#include "llvm/CodeGen/ExpandReductions.h"
#include "llvm/CodeGen/FEntryInserter.h"
#include "llvm/CodeGen/FinalizeISel.h"
#include "llvm/CodeGen/FixupStatepointCallerSaved.h"
#include "llvm/CodeGen/FuncletLayout.h"
#include "llvm/CodeGen/GCEmptyBasicBlocks.h"
#include "llvm/CodeGen/GCMetadata.h"
#include "llvm/CodeGen/GlobalMerge.h"
#include "llvm/CodeGen/GlobalMergeFunctions.h"
#include "llvm/CodeGen/ImplicitNullChecks.h"
#include "llvm/CodeGen/IndirectBrExpand.h"
#include "llvm/CodeGen/InitUndef.h"
#include "llvm/CodeGen/InlineAsmPrepare.h"
#include "llvm/CodeGen/InterleavedAccess.h"
#include "llvm/CodeGen/InterleavedLoadCombine.h"
#include "llvm/CodeGen/LiveDebugValuesPass.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/LocalStackSlotAllocation.h"
#include "llvm/CodeGen/LowerEmuTLS.h"
#include "llvm/CodeGen/MIRPrinter.h"
#include "llvm/CodeGen/MachineBlockPlacement.h"
#include "llvm/CodeGen/MachineCSE.h"
#include "llvm/CodeGen/MachineCopyPropagation.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/MachineLICM.h"
#include "llvm/CodeGen/MachineLateInstrsCleanup.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/MachineSink.h"
#include "llvm/CodeGen/MachineVerifier.h"
#include "llvm/CodeGen/OptimizePHIs.h"
#include "llvm/CodeGen/PEI.h"
#include "llvm/CodeGen/PHIElimination.h"
#include "llvm/CodeGen/PatchableFunction.h"
#include "llvm/CodeGen/PeepholeOptimizer.h"
#include "llvm/CodeGen/PostRAMachineSink.h"
#include "llvm/CodeGen/PostRASchedulerList.h"
#include "llvm/CodeGen/PreISelIntrinsicLowering.h"
#include "llvm/CodeGen/ProcessImplicitDefs.h"
#include "llvm/CodeGen/RegAllocEvictionAdvisor.h"
#include "llvm/CodeGen/RegAllocFast.h"
#include "llvm/CodeGen/RegAllocGreedyPass.h"
#include "llvm/CodeGen/RegUsageInfoCollector.h"
#include "llvm/CodeGen/RegUsageInfoPropagate.h"
#include "llvm/CodeGen/RegisterCoalescerPass.h"
#include "llvm/CodeGen/RegisterUsageInfo.h"
#include "llvm/CodeGen/RemoveLoadsIntoFakeUses.h"
#include "llvm/CodeGen/RemoveRedundantDebugValues.h"
#include "llvm/CodeGen/RenameIndependentSubregs.h"
#include "llvm/CodeGen/ReplaceWithVeclib.h"
#include "llvm/CodeGen/SafeStack.h"
#include "llvm/CodeGen/SanitizerBinaryMetadata.h"
#include "llvm/CodeGen/SelectOptimize.h"
#include "llvm/CodeGen/ShadowStackGCLowering.h"
#include "llvm/CodeGen/ShrinkWrap.h"
#include "llvm/CodeGen/SjLjEHPrepare.h"
#include "llvm/CodeGen/StackColoring.h"
#include "llvm/CodeGen/StackFrameLayoutAnalysisPass.h"
#include "llvm/CodeGen/StackProtector.h"
#include "llvm/CodeGen/StackSlotColoring.h"
#include "llvm/CodeGen/TailDuplication.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TwoAddressInstructionPass.h"
#include "llvm/CodeGen/UnreachableBlockElim.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/CodeGen/WasmEHPrepare.h"
#include "llvm/CodeGen/WinEHPrepare.h"
#include "llvm/CodeGen/XRayInstrumentation.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRPrinter/IRPrintingPasses.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/CGPassBuilderOption.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/ObjCARC.h"
#include "llvm/Transforms/Scalar/ConstantHoisting.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Scalar/LoopStrengthReduce.h"
#include "llvm/Transforms/Scalar/LoopTermFold.h"
#include "llvm/Transforms/Scalar/PartiallyInlineLibCalls.h"
#include "llvm/Transforms/Scalar/ScalarizeMaskedMemIntrin.h"
#include "llvm/Transforms/Utils/CanonicalizeFreezeInLoops.h"
#include "llvm/Transforms/Utils/EntryExitInstrumenter.h"
#include "llvm/Transforms/Utils/LowerInvoke.h"
#include <cassert>
#include <utility>

using namespace llvm;

namespace llvm {
#define DUMMY_MACHINE_FUNCTION_ANALYSIS(NAME, CREATE_PASS)                     \
  AnalysisKey PASS_NAME::Key;
#include "llvm/Passes/MachinePassRegistry.def"
} // namespace llvm

CodeGenPassBuilder::CodeGenPassBuilder(TargetMachine &TM,
                                       const CGPassBuilderOption &Opts,
                                       PassInstrumentationCallbacks *PIC)
    : TM(TM), Opt(Opts), PIC(PIC) {
  // Target could set CGPassBuilderOption::MISchedPostRA to true to achieve
  //     substitutePass(&PostRASchedulerID, &PostMachineSchedulerID)

  // Target should override TM.Options.EnableIPRA in their target-specific
  // LLVMTM ctor. See TargetMachine::setGlobalISel for example.
  if (Opt.EnableIPRA) {
    TM.Options.EnableIPRA = *Opt.EnableIPRA;
  } else {
    // If not explicitly specified, use target default.
    TM.Options.EnableIPRA |= TM.useIPRA();
  }

  if (Opt.EnableGlobalISelAbort)
    TM.Options.GlobalISelAbort = *Opt.EnableGlobalISelAbort;

  // An explicit RegAlloc choice implies its pipeline: only the fast
  // allocator uses the unoptimized one.
  if (Opt.OptimizeRegAlloc == cl::boolOrDefault::BOU_UNSET) {
    bool Optimized = Opt.RegAlloc > RegAllocType::Default
                         ? Opt.RegAlloc != RegAllocType::Fast
                         : getOptLevel() != CodeGenOptLevel::None;
    Opt.OptimizeRegAlloc =
        Optimized ? cl::boolOrDefault::BOU_TRUE : cl::boolOrDefault::BOU_FALSE;
  }
}

// Out-of-line to anchor the vtable in this translation unit.
CodeGenPassBuilder::~CodeGenPassBuilder() = default;

Error CodeGenPassBuilder::addInstSelector(PassManagerWrapper &PMW) {
  return make_error<StringError>("addInstSelector is not overridden",
                                 inconvertibleErrorCode());
}

Error CodeGenPassBuilder::addIRTranslator(PassManagerWrapper &PMW) {
  return make_error<StringError>("addIRTranslator is not overridden",
                                 inconvertibleErrorCode());
}

Error CodeGenPassBuilder::addLegalizeMachineIR(PassManagerWrapper &PMW) {
  return make_error<StringError>("addLegalizeMachineIR is not overridden",
                                 inconvertibleErrorCode());
}

Error CodeGenPassBuilder::addRegBankSelect(PassManagerWrapper &PMW) {
  return make_error<StringError>("addRegBankSelect is not overridden",
                                 inconvertibleErrorCode());
}

Error CodeGenPassBuilder::addGlobalInstructionSelect(PassManagerWrapper &PMW) {
  return make_error<StringError>("addGlobalInstructionSelect is not overridden",
                                 inconvertibleErrorCode());
}

void CodeGenPassBuilder::addAsmPrinterBegin(PassManagerWrapper &PMW) {
  llvm_unreachable("addAsmPrinterBegin is not overriden");
}

void CodeGenPassBuilder::addAsmPrinter(PassManagerWrapper &PMW) {
  llvm_unreachable("addAsmPrinter is not overridden");
}

void CodeGenPassBuilder::addAsmPrinterEnd(PassManagerWrapper &PMW) {
  llvm_unreachable("addAsmPrinterEnd is not overriden");
}

void CodeGenPassBuilder::flushFPMsToMPM(PassManagerWrapper &PMW,
                                        bool FreeMachineFunctions) {
  if (PMW.FPM.isEmpty() && PMW.MFPM.isEmpty())
    return;
  if (!PMW.MFPM.isEmpty()) {
    PMW.FPM.addPass(
        createFunctionToMachineFunctionPassAdaptor(std::move(PMW.MFPM)));
    PMW.MFPM = MachineFunctionPassManager();
  }
  if (FreeMachineFunctions)
    PMW.FPM.addPass(FreeMachineFunctionPass());
  if (AddInCGSCCOrder) {
    PMW.MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(
        createCGSCCToFunctionPassAdaptor(std::move(PMW.FPM))));
  } else {
    PMW.MPM.addPass(createModuleToFunctionPassAdaptor(std::move(PMW.FPM)));
  }
  PMW.FPM = FunctionPassManager();
}

Error CodeGenPassBuilder::buildPipeline(
    ModulePassManager &MPM, ModuleAnalysisManager &MAM, raw_pwrite_stream &Out,
    raw_pwrite_stream *DwoOut, CodeGenFileType FileType, MCContext &Ctx) {
  auto StartStopInfo = TargetPassConfig::getStartStopInfo(*PIC);
  if (!StartStopInfo)
    return StartStopInfo.takeError();
  setStartStopPasses(*StartStopInfo);

  bool PrintAsm = TargetPassConfig::willCompleteCodeGenPipeline();
  bool PrintMIR = !PrintAsm && FileType != CodeGenFileType::Null;

  PassManagerWrapper PMW(MPM);

  addModulePass(RequireAnalysisPass<MachineModuleAnalysis, Module>(), PMW,
                /*Force=*/true);
  addModulePass(RequireAnalysisPass<ProfileSummaryAnalysis, Module>(), PMW,
                /*Force=*/true);
  addModulePass(RequireAnalysisPass<CollectorMetadataAnalysis, Module>(), PMW,
                /*Force=*/true);
  addModulePass(RequireAnalysisPass<RuntimeLibraryAnalysis, Module>(), PMW,
                /*Force=*/true);
  addModulePass(RequireAnalysisPass<LibcallLoweringModuleAnalysis, Module>(),
                PMW,
                /*Force=*/true);
  addISelPasses(PMW);
  flushFPMsToMPM(PMW);

  if (PrintAsm) {
    Expected<std::unique_ptr<MCStreamer>> MCStreamerOrErr =
        TM.createMCStreamer(Out, DwoOut, FileType, Ctx);
    if (!MCStreamerOrErr)
      return MCStreamerOrErr.takeError();
    std::unique_ptr<AsmPrinter> Printer(
        TM.getTarget().createAsmPrinter(TM, std::move(*MCStreamerOrErr)));
    if (!Printer)
      return createStringError("failed to create AsmPrinter");
    MAM.registerPass([&] { return AsmPrinterAnalysis(std::move(Printer)); });
    addAsmPrinterBegin(PMW);
  }

  if (PrintMIR)
    addModulePass(PrintMIRPreparePass(Out), PMW, /*Force=*/true);

  if (auto Err = addCoreISelPasses(PMW))
    return Err;

  if (auto Err = addMachinePasses(PMW))
    return Err;

  if (!Opt.DisableVerify && TM.Options.EnableDefaultMachineVerifier)
    addMachineFunctionPass(MachineVerifierPass(), PMW);

  // We add AsmPrinter regardless if we are emitting MIR or Assembly as the
  // final output so that -stop-before=<target>-asm-printer works. When printing
  // MIR as the final output, we never end up running AsmPrinter.
  addAsmPrinter(PMW);

  if (PrintAsm) {
    flushFPMsToMPM(PMW, /*FreeMachineFunctions=*/true);
    addAsmPrinterEnd(PMW);
  } else {
    if (PrintMIR)
      addMachineFunctionPass(PrintMIRPass(Out), PMW, /*Force=*/true);
    flushFPMsToMPM(PMW, /*FreeMachineFunctions=*/true);
  }

  return verifyStartStop(*StartStopInfo);
}

void CodeGenPassBuilder::setStartStopPasses(
    const TargetPassConfig::StartStopInfo &Info) {
  if (!Info.StartPass.empty()) {
    Started = false;
    BeforeCallbacks.emplace_back([this, &Info, AfterFlag = Info.StartAfter,
                                  Count = 0u](StringRef ClassName) mutable {
      if (Count == Info.StartInstanceNum) {
        if (AfterFlag) {
          AfterFlag = false;
          Started = true;
        }
        return Started;
      }

      auto PassName = PIC->getPassNameForClassName(ClassName);
      if (Info.StartPass == PassName && ++Count == Info.StartInstanceNum)
        Started = !Info.StartAfter;

      return Started;
    });
  }

  if (!Info.StopPass.empty()) {
    Stopped = false;
    BeforeCallbacks.emplace_back([this, &Info, AfterFlag = Info.StopAfter,
                                  Count = 0u](StringRef ClassName) mutable {
      if (Count == Info.StopInstanceNum) {
        if (AfterFlag) {
          AfterFlag = false;
          Stopped = true;
        }
        return !Stopped;
      }

      auto PassName = PIC->getPassNameForClassName(ClassName);
      if (Info.StopPass == PassName && ++Count == Info.StopInstanceNum)
        Stopped = !Info.StopAfter;
      return !Stopped;
    });
  }
}

Error CodeGenPassBuilder::verifyStartStop(
    const TargetPassConfig::StartStopInfo &Info) const {
  if (Started && Stopped)
    return Error::success();

  if (!Started)
    return make_error<StringError>(
        "Can't find start pass \"" + Info.StartPass + "\".",
        std::make_error_code(std::errc::invalid_argument));
  if (!Stopped)
    return make_error<StringError>(
        "Can't find stop pass \"" + Info.StopPass + "\".",
        std::make_error_code(std::errc::invalid_argument));
  return Error::success();
}

void CodeGenPassBuilder::addISelPasses(PassManagerWrapper &PMW) {
  addGlobalMergePass(PMW);
  if (TM.useEmulatedTLS())
    addModulePass(LowerEmuTLSPass(), PMW);

  // ObjCARCContract operates on ObjC intrinsics and must run before
  // PreISelIntrinsicLowering.
  if (getOptLevel() != CodeGenOptLevel::None) {
    addFunctionPass(ObjCARCContractPass(), PMW);
    flushFPMsToMPM(PMW);
  }
  addModulePass(PreISelIntrinsicLoweringPass(&TM), PMW);
  addFunctionPass(ExpandIRInstsPass(TM, getOptLevel()), PMW);

  addIRPasses(PMW);
  addCodeGenPrepare(PMW);
  addPassesToHandleExceptions(PMW);
  addISelPrepare(PMW);
}

/// Add common target configurable passes that perform LLVM IR to IR transforms
/// following machine independent optimization.
void CodeGenPassBuilder::addIRPasses(PassManagerWrapper &PMW) {
  // Before running any passes, run the verifier to determine if the input
  // coming from the front-end and/or optimizer is valid.
  if (!Opt.DisableVerify)
    addFunctionPass(VerifierPass(), PMW, /*Force=*/true);

  // Run loop strength reduction before anything else.
  if (getOptLevel() != CodeGenOptLevel::None && !Opt.DisableLSR) {
    // These passes do not use MSSA.
    LoopPassManager LPM;
    LPM.addPass(CanonicalizeFreezeInLoopsPass());
    LPM.addPass(LoopStrengthReducePass());
    if (Opt.EnableLoopTermFold)
      LPM.addPass(LoopTermFoldPass());
    addFunctionPass(createFunctionToLoopPassAdaptor(std::move(LPM),
                                                    /*UseMemorySSA=*/false),
                    PMW);
  }

  // Run GC lowering passes for builtin collectors
  // TODO: add a pass insertion point here
  addFunctionPass(GCLoweringPass(), PMW);
  // Explicitly check to see if we should add ShadowStackGCLowering to avoid
  // splitting the function pipeline if we do not have to.
  if (runBeforeAdding(ShadowStackGCLoweringPass::name())) {
    flushFPMsToMPM(PMW);
    addModulePass(ShadowStackGCLoweringPass(), PMW);
  }

  // Make sure that no unreachable blocks are instruction selected.
  addFunctionPass(UnreachableBlockElimPass(), PMW);

  // Prepare expensive constants for SelectionDAG.
  if (getOptLevel() != CodeGenOptLevel::None && !Opt.DisableConstantHoisting)
    addFunctionPass(ConstantHoistingPass(), PMW);

  // Replace calls to LLVM intrinsics (e.g., exp, log) operating on vector
  // operands with calls to the corresponding functions in a vector library.
  if (getOptLevel() != CodeGenOptLevel::None)
    addFunctionPass(ReplaceWithVeclib(), PMW);

  if (getOptLevel() != CodeGenOptLevel::None &&
      !Opt.DisablePartialLibcallInlining)
    addFunctionPass(PartiallyInlineLibCallsPass(), PMW);

  // Instrument function entry and exit, e.g. with calls to mcount().
  addFunctionPass(EntryExitInstrumenterPass(/*PostInlining=*/true), PMW);

  // Add scalarization of target's unsupported masked memory intrinsics pass.
  // the unsupported intrinsic will be replaced with a chain of basic blocks,
  // that stores/loads element one-by-one if the appropriate mask bit is set.
  addFunctionPass(ScalarizeMaskedMemIntrinPass(), PMW);

  // Expand reduction intrinsics into shuffle sequences if the target wants to.
  if (!Opt.DisableExpandReductions)
    addFunctionPass(ExpandReductionsPass(), PMW);

  // Convert conditional moves to conditional jumps when profitable.
  if (getOptLevel() != CodeGenOptLevel::None && !Opt.DisableSelectOptimize)
    addFunctionPass(SelectOptimizePass(TM), PMW);

  if (Opt.EnableGlobalMergeFunc) {
    flushFPMsToMPM(PMW);
    addModulePass(GlobalMergeFuncPass(), PMW);
  }
}

/// Turn exception handling constructs into something the code generators can
/// handle.
void CodeGenPassBuilder::addPassesToHandleExceptions(PassManagerWrapper &PMW) {
  const MCAsmInfo &MCAI = TM.getMCAsmInfo();
  switch (MCAI.getExceptionHandlingType()) {
  case ExceptionHandling::SjLj:
    // SjLj piggy-backs on dwarf for this bit. The cleanups done apply to both
    // Dwarf EH prepare needs to be run after SjLj prepare. Otherwise,
    // catch info can get misplaced when a selector ends up more than one block
    // removed from the parent invoke(s). This could happen when a landing
    // pad is shared by multiple invokes and is also a target of a normal
    // edge from elsewhere.
    addFunctionPass(SjLjEHPreparePass(&TM), PMW);
    [[fallthrough]];
  case ExceptionHandling::DwarfCFI:
  case ExceptionHandling::ARM:
  case ExceptionHandling::AIX:
  case ExceptionHandling::ZOS:
    addFunctionPass(DwarfEHPreparePass(TM), PMW);
    break;
  case ExceptionHandling::WinEH:
    // We support using both GCC-style and MSVC-style exceptions on Windows, so
    // add both preparation passes. Each pass will only actually run if it
    // recognizes the personality function.
    addFunctionPass(WinEHPreparePass(), PMW);
    addFunctionPass(DwarfEHPreparePass(TM), PMW);
    break;
  case ExceptionHandling::Wasm:
    // Wasm EH uses Windows EH instructions, but it does not need to demote PHIs
    // on catchpads and cleanuppads because it does not outline them into
    // funclets. Catchswitch blocks are not lowered in SelectionDAG, so we
    // should remove PHIs there.
    addFunctionPass(WinEHPreparePass(/*DemoteCatchSwitchPHIOnly=*/false), PMW);
    addFunctionPass(WasmEHPreparePass(), PMW);
    break;
  case ExceptionHandling::None:
    addFunctionPass(LowerInvokePass(), PMW);

    // The lower invoke pass may create unreachable code. Remove it.
    addFunctionPass(UnreachableBlockElimPass(), PMW);
    break;
  }
}

/// Add pass to prepare the LLVM IR for code generation. This should be done
/// before exception handling preparation passes.
void CodeGenPassBuilder::addCodeGenPrepare(PassManagerWrapper &PMW) {
  if (getOptLevel() != CodeGenOptLevel::None && !Opt.DisableCGP)
    addFunctionPass(CodeGenPreparePass(TM), PMW);
  // TODO: Default ctor'd RewriteSymbolPass is no-op.
  // addPass(RewriteSymbolPass());
}

/// Add common passes that perform LLVM IR to IR transforms in preparation for
/// instruction selection.
void CodeGenPassBuilder::addISelPrepare(PassManagerWrapper &PMW) {
  addPreISel(PMW);

  if (Opt.RequiresCodeGenSCCOrder && !AddInCGSCCOrder)
    requireCGSCCOrder(PMW);

  addFunctionPass(InlineAsmPreparePass(), PMW);
  // Add both the safe stack and the stack protection passes: each of them will
  // only protect functions that have corresponding attributes.
  addFunctionPass(SafeStackPass(TM), PMW);
  addFunctionPass(StackProtectorPass(TM), PMW);

  if (Opt.PrintISelInput)
    addFunctionPass(PrintFunctionPass(
                        dbgs(), "\n\n*** Final LLVM Code input to ISel ***\n"),
                    PMW);

  // All passes which modify the LLVM IR are now complete; run the verifier
  // to ensure that the IR is valid.
  if (!Opt.DisableVerify)
    addFunctionPass(VerifierPass(), PMW, /*Force=*/true);
}

Error CodeGenPassBuilder::addCoreISelPasses(PassManagerWrapper &PMW) {
  // Enable FastISel with -fast-isel, but allow that to be overridden.
  TM.setO0WantsFastISel(Opt.EnableFastISelOption !=
                        cl::boolOrDefault::BOU_FALSE);

  // Determine an instruction selector.
  enum class SelectorType { SelectionDAG, FastISel, GlobalISel };
  SelectorType Selector;

  if (Opt.EnableFastISelOption == cl::boolOrDefault::BOU_TRUE)
    Selector = SelectorType::FastISel;
  else if (Opt.EnableGlobalISelOption == cl::boolOrDefault::BOU_TRUE ||
           (TM.Options.EnableGlobalISel &&
            Opt.EnableGlobalISelOption != cl::boolOrDefault::BOU_FALSE))
    Selector = SelectorType::GlobalISel;
  else if (TM.getOptLevel() == CodeGenOptLevel::None && TM.getO0WantsFastISel())
    Selector = SelectorType::FastISel;
  else
    Selector = SelectorType::SelectionDAG;

  // Set consistently TM.Options.EnableFastISel and EnableGlobalISel.
  if (Selector == SelectorType::FastISel) {
    TM.setFastISel(true);
    TM.setGlobalISel(false);
  } else if (Selector == SelectorType::GlobalISel) {
    TM.setFastISel(false);
    TM.setGlobalISel(true);
  }

  // Add instruction selector passes.
  if (Selector == SelectorType::GlobalISel) {
    if (auto Err = addIRTranslator(PMW))
      return Err;

    addPreLegalizeMachineIR(PMW);

    if (auto Err = addLegalizeMachineIR(PMW))
      return Err;

    // Before running the register bank selector, ask the target if it
    // wants to run some passes.
    addPreRegBankSelect(PMW);

    if (auto Err = addRegBankSelect(PMW))
      return Err;

    addPreGlobalInstructionSelect(PMW);

    if (auto Err = addGlobalInstructionSelect(PMW))
      return Err;

    // Pass to reset the MachineFunction if the ISel failed.
    addMachineFunctionPass(
        ResetMachineFunctionPass(reportDiagnosticWhenGlobalISelFallback(),
                                 isGlobalISelAbortEnabled()),
        PMW);

    // Provide a fallback path when we do not want to abort on
    // not-yet-supported input.
    if (!isGlobalISelAbortEnabled())
      if (auto Err = addInstSelector(PMW))
        return Err;

  } else if (auto Err = addInstSelector(PMW))
    return Err;

  // Expand pseudo-instructions emitted by ISel. Don't run the verifier before
  // FinalizeISel.
  addMachineFunctionPass(FinalizeISelPass(), PMW);

  // // Print the instruction selected machine code...
  // printAndVerify("After Instruction Selection");

  return Error::success();
}

/// Add the complete set of target-independent postISel code generator passes.
///
/// This can be read as the standard order of major LLVM CodeGen stages. Stages
/// with nontrivial configuration or multiple passes are broken out below in
/// add%Stage routines.
///
/// Any CodeGenPassBuilder::addXX routine may be overriden by the Target. The
/// addPre/Post methods with empty header implementations allow injecting
/// target-specific fixups just before or after major stages. Additionally,
/// targets have the flexibility to change pass order within a stage by
/// overriding default implementation of add%Stage routines below. Each
/// technique has maintainability tradeoffs because alternate pass orders are
/// not well supported. addPre/Post works better if the target pass is easily
/// tied to a common pass. But if it has subtle dependencies on multiple passes,
/// the target should override the stage instead.
Error CodeGenPassBuilder::addMachinePasses(PassManagerWrapper &PMW) {
  // Add passes that optimize machine instructions in SSA form.
  if (getOptLevel() != CodeGenOptLevel::None) {
    addMachineSSAOptimization(PMW);
  } else {
    // If the target requests it, assign local variables to stack slots relative
    // to one another and simplify frame index references where possible.
    addMachineFunctionPass(LocalStackSlotAllocationPass(), PMW);
  }

  if (TM.Options.EnableIPRA) {
    flushFPMsToMPM(PMW);
    addModulePass(RequireAnalysisPass<PhysicalRegisterUsageAnalysis, Module>(),
                  PMW, /*Force=*/true);
    addMachineFunctionPass(RegUsageInfoPropagationPass(), PMW);
  }
  // Run pre-ra passes.
  addPreRegAlloc(PMW);

  // Run register allocation and passes that are tightly coupled with it,
  // including phi elimination and scheduling.
  if (auto Err = Opt.OptimizeRegAlloc == cl::boolOrDefault::BOU_TRUE
                     ? addOptimizedRegAlloc(PMW)
                     : addFastRegAlloc(PMW))
    return Err;

  // Run post-ra passes.
  addPostRegAlloc(PMW);

  addMachineFunctionPass(RemoveRedundantDebugValuesPass(), PMW);
  addMachineFunctionPass(FixupStatepointCallerSavedPass(), PMW);

  // Insert prolog/epilog code.  Eliminate abstract frame index references...
  if (getOptLevel() != CodeGenOptLevel::None) {
    addMachineFunctionPass(PostRAMachineSinkingPass(), PMW);
    addMachineFunctionPass(ShrinkWrapPass(), PMW);
  }

  addMachineFunctionPass(PrologEpilogInserterPass(), PMW);

  /// Add passes that optimize machine instructions after register allocation.
  if (getOptLevel() != CodeGenOptLevel::None)
    addMachineLateOptimization(PMW);

  // Expand pseudo instructions before second scheduling pass.
  addMachineFunctionPass(ExpandPostRAPseudosPass(), PMW);

  // Run pre-sched2 passes.
  addPreSched2(PMW);

  if (Opt.EnableImplicitNullChecks)
    addMachineFunctionPass(ImplicitNullChecksPass(), PMW);

  // Second pass scheduler.
  // Let Target optionally insert this pass by itself at some other
  // point.
  if (getOptLevel() != CodeGenOptLevel::None &&
      !TM.targetSchedulesPostRAScheduling()) {
    if (Opt.MISchedPostRA)
      addMachineFunctionPass(PostMachineSchedulerPass(&TM), PMW);
    else
      addMachineFunctionPass(PostRASchedulerPass(&TM), PMW);
  }

  // GC
  addGCPasses(PMW);

  // Basic block placement.
  if (getOptLevel() != CodeGenOptLevel::None)
    addBlockPlacement(PMW);

  // Insert before XRay Instrumentation.
  addMachineFunctionPass(FEntryInserterPass(), PMW);

  addMachineFunctionPass(XRayInstrumentationPass(), PMW);
  addMachineFunctionPass(PatchableFunctionPass(), PMW);

  addPreEmitPass(PMW);

  if (TM.Options.EnableIPRA) {
    // Collect register usage information and produce a register mask of
    // clobbered registers, to be used to optimize call sites.
    addMachineFunctionPass(RegUsageInfoCollectorPass(), PMW);
    // If -print-regusage is specified, print the collected register usage info.
    if (Opt.PrintRegUsage) {
      flushFPMsToMPM(PMW);
      addModulePass(PhysicalRegisterUsageInfoPrinterPass(errs()), PMW);
    }
  }

  addMachineFunctionPass(FuncletLayoutPass(), PMW);

  addMachineFunctionPass(RemoveLoadsIntoFakeUsesPass(), PMW);
  addMachineFunctionPass(StackMapLivenessPass(), PMW);
  addMachineFunctionPass(
      LiveDebugValuesPass(TM.Options.ShouldEmitDebugEntryValues()), PMW);
  addMachineFunctionPass(MachineSanitizerBinaryMetadataPass(), PMW);

  if (TM.Options.EnableMachineOutliner &&
      getOptLevel() != CodeGenOptLevel::None &&
      Opt.EnableMachineOutliner != RunOutliner::NeverOutline) {
    if (Opt.EnableMachineOutliner != RunOutliner::TargetDefault ||
        TM.Options.SupportsDefaultOutlining) {
      flushFPMsToMPM(PMW);
      addModulePass(MachineOutlinerPass(Opt.EnableMachineOutliner), PMW);
    }
  }

  if (Opt.EnableGCEmptyBlocks)
    addMachineFunctionPass(GCEmptyBasicBlocksPass(), PMW);

  addPostBBSections(PMW);

  addMachineFunctionPass(StackFrameLayoutAnalysisPass(), PMW);

  // Add passes that directly emit MI after all other MI passes.
  addPreEmitPass2(PMW);

  return Error::success();
}

/// Add passes that optimize machine instructions in SSA form.
void CodeGenPassBuilder::addMachineSSAOptimization(PassManagerWrapper &PMW) {
  // Pre-ra tail duplication.
  addMachineFunctionPass(EarlyTailDuplicatePass(), PMW);

  // Optimize PHIs before DCE: removing dead PHI cycles may make more
  // instructions dead.
  addMachineFunctionPass(OptimizePHIsPass(), PMW);

  // This pass merges large allocas. StackSlotColoring is a different pass
  // which merges spill slots.
  addMachineFunctionPass(StackColoringPass(), PMW);

  // If the target requests it, assign local variables to stack slots relative
  // to one another and simplify frame index references where possible.
  addMachineFunctionPass(LocalStackSlotAllocationPass(), PMW);

  // With optimization, dead code should already be eliminated. However
  // there is one known exception: lowered code for arguments that are only
  // used by tail calls, where the tail calls reuse the incoming stack
  // arguments directly (see t11 in test/CodeGen/X86/sibcall.ll).
  addMachineFunctionPass(DeadMachineInstructionElimPass(), PMW);

  // Allow targets to insert passes that improve instruction level parallelism,
  // like if-conversion. Such passes will typically need dominator trees and
  // loop info, just like LICM and CSE below.
  addILPOpts(PMW);

  addMachineFunctionPass(EarlyMachineLICMPass(), PMW);
  addMachineFunctionPass(MachineCSEPass(), PMW);

  addMachineFunctionPass(MachineSinkingPass(Opt.EnableSinkAndFold), PMW);

  addMachineFunctionPass(PeepholeOptimizerPass(), PMW);
  // Clean-up the dead code that may have been generated by peephole
  // rewriting.
  addMachineFunctionPass(DeadMachineInstructionElimPass(), PMW);
}

//===---------------------------------------------------------------------===//
/// Register Allocation Pass Configuration
//===---------------------------------------------------------------------===//

/// Instantiate the default register allocator pass for this target for either
/// the optimized or unoptimized allocation path. This will be added to the pass
/// manager by addFastRegAlloc in the unoptimized case or addOptimizedRegAlloc
/// in the optimized case.
///
/// A target that uses the standard regalloc pass order for fast or optimized
/// allocation may still override this for per-target regalloc
/// selection. But -regalloc-npm=... always takes precedence.
/// If a target does not want to allow users to set -regalloc-npm=... at all,
/// check if Opt.RegAlloc == RegAllocType::Unset.
void CodeGenPassBuilder::addTargetRegisterAllocator(PassManagerWrapper &PMW,
                                                    bool Optimized) {
  if (Optimized)
    addMachineFunctionPass(RAGreedyPass(), PMW);
  else
    addMachineFunctionPass(RegAllocFastPass(), PMW);
}

/// Find and instantiate the register allocation pass requested by this target
/// at the current optimization level.  Different register allocators are
/// defined as separate passes because they may require different analysis.
///
/// This helper ensures that the -regalloc-npm= option is always available,
/// even for targets that override the default allocator.
void CodeGenPassBuilder::addRegAllocPass(PassManagerWrapper &PMW,
                                         bool Optimized) {
  // Use the specified -regalloc-npm={basic|greedy|fast|pbqp}
  if (Opt.RegAlloc > RegAllocType::Default) {
    switch (Opt.RegAlloc) {
    case RegAllocType::Fast:
      addMachineFunctionPass(RegAllocFastPass(), PMW);
      break;
    case RegAllocType::Greedy:
      addMachineFunctionPass(RAGreedyPass(), PMW);
      break;
    default:
      reportFatalUsageError("register allocator not supported yet");
    }
    return;
  }
  // -regalloc=default or unspecified, so pick based on the optimization level
  // or ask the target for the regalloc pass.
  addTargetRegisterAllocator(PMW, Optimized);
}

Error CodeGenPassBuilder::addRegAssignAndRewriteFast(PassManagerWrapper &PMW) {
  // TODO: Ensure allocator is default or fast.
  addRegAllocPass(PMW, false);
  return Error::success();
}

Expected<bool>
CodeGenPassBuilder::addRegAssignAndRewriteOptimized(PassManagerWrapper &PMW) {
  // Add the selected register allocation pass.
  addRegAllocPass(PMW, true);

  // Allow targets to change the register assignments before rewriting.
  addPreRewrite(PMW);

  // Finally rewrite virtual registers.
  addMachineFunctionPass(VirtRegRewriterPass(), PMW);

  return true;
}

/// Add the minimum set of target-independent passes that are required for
/// register allocation. No coalescing or scheduling.
Error CodeGenPassBuilder::addFastRegAlloc(PassManagerWrapper &PMW) {
  addMachineFunctionPass(PHIEliminationPass(), PMW);
  addMachineFunctionPass(TwoAddressInstructionPass(), PMW);
  return addRegAssignAndRewriteFast(PMW);
}

/// Add standard target-independent passes that are tightly coupled with
/// optimized register allocation, including coalescing, machine instruction
/// scheduling, and register allocation itself.
Error CodeGenPassBuilder::addOptimizedRegAlloc(PassManagerWrapper &PMW) {
  addMachineFunctionPass(DetectDeadLanesPass(), PMW);

  addMachineFunctionPass(InitUndefPass(), PMW);

  addMachineFunctionPass(ProcessImplicitDefsPass(), PMW);

  // LiveVariables currently requires pure SSA form.
  //
  // FIXME: Once TwoAddressInstruction pass no longer uses kill flags,
  // LiveVariables can be removed completely, and LiveIntervals can be directly
  // computed. (We still either need to regenerate kill flags after regalloc, or
  // preferably fix the scavenger to not depend on them).
  // FIXME: UnreachableMachineBlockElim is a dependant pass of LiveVariables.
  // When LiveVariables is removed this has to be removed/moved either.
  // Explicit addition of UnreachableMachineBlockElim allows stopping before or
  // after it with -stop-before/-stop-after.
  addMachineFunctionPass(UnreachableMachineBlockElimPass(), PMW);
  addMachineFunctionPass(
      RequireAnalysisPass<LiveVariablesAnalysis, MachineFunction>(), PMW);

  // Edge splitting is smarter with machine loop info.
  addMachineFunctionPass(
      RequireAnalysisPass<MachineLoopAnalysis, MachineFunction>(), PMW);
  addMachineFunctionPass(PHIEliminationPass(), PMW);

  // Eventually, we want to run LiveIntervals before PHI elimination.
  if (Opt.EarlyLiveIntervals)
    addMachineFunctionPass(
        RequireAnalysisPass<LiveIntervalsAnalysis, MachineFunction>(), PMW);

  addMachineFunctionPass(TwoAddressInstructionPass(), PMW);
  addMachineFunctionPass(RegisterCoalescerPass(), PMW);

  // The machine scheduler may accidentally create disconnected components
  // when moving subregister definitions around, avoid this by splitting them to
  // separate vregs before. Splitting can also improve reg. allocation quality.
  addMachineFunctionPass(RenameIndependentSubregsPass(), PMW);

  // PreRA instruction scheduling.
  addMachineFunctionPass(MachineSchedulerPass(&TM), PMW);

  Expected<bool> AddedPasses = addRegAssignAndRewriteOptimized(PMW);
  if (!AddedPasses)
    return AddedPasses.takeError();
  if (!AddedPasses.get())
    return Error::success();

  addMachineFunctionPass(StackSlotColoringPass(), PMW);

  // Allow targets to expand pseudo instructions depending on the choice of
  // registers before MachineCopyPropagation.
  addPostRewrite(PMW);

  // Copy propagate to forward register uses and try to eliminate COPYs that
  // were not coalesced.
  addMachineFunctionPass(MachineCopyPropagationPass(), PMW);

  // Run post-ra machine LICM to hoist reloads / remats.
  //
  // FIXME: can this move into MachineLateOptimization?
  addMachineFunctionPass(MachineLICMPass(), PMW);

  return Error::success();
}

//===---------------------------------------------------------------------===//
/// Post RegAlloc Pass Configuration
//===---------------------------------------------------------------------===//

/// Add passes that optimize machine instructions after register allocation.
void CodeGenPassBuilder::addMachineLateOptimization(PassManagerWrapper &PMW) {
  // Cleanup of redundant (identical) address/immediate loads.
  addMachineFunctionPass(MachineLateInstrsCleanupPass(), PMW);

  // Branch folding must be run after regalloc and prolog/epilog insertion.
  addMachineFunctionPass(BranchFolderPass(Opt.EnableTailMerge), PMW);

  // Tail duplication.
  // Note that duplicating tail just increases code size and degrades
  // performance for targets that require Structured Control Flow.
  // In addition it can also make CFG irreducible. Thus we disable it.
  if (!TM.requiresStructuredCFG())
    addMachineFunctionPass(TailDuplicatePass(), PMW);

  // Copy propagation.
  addMachineFunctionPass(MachineCopyPropagationPass(), PMW);
}

/// Add standard basic block placement passes.
void CodeGenPassBuilder::addBlockPlacement(PassManagerWrapper &PMW) {
  addMachineFunctionPass(MachineBlockPlacementPass(Opt.EnableTailMerge), PMW);
  // Run a separate pass to collect block placement statistics.
  if (Opt.EnableBlockPlacementStats)
    addMachineFunctionPass(MachineBlockPlacementStatsPass(), PMW);
}
