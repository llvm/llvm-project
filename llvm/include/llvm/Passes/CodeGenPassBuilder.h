//===- Construction of codegen pass pipelines ------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// Interfaces for producing common pass manager configurations.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_CODEGENPASSBUILDER_H
#define LLVM_PASSES_CODEGENPASSBUILDER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/RuntimeLibcallInfo.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
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
#include "llvm/CodeGen/GCEmptyBasicBlocks.h"
#include "llvm/CodeGen/GCMetadata.h"
#include "llvm/CodeGen/GlobalMerge.h"
#include "llvm/CodeGen/GlobalMergeFunctions.h"
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
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/CodeGenPassManager.h"
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
#include "llvm/Transforms/Scalar/LowerConstantIntrinsics.h"
#include "llvm/Transforms/Scalar/MergeICmps.h"
#include "llvm/Transforms/Scalar/PartiallyInlineLibCalls.h"
#include "llvm/Transforms/Scalar/ScalarizeMaskedMemIntrin.h"
#include "llvm/Transforms/Utils/CanonicalizeFreezeInLoops.h"
#include "llvm/Transforms/Utils/EntryExitInstrumenter.h"
#include "llvm/Transforms/Utils/LowerInvoke.h"
#include <cassert>
#include <utility>

namespace llvm {

// FIXME: Dummy target independent passes definitions that have not yet been
// ported to new pass manager. Once they do, remove these.
#define DUMMY_FUNCTION_PASS(NAME, PASS_NAME)                                   \
  struct PASS_NAME : public OptionalPassInfoMixin<PASS_NAME> {                 \
    template <typename... Ts> PASS_NAME(Ts &&...) {}                           \
    PreservedAnalyses run(Function &, FunctionAnalysisManager &) {             \
      return PreservedAnalyses::all();                                         \
    }                                                                          \
  };
#define DUMMY_MACHINE_MODULE_PASS(NAME, PASS_NAME)                             \
  struct PASS_NAME : public OptionalPassInfoMixin<PASS_NAME> {                 \
    template <typename... Ts> PASS_NAME(Ts &&...) {}                           \
    PreservedAnalyses run(Module &, ModuleAnalysisManager &) {                 \
      return PreservedAnalyses::all();                                         \
    }                                                                          \
  };
#define DUMMY_MACHINE_FUNCTION_PASS(NAME, PASS_NAME)                           \
  struct PASS_NAME : public OptionalPassInfoMixin<PASS_NAME> {                 \
    template <typename... Ts> PASS_NAME(Ts &&...) {}                           \
    PreservedAnalyses run(MachineFunction &,                                   \
                          MachineFunctionAnalysisManager &) {                  \
      return PreservedAnalyses::all();                                         \
    }                                                                          \
  };
#include "llvm/Passes/MachinePassRegistry.def"

/// This class provides access to building LLVM's passes.
///
/// Its members provide the baseline state available to passes during their
/// construction. The \c MachinePassRegistry.def file specifies how to construct
/// all of the built-in passes, and those may reference these members during
/// construction.
template <typename DerivedT, typename TargetMachineT> class CodeGenPassBuilder {
public:
  explicit CodeGenPassBuilder(TargetMachineT &TM,
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

    if (Opt.OptimizeRegAlloc == cl::boolOrDefault::BOU_UNSET)
      Opt.OptimizeRegAlloc = getOptLevel() != CodeGenOptLevel::None
                                 ? cl::boolOrDefault::BOU_TRUE
                                 : cl::boolOrDefault::BOU_FALSE;
  }

  Error buildPipeline(ModulePassManager &MPM, ModuleAnalysisManager &MAM,
                      raw_pwrite_stream &Out, raw_pwrite_stream *DwoOut,
                      CodeGenFileType FileType, MCContext &Ctx) const;

  PassInstrumentationCallbacks *getPassInstrumentationCallbacks() const {
    return PIC;
  }

protected:
  TargetMachineT &TM;
  CGPassBuilderOption Opt;
  PassInstrumentationCallbacks *PIC;

  template <typename TMC> TMC &getTM() const { return static_cast<TMC &>(TM); }
  CodeGenOptLevel getOptLevel() const { return TM.getOptLevel(); }

  /// Check whether or not GlobalISel should abort on error.
  /// When this is disabled, GlobalISel will fall back on SDISel instead of
  /// erroring out.
  bool isGlobalISelAbortEnabled() const {
    return TM.Options.GlobalISelAbort == GlobalISelAbortMode::Enable;
  }

  /// Check whether or not a diagnostic should be emitted when GlobalISel
  /// uses the fallback path. In other words, it will emit a diagnostic
  /// when GlobalISel failed and isGlobalISelAbortEnabled is false.
  bool reportDiagnosticWhenGlobalISelFallback() const {
    return TM.Options.GlobalISelAbort == GlobalISelAbortMode::DisableWithDiag;
  }

  /// addInstSelector - This method should install an instruction selector pass,
  /// which converts from LLVM code to machine instructions.
  Error addInstSelector(CodeGenMachineFunctionPassManager &CGMFPM) const {
    return make_error<StringError>("addInstSelector is not overridden",
                                   inconvertibleErrorCode());
  }

  /// Target can override this to add GlobalMergePass before all IR passes.
  void addGlobalMergePass(CodeGenModulePassManager &CGMPM) const {}

  /// Add passes that optimize instruction level parallelism for out-of-order
  /// targets. These passes are run while the machine code is still in SSA
  /// form, so they can use MachineTraceMetrics to control their heuristics.
  ///
  /// All passes added here should preserve the MachineDominatorTree,
  /// MachineLoopInfo, and MachineTraceMetrics analyses.
  void addILPOpts(CodeGenMachineFunctionPassManager &CGMFPM) const {}

  /// This method may be implemented by targets that want to run passes
  /// immediately before register allocation.
  void addPreRegAlloc(CodeGenMachineFunctionPassManager &CGMFPM) const {}

  /// addPreRewrite - Add passes to the optimized register allocation pipeline
  /// after register allocation is complete, but before virtual registers are
  /// rewritten to physical registers.
  ///
  /// These passes must preserve VirtRegMap and LiveIntervals, and when running
  /// after RABasic or RAGreedy, they should take advantage of LiveRegMatrix.
  /// When these passes run, VirtRegMap contains legal physreg assignments for
  /// all virtual registers.
  ///
  /// Note if the target overloads addRegAssignAndRewriteOptimized, this may not
  /// be honored. This is also not generally used for the fast variant,
  /// where the allocation and rewriting are done in one pass.
  void addPreRewrite(CodeGenMachineFunctionPassManager &CGMFPM) const {}

  /// Add passes to be run immediately after virtual registers are rewritten
  /// to physical registers.
  void addPostRewrite(CodeGenMachineFunctionPassManager &CGMFPM) const {}

  /// This method may be implemented by targets that want to run passes after
  /// register allocation pass pipeline but before prolog-epilog insertion.
  void addPostRegAlloc(CodeGenMachineFunctionPassManager &CGMFPM) const {}

  /// This method may be implemented by targets that want to run passes after
  /// prolog-epilog insertion and before the second instruction scheduling pass.
  void addPreSched2(CodeGenMachineFunctionPassManager &CGMFPM) const {}

  /// This pass may be implemented by targets that want to run passes
  /// immediately before machine code is emitted.
  void addPreEmitPass(CodeGenMachineFunctionPassManager &CGMFPM) const {}

  /// Targets may add passes immediately before machine code is emitted in this
  /// callback. This is called even later than `addPreEmitPass`.
  // FIXME: Rename `addPreEmitPass` to something more sensible given its actual
  // position and remove the `2` suffix here as this callback is what
  // `addPreEmitPass` *should* be but in reality isn't.
  void addPreEmitPass2(CodeGenMachineFunctionPassManager &CGMFPM) const {}

  /// {{@ For GlobalISel
  ///

  /// addPreISel - This method should add any "last minute" LLVM->LLVM
  /// passes (which are run just before instruction selector).
  CodeGenFunctionPassManager addPreISel() const {
    return CodeGenFunctionPassManager();
  }

  /// This method should install an IR translator pass, which converts from
  /// LLVM code to machine instructions with possibly generic opcodes.
  Error addIRTranslator(CodeGenMachineFunctionPassManager &CGMFPM) const {
    return make_error<StringError>("addIRTranslator is not overridden",
                                   inconvertibleErrorCode());
  }

  /// This method may be implemented by targets that want to run passes
  /// immediately before legalization.
  void
  addPreLegalizeMachineIR(CodeGenMachineFunctionPassManager &CGMFPM) const {}

  /// This method should install a legalize pass, which converts the instruction
  /// sequence into one that can be selected by the target.
  Error addLegalizeMachineIR(CodeGenMachineFunctionPassManager &CGMFPM) const {
    return make_error<StringError>("addLegalizeMachineIR is not overridden",
                                   inconvertibleErrorCode());
  }

  /// This method may be implemented by targets that want to run passes
  /// immediately before the register bank selection.
  void addPreRegBankSelect(CodeGenMachineFunctionPassManager &CGMFPM) const {}

  /// This method should install a register bank selector pass, which
  /// assigns register banks to virtual registers without a register
  /// class or register banks.
  Error addRegBankSelect(CodeGenMachineFunctionPassManager &CGMFPM) const {
    return make_error<StringError>("addRegBankSelect is not overridden",
                                   inconvertibleErrorCode());
  }

  /// This method may be implemented by targets that want to run passes
  /// immediately before the (global) instruction selection.
  void addPreGlobalInstructionSelect(
      CodeGenMachineFunctionPassManager &CGMFPM) const {}

  /// This method should install a (global) instruction selector pass, which
  /// converts possibly generic instructions to fully target-specific
  /// instructions, thereby constraining all generic virtual registers to
  /// register classes.
  Error
  addGlobalInstructionSelect(CodeGenMachineFunctionPassManager &CGMFPM) const {
    return make_error<StringError>(
        "addGlobalInstructionSelect is not overridden",
        inconvertibleErrorCode());
  }
  /// @}}

  /// High level function that adds all passes necessary to go from llvm IR
  /// representation to the MI representation.
  /// Adds IR based lowering and target specific optimization passes and finally
  /// the core instruction selection passes.
  void addISelPasses(CodeGenModulePassManager &CGMPM) const;

  /// Add the actual instruction selection passes. This does not include
  /// preparation passes on IR.
  Error addCoreISelPasses(CodeGenMachineFunctionPassManager &CGMFPM) const;

  /// Add the complete, standard set of LLVM CodeGen passes.
  /// Fully developed targets will not generally override this.
  Error addMachinePasses(CodeGenModulePassManager &CGMPM) const;

  /// Add passes to lower exception handling for the code generator.
  void addPassesToHandleExceptions(CodeGenFunctionPassManager &CGFPM) const;

  /// Add common target configurable passes that perform LLVM IR to IR
  /// transforms following machine independent optimization.
  void addIRPasses(CodeGenModulePassManager &CGMPM) const;

  /// Add pass to prepare the LLVM IR for code generation. This should be done
  /// before exception handling preparation passes.
  CodeGenFunctionPassManager addCodeGenPrepare() const;

  /// Add common passes that perform LLVM IR to IR transforms in preparation for
  /// instruction selection.
  void addISelPrepare(CodeGenModulePassManager &CGMPM) const;

  /// Methods with trivial inline returns are convenient points in the common
  /// codegen pass pipeline where targets may insert passes. Methods with
  /// out-of-line standard implementations are major CodeGen stages called by
  /// addMachinePasses. Some targets may override major stages when inserting
  /// passes is insufficient, but maintaining overriden stages is more work.
  ///

  /// addMachineSSAOptimization - Add standard passes that optimize machine
  /// instructions in SSA form.
  void
  addMachineSSAOptimization(CodeGenMachineFunctionPassManager &CGMFPM) const;

  /// addFastRegAlloc - Add the minimum set of target-independent passes that
  /// are required for fast register allocation.
  Error addFastRegAlloc(CodeGenMachineFunctionPassManager &CGMFPM) const;

  /// addOptimizedRegAlloc - Add passes related to register allocation.
  /// CodeGenTargetMachineImpl provides standard regalloc passes for most
  /// targets.
  Error addOptimizedRegAlloc(CodeGenMachineFunctionPassManager &CGMFPM) const;

  /// Add passes that optimize machine instructions after register allocation.
  void
  addMachineLateOptimization(CodeGenMachineFunctionPassManager &CGMFPM) const;

  /// addGCPasses - Add late codegen passes that analyze code for garbage
  /// collection. This should return true if GC info should be printed after
  /// these passes.
  void addGCPasses(CodeGenMachineFunctionPassManager &CGMFPM) const {}

  /// Add standard basic block placement passes.
  void addBlockPlacement(CodeGenMachineFunctionPassManager &CGMFPM) const;

  void addPostBBSections(CodeGenMachineFunctionPassManager &CGMFPM) const {}

  void addAsmPrinterBegin(CodeGenModulePassManager &CGMPM) const {
    llvm_unreachable("addAsmPrinterBegin is not overriden");
  }

  void addAsmPrinter(CodeGenMachineFunctionPassManager &CGMFPM) const {
    llvm_unreachable("addAsmPrinter is not overridden");
  }

  void addAsmPrinterEnd(CodeGenModulePassManager &CGMPM) const {
    llvm_unreachable("addAsmPrinterEnd is not overriden");
  }

  /// Utilities for targets to add passes to the pass manager.
  ///

  /// createTargetRegisterAllocator - Create the register allocator pass for
  /// this target at the current optimization level.
  void addTargetRegisterAllocator(CodeGenMachineFunctionPassManager &CGMFPM,
                                  bool Optimized) const;

  /// addMachinePasses helper to create the target-selected or overriden
  /// regalloc pass.
  void addRegAllocPass(CodeGenMachineFunctionPassManager &CGMFPM,
                       bool Optimized) const;

  /// Add core register allocator passes which do the actual register assignment
  /// and rewriting.
  Error addRegAssignmentFast(CodeGenMachineFunctionPassManager &CGMFPM) const;
  Error
  addRegAssignmentOptimized(CodeGenMachineFunctionPassManager &CGMFPM) const;

  /// Allow the target to disable a specific pass by default.
  /// Backend can declare unwanted passes in constructor.
  template <typename... PassTs> void disablePass() {
    (DisabledPasses.insert(PassTs::name()), ...);
  }

  /// Insert InsertedPass pass after TargetPass pass.
  /// Only machine function passes are supported.
  template <typename TargetPassT, typename InsertedPassT>
  void insertPass(InsertedPassT &&Pass) const {
    InsertPassCallbacks.emplace_back(
        [&, Visited = false](
            StringRef Name, CodeGenMachineFunctionPassManager &CGMFPM) mutable {
          if (Visited)
            return;
          if (Name == TargetPassT::name()) {
            Visited = true;
            CGMFPM.addPass(std::forward<InsertedPassT>(Pass));
          }
        });
  }

private:
  DerivedT &derived() { return static_cast<DerivedT &>(*this); }
  const DerivedT &derived() const {
    return static_cast<const DerivedT &>(*this);
  }

  void setStartStopPasses(const TargetPassConfig::StartStopInfo &Info) const;

  Error verifyStartStop(const TargetPassConfig::StartStopInfo &Info) const;

  StringSet<> DisabledPasses;
  mutable SmallVector<llvm::unique_function<void(
                          StringRef, CodeGenMachineFunctionPassManager &)>,
                      4>
      InsertPassCallbacks;

  /// Helper variable for `-start-before/-start-after/-stop-before/-stop-after`
  mutable bool Started = true;
  mutable bool Stopped = true;
};

template <typename Derived, typename TargetMachineT>
Error CodeGenPassBuilder<Derived, TargetMachineT>::buildPipeline(
    ModulePassManager &MPM, ModuleAnalysisManager &MAM, raw_pwrite_stream &Out,
    raw_pwrite_stream *DwoOut, CodeGenFileType FileType, MCContext &Ctx) const {
  auto StartStopInfo = TargetPassConfig::getStartStopInfo(*PIC);
  if (!StartStopInfo)
    return StartStopInfo.takeError();
  setStartStopPasses(*StartStopInfo);

  bool PrintAsm = TargetPassConfig::willCompleteCodeGenPipeline();
  bool PrintMIR = !PrintAsm && FileType != CodeGenFileType::Null;

  CodeGenModulePassManager CGMPM;
  CGMPM.addPass(RequireAnalysisPass<MachineModuleAnalysis, Module>());
  CGMPM.addPass(RequireAnalysisPass<ProfileSummaryAnalysis, Module>());
  CGMPM.addPass(RequireAnalysisPass<CollectorMetadataAnalysis, Module>());
  CGMPM.addPass(RequireAnalysisPass<RuntimeLibraryAnalysis, Module>());
  CGMPM.addPass(RequireAnalysisPass<LibcallLoweringModuleAnalysis, Module>());

  addISelPasses(CGMPM);

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
    derived().addAsmPrinterBegin(CGMPM);
  }

  if (PrintMIR)
    CGMPM.addPass(PrintMIRPreparePass(Out));

  CodeGenMachineFunctionPassManager CGMFPM;
  if (auto Err = addCoreISelPasses(CGMFPM))
    return std::move(Err);
  CodeGenFunctionPassManager CGFPM;
  CGFPM.addCodeGenMachineFunctionPassManager(std::move(CGMFPM));
  if (Opt.RequiresCodeGenSCCOrder)
    CGMPM.addCodeGenFunctionPassManagerInCGSCCOrder(std::move(CGFPM));
  else
    CGMPM.addCodeGenFunctionPassManager(std::move(CGFPM));

  if (auto Err = derived().addMachinePasses(CGMPM))
    return std::move(Err);

  if (!Opt.DisableVerify && TM.Options.EnableDefaultMachineVerifier)
    CGMFPM.addPass(MachineVerifierPass());

  if (PrintAsm) {
    derived().addAsmPrinter(CGMFPM);
    CGFPM.addCodeGenMachineFunctionPassManager(std::move(CGMFPM));
    CGFPM.addPass(FreeMachineFunctionPass());
    CGMPM.addCodeGenFunctionPassManager(std::move(CGFPM));
    derived().addAsmPrinterEnd(CGMPM);
  } else if (PrintMIR) {
    CGMFPM.addPass(PrintMIRPass(Out));
    CGFPM.addCodeGenMachineFunctionPassManager(std::move(CGMFPM));
    CGFPM.addPass(FreeMachineFunctionPass());
    CGMPM.addCodeGenFunctionPassManager(std::move(CGFPM));
  } else {
    CGFPM.addPass(FreeMachineFunctionPass());
    CGMPM.addCodeGenFunctionPassManager(std::move(CGFPM));
  }

  // Build the final pass manager.
  CGMPM.handleInsert(
      [this](StringRef Name, CodeGenMachineFunctionPassManager &CGMFPM) {
        for (auto &CB : InsertPassCallbacks)
          CB(Name, CGMFPM);
      });
  CGMPM.eraseIf(
      [this](StringRef Name) { return DisabledPasses.contains(Name); });

  StringSet<> NotSkippable = {
      FreeMachineFunctionPass::name(),
      PrintMIRPass::name(),
      PrintMIRPreparePass::name(),
      RequireAnalysisPass<MachineModuleAnalysis, Module>::name(),
      RequireAnalysisPass<ProfileSummaryAnalysis, Module>::name(),
      RequireAnalysisPass<RuntimeLibraryAnalysis, Module>::name(),
      RequireAnalysisPass<CollectorMetadataAnalysis, Module>::name(),
      RequireAnalysisPass<LibcallLoweringModuleAnalysis, Module>::name(),
      RequireAnalysisPass<PhysicalRegisterUsageAnalysis, Module>::name(),
  };
  CGMPM.eraseIf([this, &StartStopInfo, &NotSkippable, StartCount = 0ul,
                 StopCount = 0ul](StringRef ClassName) mutable {
    StringRef PassName = PIC->getPassNameForClassName(ClassName);
    if (!Started) {
      if (StartCount == StartStopInfo->StartInstanceNum)
        Started = true;
      if (PassName == StartStopInfo->StartPass) {
        ++StartCount;
        if (StartCount == StartStopInfo->StartInstanceNum) {
          Started = !StartStopInfo->StartAfter;
        }
      }
    }
    if (!Stopped && !StartStopInfo->StopPass.empty()) {
      if (StopCount == StartStopInfo->StopInstanceNum)
        Stopped = true;
      if (PassName == StartStopInfo->StopPass) {
        ++StopCount;
        if (StopCount == StartStopInfo->StopInstanceNum) {
          Stopped = !StartStopInfo->StopAfter;
        }
      }
    }

    if (ClassName.contains("AsmPrinter") || PassName == "verify" ||
        NotSkippable.contains(ClassName))
      return false;

    return !Started || Stopped;
  });

  CGMPM.combineSimilarPassManagers();

  MPM = CGMPM.getModulePassManager();

  return verifyStartStop(*StartStopInfo);
}

template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::setStartStopPasses(
    const TargetPassConfig::StartStopInfo &Info) const {
  Started = Info.StartPass.empty();
  Stopped = false;
}

template <typename Derived, typename TargetMachineT>
Error CodeGenPassBuilder<Derived, TargetMachineT>::verifyStartStop(
    const TargetPassConfig::StartStopInfo &Info) const {
  if (Started && Stopped)
    return Error::success();

  if (!Started)
    return make_error<StringError>(
        "Can't find start pass \"" + Info.StartPass + "\".",
        std::make_error_code(std::errc::invalid_argument));
  if (!Stopped && !Info.StopPass.empty())
    return make_error<StringError>(
        "Can't find stop pass \"" + Info.StopPass + "\".",
        std::make_error_code(std::errc::invalid_argument));
  return Error::success();
}

template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addISelPasses(
    CodeGenModulePassManager &CGMPM) const {
  derived().addGlobalMergePass(CGMPM);
  if (TM.useEmulatedTLS())
    CGMPM.addPass(LowerEmuTLSPass());

  // ObjCARCContract operates on ObjC intrinsics and must run before
  // PreISelIntrinsicLowering.
  if (getOptLevel() != CodeGenOptLevel::None) {
    CodeGenFunctionPassManager CGFPM;
    CGFPM.addPass(ObjCARCContractPass());
    CGMPM.addCodeGenFunctionPassManager(std::move(CGFPM));
  }
  CGMPM.addPass(PreISelIntrinsicLoweringPass(&TM));
  CodeGenFunctionPassManager CGFPM;
  CGFPM.addPass(ExpandIRInstsPass(TM, getOptLevel()));
  CGMPM.addCodeGenFunctionPassManager(std::move(CGFPM));

  derived().addIRPasses(CGMPM);
  if constexpr (std::is_same_v<decltype(derived().addCodeGenPrepare()),
                               CodeGenFunctionPassManager>)
    CGMPM.addCodeGenFunctionPassManager(derived().addCodeGenPrepare());
  else
    CGMPM.addPass(derived().addCodeGenPrepare());
  addPassesToHandleExceptions(CGFPM);
  if (Opt.RequiresCodeGenSCCOrder)
    CGMPM.addCodeGenFunctionPassManagerInCGSCCOrder(std::move(CGFPM));
  else
    CGMPM.addCodeGenFunctionPassManager(std::move(CGFPM));
  derived().addISelPrepare(CGMPM);
}

/// Add common target configurable passes that perform LLVM IR to IR transforms
/// following machine independent optimization.
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addIRPasses(
    CodeGenModulePassManager &CGMPM) const {
  CodeGenFunctionPassManager CGFPM;
  // Before running any passes, run the verifier to determine if the input
  // coming from the front-end and/or optimizer is valid.
  if (!Opt.DisableVerify)
    CGFPM.addPass(VerifierPass());

  // Run loop strength reduction before anything else.
  if (getOptLevel() != CodeGenOptLevel::None && !Opt.DisableLSR) {
    // These passes do not use MSSA.
    CodeGenLoopPassManager CGLPM;
    CGLPM.addPass(CanonicalizeFreezeInLoopsPass());
    CGLPM.addPass(LoopStrengthReducePass());
    if (Opt.EnableLoopTermFold)
      CGLPM.addPass(LoopTermFoldPass());
    CGFPM.addCodeGenLoopPassManager(std::move(CGLPM));
  }

  // Run GC lowering passes for builtin collectors
  // TODO: add a pass insertion point here
  CGFPM.addPass(GCLoweringPass());
  CGMPM.addCodeGenFunctionPassManager(std::move(CGFPM));
  CGMPM.addPass(ShadowStackGCLoweringPass());

  // Make sure that no unreachable blocks are instruction selected.
  CGFPM.addPass(UnreachableBlockElimPass());

  // Prepare expensive constants for SelectionDAG.
  if (getOptLevel() != CodeGenOptLevel::None && !Opt.DisableConstantHoisting)
    CGFPM.addPass(ConstantHoistingPass());

  // Replace calls to LLVM intrinsics (e.g., exp, log) operating on vector
  // operands with calls to the corresponding functions in a vector library.
  if (getOptLevel() != CodeGenOptLevel::None)
    CGFPM.addPass(ReplaceWithVeclib());

  if (getOptLevel() != CodeGenOptLevel::None &&
      !Opt.DisablePartialLibcallInlining)
    CGFPM.addPass(PartiallyInlineLibCallsPass());

  // Instrument function entry and exit, e.g. with calls to mcount().
  CGFPM.addPass(EntryExitInstrumenterPass(/*PostInlining=*/true));

  // Add scalarization of target's unsupported masked memory intrinsics pass.
  // the unsupported intrinsic will be replaced with a chain of basic blocks,
  // that stores/loads element one-by-one if the appropriate mask bit is set.
  CGFPM.addPass(ScalarizeMaskedMemIntrinPass());

  // Expand reduction intrinsics into shuffle sequences if the target wants to.
  if (!Opt.DisableExpandReductions)
    CGFPM.addPass(ExpandReductionsPass());

  // Convert conditional moves to conditional jumps when profitable.
  if (getOptLevel() != CodeGenOptLevel::None && !Opt.DisableSelectOptimize)
    CGFPM.addPass(SelectOptimizePass(TM));

  if (Opt.EnableGlobalMergeFunc) {
    CGMPM.addCodeGenFunctionPassManager(std::move(CGFPM));
    CGMPM.addPass(GlobalMergeFuncPass());
  }
  CGMPM.addCodeGenFunctionPassManager(std::move(CGFPM));
}

/// Turn exception handling constructs into something the code generators can
/// handle.
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addPassesToHandleExceptions(
    CodeGenFunctionPassManager &CGFPM) const {
  const MCAsmInfo &MCAI = TM.getMCAsmInfo();
  switch (MCAI.getExceptionHandlingType()) {
  case ExceptionHandling::SjLj:
    // SjLj piggy-backs on dwarf for this bit. The cleanups done apply to both
    // Dwarf EH prepare needs to be run after SjLj prepare. Otherwise,
    // catch info can get misplaced when a selector ends up more than one block
    // removed from the parent invoke(s). This could happen when a landing
    // pad is shared by multiple invokes and is also a target of a normal
    // edge from elsewhere.
    CGFPM.addPass(SjLjEHPreparePass(&TM));
    [[fallthrough]];
  case ExceptionHandling::DwarfCFI:
  case ExceptionHandling::ARM:
  case ExceptionHandling::AIX:
  case ExceptionHandling::ZOS:
    CGFPM.addPass(DwarfEHPreparePass(TM));
    break;
  case ExceptionHandling::WinEH:
    // We support using both GCC-style and MSVC-style exceptions on Windows, so
    // add both preparation passes. Each pass will only actually run if it
    // recognizes the personality function.
    CGFPM.addPass(WinEHPreparePass());
    CGFPM.addPass(DwarfEHPreparePass(TM));
    break;
  case ExceptionHandling::Wasm:
    // Wasm EH uses Windows EH instructions, but it does not need to demote PHIs
    // on catchpads and cleanuppads because it does not outline them into
    // funclets. Catchswitch blocks are not lowered in SelectionDAG, so we
    // should remove PHIs there.
    CGFPM.addPass(WinEHPreparePass(/*DemoteCatchSwitchPHIOnly=*/false));
    CGFPM.addPass(WasmEHPreparePass());
    break;
  case ExceptionHandling::None:
    CGFPM.addPass(LowerInvokePass());

    // The lower invoke pass may create unreachable code. Remove it.
    CGFPM.addPass(UnreachableBlockElimPass());
    break;
  }
}

/// Add pass to prepare the LLVM IR for code generation. This should be done
/// before exception handling preparation passes.
template <typename Derived, typename TargetMachineT>
CodeGenFunctionPassManager
CodeGenPassBuilder<Derived, TargetMachineT>::addCodeGenPrepare() const {
  CodeGenFunctionPassManager CGFPM;
  if (getOptLevel() != CodeGenOptLevel::None && !Opt.DisableCGP)
    CGFPM.addPass(CodeGenPreparePass(TM));
  // TODO: Default ctor'd RewriteSymbolPass is no-op.
  // addPass(RewriteSymbolPass());
  return CGFPM;
}

/// Add common passes that perform LLVM IR to IR transforms in preparation for
/// instruction selection.
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addISelPrepare(
    CodeGenModulePassManager &CGMPM) const {
  CodeGenFunctionPassManager CGFPM;
  if constexpr (std::is_same_v<decltype(derived().addPreISel()),
                               CodeGenModulePassManager>) {
    CGMPM.addPass(derived().addPreISel());
  } else {
    if (Opt.RequiresCodeGenSCCOrder)
      CGMPM.addCodeGenFunctionPassManagerInCGSCCOrder(derived().addPreISel());
    else
      CGMPM.addCodeGenFunctionPassManager(derived().addPreISel());
  }

  CGFPM.addPass(InlineAsmPreparePass());
  // Add both the safe stack and the stack protection passes: each of them will
  // only protect functions that have corresponding attributes.
  CGFPM.addPass(SafeStackPass(TM));
  CGFPM.addPass(StackProtectorPass(TM));

  if (Opt.PrintISelInput)
    CGFPM.addPass(PrintFunctionPass(
        dbgs(), "\n\n*** Final LLVM Code input to ISel ***\n"));

  // All passes which modify the LLVM IR are now complete; run the verifier
  // to ensure that the IR is valid.
  if (!Opt.DisableVerify)
    CGFPM.addPass(VerifierPass());

  if (Opt.RequiresCodeGenSCCOrder)
    CGMPM.addCodeGenFunctionPassManagerInCGSCCOrder(std::move(CGFPM));
  else
    CGMPM.addCodeGenFunctionPassManager(std::move(CGFPM));
}

template <typename Derived, typename TargetMachineT>
Error CodeGenPassBuilder<Derived, TargetMachineT>::addCoreISelPasses(
    CodeGenMachineFunctionPassManager &CGMFPM) const {
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
    if (auto Err = derived().addIRTranslator(CGMFPM))
      return std::move(Err);

    derived().addPreLegalizeMachineIR(CGMFPM);

    if (auto Err = derived().addLegalizeMachineIR(CGMFPM))
      return std::move(Err);

    // Before running the register bank selector, ask the target if it
    // wants to run some passes.
    derived().addPreRegBankSelect(CGMFPM);

    if (auto Err = derived().addRegBankSelect(CGMFPM))
      return std::move(Err);

    derived().addPreGlobalInstructionSelect(CGMFPM);

    if (auto Err = derived().addGlobalInstructionSelect(CGMFPM))
      return std::move(Err);

    // Pass to reset the MachineFunction if the ISel failed.
    CGMFPM.addPass(ResetMachineFunctionPass(
        reportDiagnosticWhenGlobalISelFallback(), isGlobalISelAbortEnabled()));

    // Provide a fallback path when we do not want to abort on
    // not-yet-supported input.
    if (!isGlobalISelAbortEnabled())
      if (auto Err = derived().addInstSelector(CGMFPM))
        return std::move(Err);

  } else if (auto Err = derived().addInstSelector(CGMFPM))
    return std::move(Err);

  // Expand pseudo-instructions emitted by ISel. Don't run the verifier before
  // FinalizeISel.
  CGMFPM.addPass(FinalizeISelPass());

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
/// Any CodeGenPassBuilder<Derived, TargetMachine>::addXX routine may be
/// overriden by the Target. The addPre/Post methods with empty header
/// implementations allow injecting target-specific fixups just before or after
/// major stages. Additionally, targets have the flexibility to change pass
/// order within a stage by overriding default implementation of add%Stage
/// routines below. Each technique has maintainability tradeoffs because
/// alternate pass orders are not well supported. addPre/Post works better if
/// the target pass is easily tied to a common pass. But if it has subtle
/// dependencies on multiple passes, the target should override the stage
/// instead.
template <typename Derived, typename TargetMachineT>
Error CodeGenPassBuilder<Derived, TargetMachineT>::addMachinePasses(
    CodeGenModulePassManager &CGMPM) const {
  CodeGenMachineFunctionPassManager CGMFPM;
  // Add passes that optimize machine instructions in SSA form.
  if (getOptLevel() != CodeGenOptLevel::None) {
    derived().addMachineSSAOptimization(CGMFPM);
  } else {
    // If the target requests it, assign local variables to stack slots relative
    // to one another and simplify frame index references where possible.
    CGMFPM.addPass(LocalStackSlotAllocationPass());
  }

  if (TM.Options.EnableIPRA) {
    CodeGenFunctionPassManager CGFPM;
    CGFPM.addCodeGenMachineFunctionPassManager(std::move(CGMFPM));
    if (Opt.RequiresCodeGenSCCOrder)
      CGMPM.addCodeGenFunctionPassManagerInCGSCCOrder(std::move(CGFPM));
    else
      CGMPM.addCodeGenFunctionPassManager(std::move(CGFPM));
    CGMPM.addPass(RequireAnalysisPass<PhysicalRegisterUsageAnalysis, Module>());
    CGMFPM.addPass(RegUsageInfoPropagationPass());
  }
  // Run pre-ra passes.
  derived().addPreRegAlloc(CGMFPM);

  // Run register allocation and passes that are tightly coupled with it,
  // including phi elimination and scheduling.
  if (auto Err = Opt.OptimizeRegAlloc == cl::boolOrDefault::BOU_TRUE
                     ? derived().addOptimizedRegAlloc(CGMFPM)
                     : derived().addFastRegAlloc(CGMFPM))
    return std::move(Err);

  // Run post-ra passes.
  derived().addPostRegAlloc(CGMFPM);

  CGMFPM.addPass(RemoveRedundantDebugValuesPass());
  CGMFPM.addPass(FixupStatepointCallerSavedPass());

  // Insert prolog/epilog code.  Eliminate abstract frame index references...
  if (getOptLevel() != CodeGenOptLevel::None) {
    CGMFPM.addPass(PostRAMachineSinkingPass());
    CGMFPM.addPass(ShrinkWrapPass());
  }

  CGMFPM.addPass(PrologEpilogInserterPass());

  /// Add passes that optimize machine instructions after register allocation.
  if (getOptLevel() != CodeGenOptLevel::None)
    derived().addMachineLateOptimization(CGMFPM);

  // Expand pseudo instructions before second scheduling pass.
  CGMFPM.addPass(ExpandPostRAPseudosPass());

  // Run pre-sched2 passes.
  derived().addPreSched2(CGMFPM);

  if (Opt.EnableImplicitNullChecks)
    CGMFPM.addPass(ImplicitNullChecksPass());

  // Second pass scheduler.
  // Let Target optionally insert this pass by itself at some other
  // point.
  if (getOptLevel() != CodeGenOptLevel::None &&
      !TM.targetSchedulesPostRAScheduling()) {
    if (Opt.MISchedPostRA)
      CGMFPM.addPass(PostMachineSchedulerPass(&TM));
    else
      CGMFPM.addPass(PostRASchedulerPass(&TM));
  }

  // GC
  derived().addGCPasses(CGMFPM);

  // Basic block placement.
  if (getOptLevel() != CodeGenOptLevel::None)
    derived().addBlockPlacement(CGMFPM);

  // Insert before XRay Instrumentation.
  CGMFPM.addPass(FEntryInserterPass());

  CGMFPM.addPass(XRayInstrumentationPass());
  CGMFPM.addPass(PatchableFunctionPass());

  derived().addPreEmitPass(CGMFPM);

  if (TM.Options.EnableIPRA) {
    // Collect register usage information and produce a register mask of
    // clobbered registers, to be used to optimize call sites.
    CGMFPM.addPass(RegUsageInfoCollectorPass());
    // If -print-regusage is specified, print the collected register usage info.
    if (Opt.PrintRegUsage) {
      CodeGenFunctionPassManager CGFPM;
      CGFPM.addCodeGenMachineFunctionPassManager(std::move(CGMFPM));
      CGMPM.addCodeGenFunctionPassManager(std::move(CGFPM));
      CGMPM.addPass(PhysicalRegisterUsageInfoPrinterPass(errs()));
    }
  }

  CGMFPM.addPass(FuncletLayoutPass());

  CGMFPM.addPass(RemoveLoadsIntoFakeUsesPass());
  CGMFPM.addPass(StackMapLivenessPass());
  CGMFPM.addPass(LiveDebugValuesPass(
      getTM<TargetMachine>().Options.ShouldEmitDebugEntryValues()));
  CGMFPM.addPass(MachineSanitizerBinaryMetadataPass());

  if (TM.Options.EnableMachineOutliner &&
      getOptLevel() != CodeGenOptLevel::None &&
      Opt.EnableMachineOutliner != RunOutliner::NeverOutline) {
    if (Opt.EnableMachineOutliner != RunOutliner::TargetDefault ||
        TM.Options.SupportsDefaultOutlining) {
      CodeGenFunctionPassManager CGFPM;
      CGFPM.addCodeGenMachineFunctionPassManager(std::move(CGMFPM));
      if (Opt.RequiresCodeGenSCCOrder)
        CGMPM.addCodeGenFunctionPassManagerInCGSCCOrder(std::move(CGFPM));
      else
        CGMPM.addCodeGenFunctionPassManager(std::move(CGFPM));
      CGMPM.addPass(MachineOutlinerPass(Opt.EnableMachineOutliner));
    }
  }

  if (Opt.EnableGCEmptyBlocks)
    CGMFPM.addPass(GCEmptyBasicBlocksPass());

  derived().addPostBBSections(CGMFPM);

  CGMFPM.addPass(StackFrameLayoutAnalysisPass());

  // Add passes that directly emit MI after all other MI passes.
  derived().addPreEmitPass2(CGMFPM);

  CodeGenFunctionPassManager CGFPM;
  CGFPM.addCodeGenMachineFunctionPassManager(std::move(CGMFPM));
  if (Opt.RequiresCodeGenSCCOrder)
    CGMPM.addCodeGenFunctionPassManagerInCGSCCOrder(std::move(CGFPM));
  else
    CGMPM.addCodeGenFunctionPassManager(std::move(CGFPM));

  return Error::success();
}

/// Add passes that optimize machine instructions in SSA form.
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addMachineSSAOptimization(
    CodeGenMachineFunctionPassManager &CGMFPM) const {
  // Pre-ra tail duplication.
  CGMFPM.addPass(EarlyTailDuplicatePass());

  // Optimize PHIs before DCE: removing dead PHI cycles may make more
  // instructions dead.
  CGMFPM.addPass(OptimizePHIsPass());

  // This pass merges large allocas. StackSlotColoring is a different pass
  // which merges spill slots.
  CGMFPM.addPass(StackColoringPass());

  // If the target requests it, assign local variables to stack slots relative
  // to one another and simplify frame index references where possible.
  CGMFPM.addPass(LocalStackSlotAllocationPass());

  // With optimization, dead code should already be eliminated. However
  // there is one known exception: lowered code for arguments that are only
  // used by tail calls, where the tail calls reuse the incoming stack
  // arguments directly (see t11 in test/CodeGen/X86/sibcall.ll).
  CGMFPM.addPass(DeadMachineInstructionElimPass());

  // Allow targets to insert passes that improve instruction level parallelism,
  // like if-conversion. Such passes will typically need dominator trees and
  // loop info, just like LICM and CSE below.
  derived().addILPOpts(CGMFPM);

  CGMFPM.addPass(EarlyMachineLICMPass());
  CGMFPM.addPass(MachineCSEPass());

  CGMFPM.addPass(MachineSinkingPass(Opt.EnableSinkAndFold));

  CGMFPM.addPass(PeepholeOptimizerPass());
  // Clean-up the dead code that may have been generated by peephole
  // rewriting.
  CGMFPM.addPass(DeadMachineInstructionElimPass());
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
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addTargetRegisterAllocator(
    CodeGenMachineFunctionPassManager &CGMFPM, bool Optimized) const {
  if (Optimized)
    CGMFPM.addPass(RAGreedyPass());
  else
    CGMFPM.addPass(RegAllocFastPass());
}

/// Find and instantiate the register allocation pass requested by this target
/// at the current optimization level.  Different register allocators are
/// defined as separate passes because they may require different analysis.
///
/// This helper ensures that the -regalloc-npm= option is always available,
/// even for targets that override the default allocator.
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addRegAllocPass(
    CodeGenMachineFunctionPassManager &CGMFPM, bool Optimized) const {
  // Use the specified -regalloc-npm={basic|greedy|fast|pbqp}
  if (Opt.RegAlloc > RegAllocType::Default) {
    switch (Opt.RegAlloc) {
    case RegAllocType::Fast:
      CGMFPM.addPass(RegAllocFastPass());
      break;
    case RegAllocType::Greedy:
      CGMFPM.addPass(RAGreedyPass());
      break;
    default:
      reportFatalUsageError("register allocator not supported yet");
    }
    return;
  }
  // -regalloc=default or unspecified, so pick based on the optimization level
  // or ask the target for the regalloc pass.
  derived().addTargetRegisterAllocator(CGMFPM, Optimized);
}

template <typename Derived, typename TargetMachineT>
Error CodeGenPassBuilder<Derived, TargetMachineT>::addRegAssignmentFast(
    CodeGenMachineFunctionPassManager &CGMFPM) const {
  // TODO: Ensure allocator is default or fast.
  addRegAllocPass(CGMFPM, false);
  return Error::success();
}

template <typename Derived, typename TargetMachineT>
Error CodeGenPassBuilder<Derived, TargetMachineT>::addRegAssignmentOptimized(
    CodeGenMachineFunctionPassManager &CGMFPM) const {
  // Add the selected register allocation pass.
  addRegAllocPass(CGMFPM, true);

  // Allow targets to change the register assignments before rewriting.
  derived().addPreRewrite(CGMFPM);

  // Finally rewrite virtual registers.
  CGMFPM.addPass(VirtRegRewriterPass());
  // Perform stack slot coloring and post-ra machine LICM.
  //
  // FIXME: Re-enable coloring with register when it's capable of adding
  // kill markers.
  CGMFPM.addPass(StackSlotColoringPass());

  return Error::success();
}

/// Add the minimum set of target-independent passes that are required for
/// register allocation. No coalescing or scheduling.
template <typename Derived, typename TargetMachineT>
Error CodeGenPassBuilder<Derived, TargetMachineT>::addFastRegAlloc(
    CodeGenMachineFunctionPassManager &CGMFPM) const {
  CGMFPM.addPass(PHIEliminationPass());
  CGMFPM.addPass(TwoAddressInstructionPass());
  return derived().addRegAssignmentFast(CGMFPM);
}

/// Add standard target-independent passes that are tightly coupled with
/// optimized register allocation, including coalescing, machine instruction
/// scheduling, and register allocation itself.
template <typename Derived, typename TargetMachineT>
Error CodeGenPassBuilder<Derived, TargetMachineT>::addOptimizedRegAlloc(
    CodeGenMachineFunctionPassManager &CGMFPM) const {
  CGMFPM.addPass(DetectDeadLanesPass());

  CGMFPM.addPass(InitUndefPass());

  CGMFPM.addPass(ProcessImplicitDefsPass());

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
  CGMFPM.addPass(UnreachableMachineBlockElimPass());
  CGMFPM.addPass(RequireAnalysisPass<LiveVariablesAnalysis, MachineFunction>());

  // Edge splitting is smarter with machine loop info.
  CGMFPM.addPass(RequireAnalysisPass<MachineLoopAnalysis, MachineFunction>());
  CGMFPM.addPass(PHIEliminationPass());

  // Eventually, we want to run LiveIntervals before PHI elimination.
  if (Opt.EarlyLiveIntervals)
    CGMFPM.addPass(
        RequireAnalysisPass<LiveIntervalsAnalysis, MachineFunction>());

  CGMFPM.addPass(TwoAddressInstructionPass());
  CGMFPM.addPass(RegisterCoalescerPass());

  // The machine scheduler may accidentally create disconnected components
  // when moving subregister definitions around, avoid this by splitting them to
  // separate vregs before. Splitting can also improve reg. allocation quality.
  CGMFPM.addPass(RenameIndependentSubregsPass());

  // PreRA instruction scheduling.
  CGMFPM.addPass(MachineSchedulerPass(&TM));

  if (auto E = derived().addRegAssignmentOptimized(CGMFPM))
    return std::move(E);

  CGMFPM.addPass(StackSlotColoringPass());

  // Allow targets to expand pseudo instructions depending on the choice of
  // registers before MachineCopyPropagation.
  derived().addPostRewrite(CGMFPM);

  // Copy propagate to forward register uses and try to eliminate COPYs that
  // were not coalesced.
  CGMFPM.addPass(MachineCopyPropagationPass());

  // Run post-ra machine LICM to hoist reloads / remats.
  //
  // FIXME: can this move into MachineLateOptimization?
  CGMFPM.addPass(MachineLICMPass());

  return Error::success();
}

//===---------------------------------------------------------------------===//
/// Post RegAlloc Pass Configuration
//===---------------------------------------------------------------------===//

/// Add passes that optimize machine instructions after register allocation.
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addMachineLateOptimization(
    CodeGenMachineFunctionPassManager &CGMFPM) const {
  // Cleanup of redundant (identical) address/immediate loads.
  CGMFPM.addPass(MachineLateInstrsCleanupPass());

  // Branch folding must be run after regalloc and prolog/epilog insertion.
  CGMFPM.addPass(BranchFolderPass(Opt.EnableTailMerge));

  // Tail duplication.
  // Note that duplicating tail just increases code size and degrades
  // performance for targets that require Structured Control Flow.
  // In addition it can also make CFG irreducible. Thus we disable it.
  if (!TM.requiresStructuredCFG())
    CGMFPM.addPass(TailDuplicatePass());

  // Copy propagation.
  CGMFPM.addPass(MachineCopyPropagationPass());
}

/// Add standard basic block placement passes.
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addBlockPlacement(
    CodeGenMachineFunctionPassManager &CGMFPM) const {
  CGMFPM.addPass(MachineBlockPlacementPass(Opt.EnableTailMerge));
  // Run a separate pass to collect block placement statistics.
  if (Opt.EnableBlockPlacementStats)
    CGMFPM.addPass(MachineBlockPlacementStatsPass());
}

} // namespace llvm

#endif // LLVM_PASSES_CODEGENPASSBUILDER_H
