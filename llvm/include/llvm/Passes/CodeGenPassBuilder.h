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
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/RuntimeLibcallInfo.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/CodeGen/AssignmentTrackingAnalysis.h"
#include "llvm/CodeGen/BranchFoldingPass.h"
#include "llvm/CodeGen/CallBrPrepare.h"
#include "llvm/CodeGen/CodeGenPrepare.h"
#include "llvm/CodeGen/DeadMachineInstructionElim.h"
#include "llvm/CodeGen/DetectDeadLanes.h"
#include "llvm/CodeGen/DwarfEHPrepare.h"
#include "llvm/CodeGen/EarlyIfConversion.h"
#include "llvm/CodeGen/ExpandFp.h"
#include "llvm/CodeGen/ExpandLargeDivRem.h"
#include "llvm/CodeGen/ExpandMemCmp.h"
#include "llvm/CodeGen/ExpandPostRAPseudos.h"
#include "llvm/CodeGen/ExpandReductions.h"
#include "llvm/CodeGen/FEntryInserter.h"
#include "llvm/CodeGen/FinalizeISel.h"
#include "llvm/CodeGen/FixupStatepointCallerSaved.h"
#include "llvm/CodeGen/GCMetadata.h"
#include "llvm/CodeGen/GlobalMerge.h"
#include "llvm/CodeGen/GlobalMergeFunctions.h"
#include "llvm/CodeGen/IndirectBrExpand.h"
#include "llvm/CodeGen/InitUndef.h"
#include "llvm/CodeGen/InterleavedAccess.h"
#include "llvm/CodeGen/InterleavedLoadCombine.h"
#include "llvm/CodeGen/JMCInstrumenter.h"
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
#include "llvm/Pass.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/CGPassBuilderOption.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/CFGuard.h"
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
  struct PASS_NAME : public PassInfoMixin<PASS_NAME> {                         \
    template <typename... Ts> PASS_NAME(Ts &&...) {}                           \
    PreservedAnalyses run(Function &, FunctionAnalysisManager &) {             \
      return PreservedAnalyses::all();                                         \
    }                                                                          \
  };
#define DUMMY_MACHINE_MODULE_PASS(NAME, PASS_NAME)                             \
  struct PASS_NAME : public PassInfoMixin<PASS_NAME> {                         \
    template <typename... Ts> PASS_NAME(Ts &&...) {}                           \
    PreservedAnalyses run(Module &, ModuleAnalysisManager &) {                 \
      return PreservedAnalyses::all();                                         \
    }                                                                          \
  };
#define DUMMY_MACHINE_FUNCTION_PASS(NAME, PASS_NAME)                           \
  struct PASS_NAME : public PassInfoMixin<PASS_NAME> {                         \
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

    if (!Opt.OptimizeRegAlloc)
      Opt.OptimizeRegAlloc = getOptLevel() != CodeGenOptLevel::None;
  }

  ~CodeGenPassBuilder() {
    assert(FPM.isEmpty() && MFPM.isEmpty() &&
           "There are passes that have not been flushed from the function "
           "pipelines.");
  }

  Error buildPipeline(ModulePassManager &MPM, raw_pwrite_stream &Out,
                      raw_pwrite_stream *DwoOut,
                      CodeGenFileType FileType) const;

  PassInstrumentationCallbacks *getPassInstrumentationCallbacks() const {
    return PIC;
  }

protected:
  template <typename PassT>
  using is_module_pass_t = decltype(std::declval<PassT &>().run(
      std::declval<Module &>(), std::declval<ModuleAnalysisManager &>()));

  template <typename PassT>
  using is_function_pass_t = decltype(std::declval<PassT &>().run(
      std::declval<Function &>(), std::declval<FunctionAnalysisManager &>()));

  template <typename PassT>
  using is_machine_function_pass_t = decltype(std::declval<PassT &>().run(
      std::declval<MachineFunction &>(),
      std::declval<MachineFunctionAnalysisManager &>()));

  template <typename PassT>
  void addFunctionPass(PassT &&Pass, bool Force = false,
                       StringRef Name = PassT::name()) const {
    static_assert(is_detected<is_function_pass_t, PassT>::value &&
                  "Only function passes are supported.");
    if (!Force && !runBeforeAdding(Name))
      return;
    FPM.addPass(std::forward<PassT>(Pass));
  }

  template <typename PassT>
  void addModulePass(PassT &&Pass, ModulePassManager &MPM, bool Force = false,
                     StringRef Name = PassT::name()) const {
    static_assert(is_detected<is_module_pass_t, PassT>::value &&
                  "Only module passes are suported.");
    assert(FPM.isEmpty() && MFPM.isEmpty() &&
           "You cannot insert a module pass without first flushing the current "
           "function pipelines to the module pipeline.");
    if (!Force && !runBeforeAdding(Name))
      return;
    MPM.addPass(std::forward<PassT>(Pass));
  }

  template <typename PassT>
  void addMachineFunctionPass(PassT &&Pass, bool Force = false,
                              StringRef Name = PassT::name()) const {
    static_assert(is_detected<is_machine_function_pass_t, PassT>::value &&
                  "Only machine function passes are supported.");

    if (!Force && !runBeforeAdding(Name))
      return;
    MFPM.addPass(std::forward<PassT>(Pass));
    for (auto &C : AfterCallbacks)
      C(Name, MFPM);
  }

  void flushFPMsToMPM(ModulePassManager &MPM,
                      bool FreeMachineFunctions = false) const {
    if (FPM.isEmpty() && MFPM.isEmpty())
      return;
    if (!MFPM.isEmpty()) {
      FPM.addPass(createFunctionToMachineFunctionPassAdaptor(std::move(MFPM)));
      MFPM = MachineFunctionPassManager();
    }
    if (FreeMachineFunctions)
      FPM.addPass(FreeMachineFunctionPass());
    if (AddInCGSCCOrder) {
      MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(
          createCGSCCToFunctionPassAdaptor(std::move(FPM))));
    } else {
      MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
    }
    FPM = FunctionPassManager();
  }

  void requireCGSCCOrder(ModulePassManager &MPM) const {
    assert(!AddInCGSCCOrder);
    flushFPMsToMPM(MPM);
    AddInCGSCCOrder = true;
  }

  void stopAddingInCGSCCOrder(ModulePassManager &MPM) const {
    assert(AddInCGSCCOrder);
    flushFPMsToMPM(MPM);
    AddInCGSCCOrder = false;
  }

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
  Error addInstSelector(ModulePassManager &MPM) const {
    return make_error<StringError>("addInstSelector is not overridden",
                                   inconvertibleErrorCode());
  }

  /// Target can override this to add GlobalMergePass before all IR passes.
  void addGlobalMergePass(ModulePassManager &MPM) const {}

  /// Add passes that optimize instruction level parallelism for out-of-order
  /// targets. These passes are run while the machine code is still in SSA
  /// form, so they can use MachineTraceMetrics to control their heuristics.
  ///
  /// All passes added here should preserve the MachineDominatorTree,
  /// MachineLoopInfo, and MachineTraceMetrics analyses.
  void addILPOpts(ModulePassManager &MPM) const {}

  /// This method may be implemented by targets that want to run passes
  /// immediately before register allocation.
  void addPreRegAlloc(ModulePassManager &MPM) const {}

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
  void addPreRewrite(ModulePassManager &MPM) const {}

  /// Add passes to be run immediately after virtual registers are rewritten
  /// to physical registers.
  void addPostRewrite(ModulePassManager &MPM) const {}

  /// This method may be implemented by targets that want to run passes after
  /// register allocation pass pipeline but before prolog-epilog insertion.
  void addPostRegAlloc(ModulePassManager &MPM) const {}

  /// This method may be implemented by targets that want to run passes after
  /// prolog-epilog insertion and before the second instruction scheduling pass.
  void addPreSched2(ModulePassManager &MPM) const {}

  /// This pass may be implemented by targets that want to run passes
  /// immediately before machine code is emitted.
  void addPreEmitPass(ModulePassManager &MPM) const {}

  /// Targets may add passes immediately before machine code is emitted in this
  /// callback. This is called even later than `addPreEmitPass`.
  // FIXME: Rename `addPreEmitPass` to something more sensible given its actual
  // position and remove the `2` suffix here as this callback is what
  // `addPreEmitPass` *should* be but in reality isn't.
  void addPreEmitPass2(ModulePassManager &MPM) const {}

  /// {{@ For GlobalISel
  ///

  /// addPreISel - This method should add any "last minute" LLVM->LLVM
  /// passes (which are run just before instruction selector).
  void addPreISel(ModulePassManager &MPM) const {
    llvm_unreachable("addPreISel is not overridden");
  }

  /// This method should install an IR translator pass, which converts from
  /// LLVM code to machine instructions with possibly generic opcodes.
  Error addIRTranslator(ModulePassManager &MPM) const {
    return make_error<StringError>("addIRTranslator is not overridden",
                                   inconvertibleErrorCode());
  }

  /// This method may be implemented by targets that want to run passes
  /// immediately before legalization.
  void addPreLegalizeMachineIR(ModulePassManager &MPM) const {}

  /// This method should install a legalize pass, which converts the instruction
  /// sequence into one that can be selected by the target.
  Error addLegalizeMachineIR(ModulePassManager &MPM) const {
    return make_error<StringError>("addLegalizeMachineIR is not overridden",
                                   inconvertibleErrorCode());
  }

  /// This method may be implemented by targets that want to run passes
  /// immediately before the register bank selection.
  void addPreRegBankSelect(ModulePassManager &MPM) const {}

  /// This method should install a register bank selector pass, which
  /// assigns register banks to virtual registers without a register
  /// class or register banks.
  Error addRegBankSelect(ModulePassManager &MPM) const {
    return make_error<StringError>("addRegBankSelect is not overridden",
                                   inconvertibleErrorCode());
  }

  /// This method may be implemented by targets that want to run passes
  /// immediately before the (global) instruction selection.
  void addPreGlobalInstructionSelect(ModulePassManager &MPM) const {}

  /// This method should install a (global) instruction selector pass, which
  /// converts possibly generic instructions to fully target-specific
  /// instructions, thereby constraining all generic virtual registers to
  /// register classes.
  Error addGlobalInstructionSelect(ModulePassManager &MPM) const {
    return make_error<StringError>(
        "addGlobalInstructionSelect is not overridden",
        inconvertibleErrorCode());
  }
  /// @}}

  /// High level function that adds all passes necessary to go from llvm IR
  /// representation to the MI representation.
  /// Adds IR based lowering and target specific optimization passes and finally
  /// the core instruction selection passes.
  void addISelPasses(ModulePassManager &MPM) const;

  /// Add the actual instruction selection passes. This does not include
  /// preparation passes on IR.
  Error addCoreISelPasses(ModulePassManager &MPM) const;

  /// Add the complete, standard set of LLVM CodeGen passes.
  /// Fully developed targets will not generally override this.
  Error addMachinePasses(ModulePassManager &MPM) const;

  /// Add passes to lower exception handling for the code generator.
  void addPassesToHandleExceptions(ModulePassManager &MPM) const;

  /// Add common target configurable passes that perform LLVM IR to IR
  /// transforms following machine independent optimization.
  void addIRPasses(ModulePassManager &MPM) const;

  /// Add pass to prepare the LLVM IR for code generation. This should be done
  /// before exception handling preparation passes.
  void addCodeGenPrepare(ModulePassManager &MPM) const;

  /// Add common passes that perform LLVM IR to IR transforms in preparation for
  /// instruction selection.
  void addISelPrepare(ModulePassManager &MPM) const;

  /// Methods with trivial inline returns are convenient points in the common
  /// codegen pass pipeline where targets may insert passes. Methods with
  /// out-of-line standard implementations are major CodeGen stages called by
  /// addMachinePasses. Some targets may override major stages when inserting
  /// passes is insufficient, but maintaining overriden stages is more work.
  ///

  /// addMachineSSAOptimization - Add standard passes that optimize machine
  /// instructions in SSA form.
  void addMachineSSAOptimization(ModulePassManager &MPM) const;

  /// addFastRegAlloc - Add the minimum set of target-independent passes that
  /// are required for fast register allocation.
  Error addFastRegAlloc(ModulePassManager &MPM) const;

  /// addOptimizedRegAlloc - Add passes related to register allocation.
  /// CodeGenTargetMachineImpl provides standard regalloc passes for most
  /// targets.
  void addOptimizedRegAlloc(ModulePassManager &MPM) const;

  /// Add passes that optimize machine instructions after register allocation.
  void addMachineLateOptimization(ModulePassManager &MPM) const;

  /// addGCPasses - Add late codegen passes that analyze code for garbage
  /// collection. This should return true if GC info should be printed after
  /// these passes.
  void addGCPasses(ModulePassManager &MPM) const {}

  /// Add standard basic block placement passes.
  void addBlockPlacement(ModulePassManager &MPM) const;

  using CreateMCStreamer =
      std::function<Expected<std::unique_ptr<MCStreamer>>(MCContext &)>;
  void addAsmPrinter(ModulePassManager &MPM, CreateMCStreamer) const {
    llvm_unreachable("addAsmPrinter is not overridden");
  }

  /// Utilities for targets to add passes to the pass manager.
  ///

  /// createTargetRegisterAllocator - Create the register allocator pass for
  /// this target at the current optimization level.
  void addTargetRegisterAllocator(ModulePassManager &MPM, bool Optimized) const;

  /// addMachinePasses helper to create the target-selected or overriden
  /// regalloc pass.
  void addRegAllocPass(ModulePassManager &MPM, bool Optimized) const;

  /// Add core register alloator passes which do the actual register assignment
  /// and rewriting. \returns true if any passes were added.
  Error addRegAssignmentFast(ModulePassManager &MPM) const;
  Error addRegAssignmentOptimized(ModulePassManager &MPM) const;

  /// Allow the target to disable a specific pass by default.
  /// Backend can declare unwanted passes in constructor.
  template <typename... PassTs> void disablePass() {
    BeforeCallbacks.emplace_back(
        [](StringRef Name) { return ((Name != PassTs::name()) && ...); });
  }

  /// Insert InsertedPass pass after TargetPass pass.
  /// Only machine function passes are supported.
  template <typename TargetPassT, typename InsertedPassT>
  void insertPass(InsertedPassT &&Pass) const {
    AfterCallbacks.emplace_back(
        [&](StringRef Name, MachineFunctionPassManager &MFPM) mutable {
          if (Name == TargetPassT::name() &&
              runBeforeAdding(InsertedPassT::name())) {
            MFPM.addPass(std::forward<InsertedPassT>(Pass));
          }
        });
  }

private:
  DerivedT &derived() { return static_cast<DerivedT &>(*this); }
  const DerivedT &derived() const {
    return static_cast<const DerivedT &>(*this);
  }

  bool runBeforeAdding(StringRef Name) const {
    bool ShouldAdd = true;
    for (auto &C : BeforeCallbacks)
      ShouldAdd &= C(Name);
    return ShouldAdd;
  }

  void setStartStopPasses(const TargetPassConfig::StartStopInfo &Info) const;

  Error verifyStartStop(const TargetPassConfig::StartStopInfo &Info) const;

  mutable SmallVector<llvm::unique_function<bool(StringRef)>, 4>
      BeforeCallbacks;
  mutable SmallVector<
      llvm::unique_function<void(StringRef, MachineFunctionPassManager &)>, 4>
      AfterCallbacks;

  /// Helper variable for `-start-before/-start-after/-stop-before/-stop-after`
  mutable bool Started = true;
  mutable bool Stopped = true;
  mutable bool AddInCGSCCOrder = false;

  mutable FunctionPassManager FPM;
  mutable MachineFunctionPassManager MFPM;
};

template <typename Derived, typename TargetMachineT>
Error CodeGenPassBuilder<Derived, TargetMachineT>::buildPipeline(
    ModulePassManager &MPM, raw_pwrite_stream &Out, raw_pwrite_stream *DwoOut,
    CodeGenFileType FileType) const {
  auto StartStopInfo = TargetPassConfig::getStartStopInfo(*PIC);
  if (!StartStopInfo)
    return StartStopInfo.takeError();
  setStartStopPasses(*StartStopInfo);

  bool PrintAsm = TargetPassConfig::willCompleteCodeGenPipeline();
  bool PrintMIR = !PrintAsm && FileType != CodeGenFileType::Null;

  addModulePass(RequireAnalysisPass<MachineModuleAnalysis, Module>(), MPM,
                /*Force=*/true);
  addModulePass(RequireAnalysisPass<ProfileSummaryAnalysis, Module>(), MPM,
                /*Force=*/true);
  addModulePass(RequireAnalysisPass<CollectorMetadataAnalysis, Module>(), MPM,
                /*Force=*/true);
  addModulePass(RequireAnalysisPass<RuntimeLibraryAnalysis, Module>(), MPM,
                /*Force=*/true);
  addISelPasses(MPM);
  flushFPMsToMPM(MPM);

  if (PrintMIR)
    addModulePass(PrintMIRPreparePass(Out), MPM, /*Force=*/true);

  if (auto Err = addCoreISelPasses(MPM))
    return std::move(Err);

  if (auto Err = derived().addMachinePasses(MPM))
    return std::move(Err);

  if (!Opt.DisableVerify)
    addMachineFunctionPass(MachineVerifierPass());

  if (PrintAsm) {
    derived().addAsmPrinter(
        MPM, [this, &Out, DwoOut, FileType](MCContext &Ctx) {
          return this->TM.createMCStreamer(Out, DwoOut, FileType, Ctx);
        });
  }

  if (PrintMIR)
    addMachineFunctionPass(PrintMIRPass(Out), /*Force=*/true);

  flushFPMsToMPM(MPM, true);

  return verifyStartStop(*StartStopInfo);
}

template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::setStartStopPasses(
    const TargetPassConfig::StartStopInfo &Info) const {
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

template <typename Derived, typename TargetMachineT>
Error CodeGenPassBuilder<Derived, TargetMachineT>::verifyStartStop(
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

template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addISelPasses(
    ModulePassManager &MPM) const {
  derived().addGlobalMergePass(MPM);
  if (TM.useEmulatedTLS())
    addModulePass(LowerEmuTLSPass(), MPM);

  addModulePass(PreISelIntrinsicLoweringPass(&TM), MPM);
  addFunctionPass(ExpandLargeDivRemPass(TM));
  addFunctionPass(ExpandFpPass(TM, getOptLevel()));

  derived().addIRPasses(MPM);
  derived().addCodeGenPrepare(MPM);
  addPassesToHandleExceptions(MPM);
  derived().addISelPrepare(MPM);
}

/// Add common target configurable passes that perform LLVM IR to IR transforms
/// following machine independent optimization.
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addIRPasses(
    ModulePassManager &MPM) const {
  // Before running any passes, run the verifier to determine if the input
  // coming from the front-end and/or optimizer is valid.
  if (!Opt.DisableVerify)
    addFunctionPass(VerifierPass(), /*Force=*/true);

  // Run loop strength reduction before anything else.
  if (getOptLevel() != CodeGenOptLevel::None && !Opt.DisableLSR) {
    LoopPassManager LPM;
    LPM.addPass(CanonicalizeFreezeInLoopsPass());
    LPM.addPass(LoopStrengthReducePass());
    if (Opt.EnableLoopTermFold)
      LPM.addPass(LoopTermFoldPass());
    addFunctionPass(createFunctionToLoopPassAdaptor(std::move(LPM),
                                                    /*UseMemorySSA=*/true));
  }

  if (getOptLevel() != CodeGenOptLevel::None) {
    // The MergeICmpsPass tries to create memcmp calls by grouping sequences of
    // loads and compares. ExpandMemCmpPass then tries to expand those calls
    // into optimally-sized loads and compares. The transforms are enabled by a
    // target lowering hook.
    if (!Opt.DisableMergeICmps)
      addFunctionPass(MergeICmpsPass());
    addFunctionPass(ExpandMemCmpPass(TM));
  }

  // Run GC lowering passes for builtin collectors
  // TODO: add a pass insertion point here
  addFunctionPass(GCLoweringPass());
  // Explicitly check to see if we should add ShadowStackGCLowering to avoid
  // splitting the function pipeline if we do not have to.
  if (runBeforeAdding(ShadowStackGCLoweringPass::name())) {
    flushFPMsToMPM(MPM);
    addModulePass(ShadowStackGCLoweringPass(), MPM);
  }
  addFunctionPass(LowerConstantIntrinsicsPass());

  // Make sure that no unreachable blocks are instruction selected.
  addFunctionPass(UnreachableBlockElimPass());

  // Prepare expensive constants for SelectionDAG.
  if (getOptLevel() != CodeGenOptLevel::None && !Opt.DisableConstantHoisting)
    addFunctionPass(ConstantHoistingPass());

  // Replace calls to LLVM intrinsics (e.g., exp, log) operating on vector
  // operands with calls to the corresponding functions in a vector library.
  if (getOptLevel() != CodeGenOptLevel::None)
    addFunctionPass(ReplaceWithVeclib());

  if (getOptLevel() != CodeGenOptLevel::None &&
      !Opt.DisablePartialLibcallInlining)
    addFunctionPass(PartiallyInlineLibCallsPass());

  // Instrument function entry and exit, e.g. with calls to mcount().
  addFunctionPass(EntryExitInstrumenterPass(/*PostInlining=*/true));

  // Add scalarization of target's unsupported masked memory intrinsics pass.
  // the unsupported intrinsic will be replaced with a chain of basic blocks,
  // that stores/loads element one-by-one if the appropriate mask bit is set.
  addFunctionPass(ScalarizeMaskedMemIntrinPass());

  // Expand reduction intrinsics into shuffle sequences if the target wants to.
  if (!Opt.DisableExpandReductions)
    addFunctionPass(ExpandReductionsPass());

  // Convert conditional moves to conditional jumps when profitable.
  if (getOptLevel() != CodeGenOptLevel::None && !Opt.DisableSelectOptimize)
    addFunctionPass(SelectOptimizePass(TM));

  if (Opt.EnableGlobalMergeFunc) {
    flushFPMsToMPM(MPM);
    addModulePass(GlobalMergeFuncPass(), MPM);
  }
}

/// Turn exception handling constructs into something the code generators can
/// handle.
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addPassesToHandleExceptions(
    ModulePassManager &MPM) const {
  const MCAsmInfo *MCAI = TM.getMCAsmInfo();
  assert(MCAI && "No MCAsmInfo");
  switch (MCAI->getExceptionHandlingType()) {
  case ExceptionHandling::SjLj:
    // SjLj piggy-backs on dwarf for this bit. The cleanups done apply to both
    // Dwarf EH prepare needs to be run after SjLj prepare. Otherwise,
    // catch info can get misplaced when a selector ends up more than one block
    // removed from the parent invoke(s). This could happen when a landing
    // pad is shared by multiple invokes and is also a target of a normal
    // edge from elsewhere.
    addFunctionPass(SjLjEHPreparePass(&TM));
    [[fallthrough]];
  case ExceptionHandling::DwarfCFI:
  case ExceptionHandling::ARM:
  case ExceptionHandling::AIX:
  case ExceptionHandling::ZOS:
    addFunctionPass(DwarfEHPreparePass(TM));
    break;
  case ExceptionHandling::WinEH:
    // We support using both GCC-style and MSVC-style exceptions on Windows, so
    // add both preparation passes. Each pass will only actually run if it
    // recognizes the personality function.
    addFunctionPass(WinEHPreparePass());
    addFunctionPass(DwarfEHPreparePass(TM));
    break;
  case ExceptionHandling::Wasm:
    // Wasm EH uses Windows EH instructions, but it does not need to demote PHIs
    // on catchpads and cleanuppads because it does not outline them into
    // funclets. Catchswitch blocks are not lowered in SelectionDAG, so we
    // should remove PHIs there.
    addFunctionPass(WinEHPreparePass(/*DemoteCatchSwitchPHIOnly=*/false));
    addFunctionPass(WasmEHPreparePass());
    break;
  case ExceptionHandling::None:
    addFunctionPass(LowerInvokePass());

    // The lower invoke pass may create unreachable code. Remove it.
    addFunctionPass(UnreachableBlockElimPass());
    break;
  }
}

/// Add pass to prepare the LLVM IR for code generation. This should be done
/// before exception handling preparation passes.
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addCodeGenPrepare(
    ModulePassManager &MPM) const {
  if (getOptLevel() != CodeGenOptLevel::None && !Opt.DisableCGP)
    addFunctionPass(CodeGenPreparePass(TM));
  // TODO: Default ctor'd RewriteSymbolPass is no-op.
  // addPass(RewriteSymbolPass());
}

/// Add common passes that perform LLVM IR to IR transforms in preparation for
/// instruction selection.
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addISelPrepare(
    ModulePassManager &MPM) const {
  derived().addPreISel(MPM);

  if (Opt.RequiresCodeGenSCCOrder && !AddInCGSCCOrder)
    requireCGSCCOrder(MPM);

  if (getOptLevel() != CodeGenOptLevel::None)
    addFunctionPass(ObjCARCContractPass());

  addFunctionPass(CallBrPreparePass());
  // Add both the safe stack and the stack protection passes: each of them will
  // only protect functions that have corresponding attributes.
  addFunctionPass(SafeStackPass(TM));
  addFunctionPass(StackProtectorPass(TM));

  if (Opt.PrintISelInput)
    addFunctionPass(PrintFunctionPass(
        dbgs(), "\n\n*** Final LLVM Code input to ISel ***\n"));

  // All passes which modify the LLVM IR are now complete; run the verifier
  // to ensure that the IR is valid.
  if (!Opt.DisableVerify)
    addFunctionPass(VerifierPass(), /*Force=*/true);
}

template <typename Derived, typename TargetMachineT>
Error CodeGenPassBuilder<Derived, TargetMachineT>::addCoreISelPasses(
    ModulePassManager &MPM) const {
  // Enable FastISel with -fast-isel, but allow that to be overridden.
  TM.setO0WantsFastISel(Opt.EnableFastISelOption.value_or(true));

  // Determine an instruction selector.
  enum class SelectorType { SelectionDAG, FastISel, GlobalISel };
  SelectorType Selector;

  if (Opt.EnableFastISelOption && *Opt.EnableFastISelOption == true)
    Selector = SelectorType::FastISel;
  else if ((Opt.EnableGlobalISelOption &&
            *Opt.EnableGlobalISelOption == true) ||
           (TM.Options.EnableGlobalISel &&
            (!Opt.EnableGlobalISelOption ||
             *Opt.EnableGlobalISelOption == false)))
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
    if (auto Err = derived().addIRTranslator(MPM))
      return std::move(Err);

    derived().addPreLegalizeMachineIR(MPM);

    if (auto Err = derived().addLegalizeMachineIR(MPM))
      return std::move(Err);

    // Before running the register bank selector, ask the target if it
    // wants to run some passes.
    derived().addPreRegBankSelect(MPM);

    if (auto Err = derived().addRegBankSelect(MPM))
      return std::move(Err);

    derived().addPreGlobalInstructionSelect(MPM);

    if (auto Err = derived().addGlobalInstructionSelect(MPM))
      return std::move(Err);

    // Pass to reset the MachineFunction if the ISel failed.
    addMachineFunctionPass(ResetMachineFunctionPass(
        reportDiagnosticWhenGlobalISelFallback(), isGlobalISelAbortEnabled()));

    // Provide a fallback path when we do not want to abort on
    // not-yet-supported input.
    if (!isGlobalISelAbortEnabled())
      if (auto Err = derived().addInstSelector(MPM))
        return std::move(Err);

  } else if (auto Err = derived().addInstSelector(MPM))
    return std::move(Err);

  // Expand pseudo-instructions emitted by ISel. Don't run the verifier before
  // FinalizeISel.
  addMachineFunctionPass(FinalizeISelPass());

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
    ModulePassManager &MPM) const {
  // Add passes that optimize machine instructions in SSA form.
  if (getOptLevel() != CodeGenOptLevel::None) {
    derived().addMachineSSAOptimization(MPM);
  } else {
    // If the target requests it, assign local variables to stack slots relative
    // to one another and simplify frame index references where possible.
    addMachineFunctionPass(LocalStackSlotAllocationPass());
  }

  if (TM.Options.EnableIPRA) {
    flushFPMsToMPM(MPM);
    addModulePass(RequireAnalysisPass<PhysicalRegisterUsageAnalysis, Module>(),
                  MPM);
    addMachineFunctionPass(RegUsageInfoPropagationPass());
  }
  // Run pre-ra passes.
  derived().addPreRegAlloc(MPM);

  // Run register allocation and passes that are tightly coupled with it,
  // including phi elimination and scheduling.
  if (*Opt.OptimizeRegAlloc) {
    derived().addOptimizedRegAlloc(MPM);
  } else {
    if (auto Err = derived().addFastRegAlloc(MPM))
      return Err;
  }

  // Run post-ra passes.
  derived().addPostRegAlloc(MPM);

  addMachineFunctionPass(RemoveRedundantDebugValuesPass());
  addMachineFunctionPass(FixupStatepointCallerSavedPass());

  // Insert prolog/epilog code.  Eliminate abstract frame index references...
  if (getOptLevel() != CodeGenOptLevel::None) {
    addMachineFunctionPass(PostRAMachineSinkingPass());
    addMachineFunctionPass(ShrinkWrapPass());
  }

  addMachineFunctionPass(PrologEpilogInserterPass());

  /// Add passes that optimize machine instructions after register allocation.
  if (getOptLevel() != CodeGenOptLevel::None)
    derived().addMachineLateOptimization(MPM);

  // Expand pseudo instructions before second scheduling pass.
  addMachineFunctionPass(ExpandPostRAPseudosPass());

  // Run pre-sched2 passes.
  derived().addPreSched2(MPM);

  if (Opt.EnableImplicitNullChecks)
    addMachineFunctionPass(ImplicitNullChecksPass());

  // Second pass scheduler.
  // Let Target optionally insert this pass by itself at some other
  // point.
  if (getOptLevel() != CodeGenOptLevel::None &&
      !TM.targetSchedulesPostRAScheduling()) {
    if (Opt.MISchedPostRA)
      addMachineFunctionPass(PostMachineSchedulerPass(&TM));
    else
      addMachineFunctionPass(PostRASchedulerPass(&TM));
  }

  // GC
  derived().addGCPasses(MPM);

  // Basic block placement.
  if (getOptLevel() != CodeGenOptLevel::None)
    derived().addBlockPlacement(MPM);

  // Insert before XRay Instrumentation.
  addMachineFunctionPass(FEntryInserterPass());

  addMachineFunctionPass(XRayInstrumentationPass());
  addMachineFunctionPass(PatchableFunctionPass());

  derived().addPreEmitPass(MPM);

  if (TM.Options.EnableIPRA)
    // Collect register usage information and produce a register mask of
    // clobbered registers, to be used to optimize call sites.
    addMachineFunctionPass(RegUsageInfoCollectorPass());

  addMachineFunctionPass(FuncletLayoutPass());

  addMachineFunctionPass(RemoveLoadsIntoFakeUsesPass());
  addMachineFunctionPass(StackMapLivenessPass());
  addMachineFunctionPass(LiveDebugValuesPass(
      getTM<TargetMachine>().Options.ShouldEmitDebugEntryValues()));
  addMachineFunctionPass(MachineSanitizerBinaryMetadataPass());

  if (TM.Options.EnableMachineOutliner &&
      getOptLevel() != CodeGenOptLevel::None &&
      Opt.EnableMachineOutliner != RunOutliner::NeverOutline) {
    if (Opt.EnableMachineOutliner != RunOutliner::TargetDefault ||
        TM.Options.SupportsDefaultOutlining) {
      flushFPMsToMPM(MPM);
      addModulePass(MachineOutlinerPass(Opt.EnableMachineOutliner), MPM);
    }
  }

  addMachineFunctionPass(StackFrameLayoutAnalysisPass());

  // Add passes that directly emit MI after all other MI passes.
  derived().addPreEmitPass2(MPM);

  return Error::success();
}

/// Add passes that optimize machine instructions in SSA form.
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addMachineSSAOptimization(
    ModulePassManager &MPM) const {
  // Pre-ra tail duplication.
  addMachineFunctionPass(EarlyTailDuplicatePass());

  // Optimize PHIs before DCE: removing dead PHI cycles may make more
  // instructions dead.
  addMachineFunctionPass(OptimizePHIsPass());

  // This pass merges large allocas. StackSlotColoring is a different pass
  // which merges spill slots.
  addMachineFunctionPass(StackColoringPass());

  // If the target requests it, assign local variables to stack slots relative
  // to one another and simplify frame index references where possible.
  addMachineFunctionPass(LocalStackSlotAllocationPass());

  // With optimization, dead code should already be eliminated. However
  // there is one known exception: lowered code for arguments that are only
  // used by tail calls, where the tail calls reuse the incoming stack
  // arguments directly (see t11 in test/CodeGen/X86/sibcall.ll).
  addMachineFunctionPass(DeadMachineInstructionElimPass());

  // Allow targets to insert passes that improve instruction level parallelism,
  // like if-conversion. Such passes will typically need dominator trees and
  // loop info, just like LICM and CSE below.
  derived().addILPOpts(MPM);

  addMachineFunctionPass(EarlyMachineLICMPass());
  addMachineFunctionPass(MachineCSEPass());

  addMachineFunctionPass(MachineSinkingPass(Opt.EnableSinkAndFold));

  addMachineFunctionPass(PeepholeOptimizerPass());
  // Clean-up the dead code that may have been generated by peephole
  // rewriting.
  addMachineFunctionPass(DeadMachineInstructionElimPass());
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
    ModulePassManager &MPM, bool Optimized) const {
  if (Optimized)
    addMachineFunctionPass(RAGreedyPass());
  else
    addMachineFunctionPass(RegAllocFastPass());
}

/// Find and instantiate the register allocation pass requested by this target
/// at the current optimization level.  Different register allocators are
/// defined as separate passes because they may require different analysis.
///
/// This helper ensures that the -regalloc-npm= option is always available,
/// even for targets that override the default allocator.
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addRegAllocPass(
    ModulePassManager &MPM, bool Optimized) const {
  // Use the specified -regalloc-npm={basic|greedy|fast|pbqp}
  if (Opt.RegAlloc > RegAllocType::Default) {
    switch (Opt.RegAlloc) {
    case RegAllocType::Fast:
      addMachineFunctionPass(RegAllocFastPass());
      break;
    case RegAllocType::Greedy:
      addMachineFunctionPass(RAGreedyPass());
      break;
    default:
      reportFatalUsageError("register allocator not supported yet");
    }
    return;
  }
  // -regalloc=default or unspecified, so pick based on the optimization level
  // or ask the target for the regalloc pass.
  derived().addTargetRegisterAllocator(MPM, Optimized);
}

template <typename Derived, typename TargetMachineT>
Error CodeGenPassBuilder<Derived, TargetMachineT>::addRegAssignmentFast(
    ModulePassManager &MPM) const {
  // TODO: Ensure allocator is default or fast.
  addRegAllocPass(MPM, false);
  return Error::success();
}

template <typename Derived, typename TargetMachineT>
Error CodeGenPassBuilder<Derived, TargetMachineT>::addRegAssignmentOptimized(
    ModulePassManager &MPM) const {
  // Add the selected register allocation pass.
  addRegAllocPass(MPM, true);

  // Allow targets to change the register assignments before rewriting.
  derived().addPreRewrite(MPM);

  // Finally rewrite virtual registers.
  addMachineFunctionPass(VirtRegRewriterPass());
  // Perform stack slot coloring and post-ra machine LICM.
  //
  // FIXME: Re-enable coloring with register when it's capable of adding
  // kill markers.
  addMachineFunctionPass(StackSlotColoringPass());

  return Error::success();
}

/// Add the minimum set of target-independent passes that are required for
/// register allocation. No coalescing or scheduling.
template <typename Derived, typename TargetMachineT>
Error CodeGenPassBuilder<Derived, TargetMachineT>::addFastRegAlloc(
    ModulePassManager &MPM) const {
  addMachineFunctionPass(PHIEliminationPass());
  addMachineFunctionPass(TwoAddressInstructionPass());
  return derived().addRegAssignmentFast(MPM);
}

/// Add standard target-independent passes that are tightly coupled with
/// optimized register allocation, including coalescing, machine instruction
/// scheduling, and register allocation itself.
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addOptimizedRegAlloc(
    ModulePassManager &MPM) const {
  addMachineFunctionPass(DetectDeadLanesPass());

  addMachineFunctionPass(InitUndefPass());

  addMachineFunctionPass(ProcessImplicitDefsPass());

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
  addMachineFunctionPass(UnreachableMachineBlockElimPass());
  addMachineFunctionPass(
      RequireAnalysisPass<LiveVariablesAnalysis, MachineFunction>());

  // Edge splitting is smarter with machine loop info.
  addMachineFunctionPass(
      RequireAnalysisPass<MachineLoopAnalysis, MachineFunction>());
  addMachineFunctionPass(PHIEliminationPass());

  // Eventually, we want to run LiveIntervals before PHI elimination.
  if (Opt.EarlyLiveIntervals)
    addMachineFunctionPass(
        RequireAnalysisPass<LiveIntervalsAnalysis, MachineFunction>());

  addMachineFunctionPass(TwoAddressInstructionPass());
  addMachineFunctionPass(RegisterCoalescerPass());

  // The machine scheduler may accidentally create disconnected components
  // when moving subregister definitions around, avoid this by splitting them to
  // separate vregs before. Splitting can also improve reg. allocation quality.
  addMachineFunctionPass(RenameIndependentSubregsPass());

  // PreRA instruction scheduling.
  addMachineFunctionPass(MachineSchedulerPass(&TM));

  if (auto E = derived().addRegAssignmentOptimized(MPM)) {
    // addRegAssignmentOptimized did not add a reg alloc pass, so do nothing.
    return;
  }
  // Allow targets to expand pseudo instructions depending on the choice of
  // registers before MachineCopyPropagation.
  derived().addPostRewrite(MPM);

  // Copy propagate to forward register uses and try to eliminate COPYs that
  // were not coalesced.
  addMachineFunctionPass(MachineCopyPropagationPass());

  // Run post-ra machine LICM to hoist reloads / remats.
  //
  // FIXME: can this move into MachineLateOptimization?
  addMachineFunctionPass(MachineLICMPass());
}

//===---------------------------------------------------------------------===//
/// Post RegAlloc Pass Configuration
//===---------------------------------------------------------------------===//

/// Add passes that optimize machine instructions after register allocation.
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addMachineLateOptimization(
    ModulePassManager &MPM) const {
  // Branch folding must be run after regalloc and prolog/epilog insertion.
  addMachineFunctionPass(BranchFolderPass(Opt.EnableTailMerge));

  // Tail duplication.
  // Note that duplicating tail just increases code size and degrades
  // performance for targets that require Structured Control Flow.
  // In addition it can also make CFG irreducible. Thus we disable it.
  if (!TM.requiresStructuredCFG())
    addMachineFunctionPass(TailDuplicatePass());

  // Cleanup of redundant (identical) address/immediate loads.
  addMachineFunctionPass(MachineLateInstrsCleanupPass());

  // Copy propagation.
  addMachineFunctionPass(MachineCopyPropagationPass());
}

/// Add standard basic block placement passes.
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addBlockPlacement(
    ModulePassManager &MPM) const {
  addMachineFunctionPass(MachineBlockPlacementPass(Opt.EnableTailMerge));
  // Run a separate pass to collect block placement statistics.
  if (Opt.EnableBlockPlacementStats)
    addMachineFunctionPass(MachineBlockPlacementStatsPass());
}

} // namespace llvm

#endif // LLVM_PASSES_CODEGENPASSBUILDER_H
