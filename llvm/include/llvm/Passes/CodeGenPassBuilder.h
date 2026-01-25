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
#include "llvm/CodeGen/BranchFoldingPass.h"
#include "llvm/CodeGen/CallBrPrepare.h"
#include "llvm/CodeGen/CodeGenPrepare.h"
#include "llvm/CodeGen/DeadMachineInstructionElim.h"
#include "llvm/CodeGen/DetectDeadLanes.h"
#include "llvm/CodeGen/DwarfEHPrepare.h"
#include "llvm/CodeGen/ExpandIRInsts.h"
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

class PassManagerWrapper {
private:
  PassManagerWrapper(ModulePassManager &ModulePM) : MPM(ModulePM) {};

  ModulePassManager &MPM;
  FunctionPassManager FPM;
  MachineFunctionPassManager MFPM;

  template <typename DerivedT, typename TargetMachineT>
  friend class CodeGenPassBuilder;
};

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
  void addFunctionPass(PassT &&Pass, PassManagerWrapper &PMW,
                       bool Force = false,
                       StringRef Name = PassT::name()) const {
    static_assert(is_detected<is_function_pass_t, PassT>::value &&
                  "Only function passes are supported.");
    if (!Force && !runBeforeAdding(Name))
      return;
    PMW.FPM.addPass(std::forward<PassT>(Pass));
  }

  template <typename PassT>
  void addModulePass(PassT &&Pass, PassManagerWrapper &PMW, bool Force = false,
                     StringRef Name = PassT::name()) const {
    static_assert(is_detected<is_module_pass_t, PassT>::value &&
                  "Only module passes are suported.");
    assert(PMW.FPM.isEmpty() && PMW.MFPM.isEmpty() &&
           "You cannot insert a module pass without first flushing the current "
           "function pipelines to the module pipeline.");
    if (!Force && !runBeforeAdding(Name))
      return;
    PMW.MPM.addPass(std::forward<PassT>(Pass));
  }

  template <typename PassT>
  void addMachineFunctionPass(PassT &&Pass, PassManagerWrapper &PMW,
                              bool Force = false,
                              StringRef Name = PassT::name()) const {
    static_assert(is_detected<is_machine_function_pass_t, PassT>::value &&
                  "Only machine function passes are supported.");

    if (!Force && !runBeforeAdding(Name))
      return;
    PMW.MFPM.addPass(std::forward<PassT>(Pass));
    for (auto &C : AfterCallbacks)
      C(Name, PMW.MFPM);
  }

  void flushFPMsToMPM(PassManagerWrapper &PMW,
                      bool FreeMachineFunctions = false) const {
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

  void requireCGSCCOrder(PassManagerWrapper &PMW) const {
    assert(!AddInCGSCCOrder);
    assert(PMW.FPM.isEmpty() && PMW.MFPM.isEmpty() &&
           "Requiring CGSCC ordering requires flushing the current function "
           "pipelines to the MPM.");
    AddInCGSCCOrder = true;
  }

  void stopAddingInCGSCCOrder(PassManagerWrapper &PMW) const {
    assert(AddInCGSCCOrder);
    assert(PMW.FPM.isEmpty() && PMW.MFPM.isEmpty() &&
           "Stopping CGSCC ordering requires flushing the current function "
           "pipelines to the MPM.");
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
  Error addInstSelector(PassManagerWrapper &PMW) const {
    return make_error<StringError>("addInstSelector is not overridden",
                                   inconvertibleErrorCode());
  }

  /// Target can override this to add GlobalMergePass before all IR passes.
  void addGlobalMergePass(PassManagerWrapper &PMW) const {}

  /// Add passes that optimize instruction level parallelism for out-of-order
  /// targets. These passes are run while the machine code is still in SSA
  /// form, so they can use MachineTraceMetrics to control their heuristics.
  ///
  /// All passes added here should preserve the MachineDominatorTree,
  /// MachineLoopInfo, and MachineTraceMetrics analyses.
  void addILPOpts(PassManagerWrapper &PMW) const {}

  /// This method may be implemented by targets that want to run passes
  /// immediately before register allocation.
  void addPreRegAlloc(PassManagerWrapper &PMW) const {}

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
  void addPreRewrite(PassManagerWrapper &PMW) const {}

  /// Add passes to be run immediately after virtual registers are rewritten
  /// to physical registers.
  void addPostRewrite(PassManagerWrapper &PMW) const {}

  /// This method may be implemented by targets that want to run passes after
  /// register allocation pass pipeline but before prolog-epilog insertion.
  void addPostRegAlloc(PassManagerWrapper &PMW) const {}

  /// This method may be implemented by targets that want to run passes after
  /// prolog-epilog insertion and before the second instruction scheduling pass.
  void addPreSched2(PassManagerWrapper &PMW) const {}

  /// This pass may be implemented by targets that want to run passes
  /// immediately before machine code is emitted.
  void addPreEmitPass(PassManagerWrapper &PMW) const {}

  /// Targets may add passes immediately before machine code is emitted in this
  /// callback. This is called even later than `addPreEmitPass`.
  // FIXME: Rename `addPreEmitPass` to something more sensible given its actual
  // position and remove the `2` suffix here as this callback is what
  // `addPreEmitPass` *should* be but in reality isn't.
  void addPreEmitPass2(PassManagerWrapper &PMW) const {}

  /// {{@ For GlobalISel
  ///

  /// addPreISel - This method should add any "last minute" LLVM->LLVM
  /// passes (which are run just before instruction selector).
  void addPreISel(PassManagerWrapper &PMW) const {
    llvm_unreachable("addPreISel is not overridden");
  }

  /// This method should install an IR translator pass, which converts from
  /// LLVM code to machine instructions with possibly generic opcodes.
  Error addIRTranslator(PassManagerWrapper &PMW) const {
    return make_error<StringError>("addIRTranslator is not overridden",
                                   inconvertibleErrorCode());
  }

  /// This method may be implemented by targets that want to run passes
  /// immediately before legalization.
  void addPreLegalizeMachineIR(PassManagerWrapper &PMW) const {}

  /// This method should install a legalize pass, which converts the instruction
  /// sequence into one that can be selected by the target.
  Error addLegalizeMachineIR(PassManagerWrapper &PMW) const {
    return make_error<StringError>("addLegalizeMachineIR is not overridden",
                                   inconvertibleErrorCode());
  }

  /// This method may be implemented by targets that want to run passes
  /// immediately before the register bank selection.
  void addPreRegBankSelect(PassManagerWrapper &PMW) const {}

  /// This method should install a register bank selector pass, which
  /// assigns register banks to virtual registers without a register
  /// class or register banks.
  Error addRegBankSelect(PassManagerWrapper &PMW) const {
    return make_error<StringError>("addRegBankSelect is not overridden",
                                   inconvertibleErrorCode());
  }

  /// This method may be implemented by targets that want to run passes
  /// immediately before the (global) instruction selection.
  void addPreGlobalInstructionSelect(PassManagerWrapper &PMW) const {}

  /// This method should install a (global) instruction selector pass, which
  /// converts possibly generic instructions to fully target-specific
  /// instructions, thereby constraining all generic virtual registers to
  /// register classes.
  Error addGlobalInstructionSelect(PassManagerWrapper &PMWM) const {
    return make_error<StringError>(
        "addGlobalInstructionSelect is not overridden",
        inconvertibleErrorCode());
  }
  /// @}}

  /// High level function that adds all passes necessary to go from llvm IR
  /// representation to the MI representation.
  /// Adds IR based lowering and target specific optimization passes and finally
  /// the core instruction selection passes.
  void addISelPasses(PassManagerWrapper &PMW) const;

  /// Add the actual instruction selection passes. This does not include
  /// preparation passes on IR.
  Error addCoreISelPasses(PassManagerWrapper &PMW) const;

  /// Add the complete, standard set of LLVM CodeGen passes.
  /// Fully developed targets will not generally override this.
  Error addMachinePasses(PassManagerWrapper &PMW) const;

  /// Add passes to lower exception handling for the code generator.
  void addPassesToHandleExceptions(PassManagerWrapper &PMW) const;

  /// Add common target configurable passes that perform LLVM IR to IR
  /// transforms following machine independent optimization.
  void addIRPasses(PassManagerWrapper &PMW) const;

  /// Add pass to prepare the LLVM IR for code generation. This should be done
  /// before exception handling preparation passes.
  void addCodeGenPrepare(PassManagerWrapper &PMW) const;

  /// Add common passes that perform LLVM IR to IR transforms in preparation for
  /// instruction selection.
  void addISelPrepare(PassManagerWrapper &PMW) const;

  /// Methods with trivial inline returns are convenient points in the common
  /// codegen pass pipeline where targets may insert passes. Methods with
  /// out-of-line standard implementations are major CodeGen stages called by
  /// addMachinePasses. Some targets may override major stages when inserting
  /// passes is insufficient, but maintaining overriden stages is more work.
  ///

  /// addMachineSSAOptimization - Add standard passes that optimize machine
  /// instructions in SSA form.
  void addMachineSSAOptimization(PassManagerWrapper &PMW) const;

  /// addFastRegAlloc - Add the minimum set of target-independent passes that
  /// are required for fast register allocation.
  Error addFastRegAlloc(PassManagerWrapper &PMW) const;

  /// addOptimizedRegAlloc - Add passes related to register allocation.
  /// CodeGenTargetMachineImpl provides standard regalloc passes for most
  /// targets.
  void addOptimizedRegAlloc(PassManagerWrapper &PMW) const;

  /// Add passes that optimize machine instructions after register allocation.
  void addMachineLateOptimization(PassManagerWrapper &PMW) const;

  /// addGCPasses - Add late codegen passes that analyze code for garbage
  /// collection. This should return true if GC info should be printed after
  /// these passes.
  void addGCPasses(PassManagerWrapper &PMW) const {}

  /// Add standard basic block placement passes.
  void addBlockPlacement(PassManagerWrapper &PMW) const;

  void addPostBBSections(PassManagerWrapper &PMW) const {}

  using CreateMCStreamer =
      std::function<Expected<std::unique_ptr<MCStreamer>>(MCContext &)>;
  void addAsmPrinter(PassManagerWrapper &PMW, CreateMCStreamer) const {
    llvm_unreachable("addAsmPrinter is not overridden");
  }

  /// Utilities for targets to add passes to the pass manager.
  ///

  /// createTargetRegisterAllocator - Create the register allocator pass for
  /// this target at the current optimization level.
  void addTargetRegisterAllocator(PassManagerWrapper &PMW,
                                  bool Optimized) const;

  /// addMachinePasses helper to create the target-selected or overriden
  /// regalloc pass.
  void addRegAllocPass(PassManagerWrapper &PMW, bool Optimized) const;

  /// Add core register alloator passes which do the actual register assignment
  /// and rewriting. \returns true if any passes were added.
  Error addRegAssignmentFast(PassManagerWrapper &PMW) const;
  Error addRegAssignmentOptimized(PassManagerWrapper &PMWM) const;

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

  PassManagerWrapper PMW(MPM);

  addModulePass(RequireAnalysisPass<MachineModuleAnalysis, Module>(), PMW,
                /*Force=*/true);
  addModulePass(RequireAnalysisPass<ProfileSummaryAnalysis, Module>(), PMW,
                /*Force=*/true);
  addModulePass(RequireAnalysisPass<CollectorMetadataAnalysis, Module>(), PMW,
                /*Force=*/true);
  addModulePass(RequireAnalysisPass<RuntimeLibraryAnalysis, Module>(), PMW,
                /*Force=*/true);
  addISelPasses(PMW);
  flushFPMsToMPM(PMW);

  if (PrintMIR)
    addModulePass(PrintMIRPreparePass(Out), PMW, /*Force=*/true);

  if (auto Err = addCoreISelPasses(PMW))
    return std::move(Err);

  if (auto Err = derived().addMachinePasses(PMW))
    return std::move(Err);

  if (!Opt.DisableVerify)
    addMachineFunctionPass(MachineVerifierPass(), PMW);

  if (PrintAsm) {
    derived().addAsmPrinter(
        PMW, [this, &Out, DwoOut, FileType](MCContext &Ctx) {
          return this->TM.createMCStreamer(Out, DwoOut, FileType, Ctx);
        });
  }

  if (PrintMIR)
    addMachineFunctionPass(PrintMIRPass(Out), PMW, /*Force=*/true);

  flushFPMsToMPM(PMW, /*FreeMachineFunctions=*/true);

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
    PassManagerWrapper &PMW) const {
  derived().addGlobalMergePass(PMW);
  if (TM.useEmulatedTLS())
    addModulePass(LowerEmuTLSPass(), PMW);

  addModulePass(PreISelIntrinsicLoweringPass(&TM), PMW);
  addFunctionPass(ExpandIRInstsPass(TM, getOptLevel()), PMW);

  derived().addIRPasses(PMW);
  derived().addCodeGenPrepare(PMW);
  addPassesToHandleExceptions(PMW);
  derived().addISelPrepare(PMW);
}

/// Add common target configurable passes that perform LLVM IR to IR transforms
/// following machine independent optimization.
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addIRPasses(
    PassManagerWrapper &PMW) const {
  // Before running any passes, run the verifier to determine if the input
  // coming from the front-end and/or optimizer is valid.
  if (!Opt.DisableVerify)
    addFunctionPass(VerifierPass(), PMW, /*Force=*/true);

  // Run loop strength reduction before anything else.
  if (getOptLevel() != CodeGenOptLevel::None && !Opt.DisableLSR) {
    LoopPassManager LPM;
    LPM.addPass(CanonicalizeFreezeInLoopsPass());
    LPM.addPass(LoopStrengthReducePass());
    if (Opt.EnableLoopTermFold)
      LPM.addPass(LoopTermFoldPass());
    addFunctionPass(createFunctionToLoopPassAdaptor(std::move(LPM),
                                                    /*UseMemorySSA=*/true),
                    PMW);
  }

  if (getOptLevel() != CodeGenOptLevel::None) {
    // The MergeICmpsPass tries to create memcmp calls by grouping sequences of
    // loads and compares. ExpandMemCmpPass then tries to expand those calls
    // into optimally-sized loads and compares. The transforms are enabled by a
    // target lowering hook.
    if (!Opt.DisableMergeICmps)
      addFunctionPass(MergeICmpsPass(), PMW);
    addFunctionPass(ExpandMemCmpPass(TM), PMW);
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
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addPassesToHandleExceptions(
    PassManagerWrapper &PMW) const {
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
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addCodeGenPrepare(
    PassManagerWrapper &PMW) const {
  if (getOptLevel() != CodeGenOptLevel::None && !Opt.DisableCGP)
    addFunctionPass(CodeGenPreparePass(TM), PMW);
  // TODO: Default ctor'd RewriteSymbolPass is no-op.
  // addPass(RewriteSymbolPass());
}

/// Add common passes that perform LLVM IR to IR transforms in preparation for
/// instruction selection.
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addISelPrepare(
    PassManagerWrapper &PMW) const {
  derived().addPreISel(PMW);

  if (Opt.RequiresCodeGenSCCOrder && !AddInCGSCCOrder)
    requireCGSCCOrder(PMW);

  if (getOptLevel() != CodeGenOptLevel::None)
    addFunctionPass(ObjCARCContractPass(), PMW);

  addFunctionPass(CallBrPreparePass(), PMW);
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

template <typename Derived, typename TargetMachineT>
Error CodeGenPassBuilder<Derived, TargetMachineT>::addCoreISelPasses(
    PassManagerWrapper &PMW) const {
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
    if (auto Err = derived().addIRTranslator(PMW))
      return std::move(Err);

    derived().addPreLegalizeMachineIR(PMW);

    if (auto Err = derived().addLegalizeMachineIR(PMW))
      return std::move(Err);

    // Before running the register bank selector, ask the target if it
    // wants to run some passes.
    derived().addPreRegBankSelect(PMW);

    if (auto Err = derived().addRegBankSelect(PMW))
      return std::move(Err);

    derived().addPreGlobalInstructionSelect(PMW);

    if (auto Err = derived().addGlobalInstructionSelect(PMW))
      return std::move(Err);

    // Pass to reset the MachineFunction if the ISel failed.
    addMachineFunctionPass(
        ResetMachineFunctionPass(reportDiagnosticWhenGlobalISelFallback(),
                                 isGlobalISelAbortEnabled()),
        PMW);

    // Provide a fallback path when we do not want to abort on
    // not-yet-supported input.
    if (!isGlobalISelAbortEnabled())
      if (auto Err = derived().addInstSelector(PMW))
        return std::move(Err);

  } else if (auto Err = derived().addInstSelector(PMW))
    return std::move(Err);

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
    PassManagerWrapper &PMW) const {
  // Add passes that optimize machine instructions in SSA form.
  if (getOptLevel() != CodeGenOptLevel::None) {
    derived().addMachineSSAOptimization(PMW);
  } else {
    // If the target requests it, assign local variables to stack slots relative
    // to one another and simplify frame index references where possible.
    addMachineFunctionPass(LocalStackSlotAllocationPass(), PMW);
  }

  if (TM.Options.EnableIPRA) {
    flushFPMsToMPM(PMW);
    addModulePass(RequireAnalysisPass<PhysicalRegisterUsageAnalysis, Module>(),
                  PMW);
    addMachineFunctionPass(RegUsageInfoPropagationPass(), PMW);
  }
  // Run pre-ra passes.
  derived().addPreRegAlloc(PMW);

  // Run register allocation and passes that are tightly coupled with it,
  // including phi elimination and scheduling.
  if (*Opt.OptimizeRegAlloc) {
    derived().addOptimizedRegAlloc(PMW);
  } else {
    if (auto Err = derived().addFastRegAlloc(PMW))
      return Err;
  }

  // Run post-ra passes.
  derived().addPostRegAlloc(PMW);

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
    derived().addMachineLateOptimization(PMW);

  // Expand pseudo instructions before second scheduling pass.
  addMachineFunctionPass(ExpandPostRAPseudosPass(), PMW);

  // Run pre-sched2 passes.
  derived().addPreSched2(PMW);

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
  derived().addGCPasses(PMW);

  // Basic block placement.
  if (getOptLevel() != CodeGenOptLevel::None)
    derived().addBlockPlacement(PMW);

  // Insert before XRay Instrumentation.
  addMachineFunctionPass(FEntryInserterPass(), PMW);

  addMachineFunctionPass(XRayInstrumentationPass(), PMW);
  addMachineFunctionPass(PatchableFunctionPass(), PMW);

  derived().addPreEmitPass(PMW);

  if (TM.Options.EnableIPRA) {
    // Collect register usage information and produce a register mask of
    // clobbered registers, to be used to optimize call sites.
    flushFPMsToMPM(PMW);
    addModulePass(RequireAnalysisPass<PhysicalRegisterUsageAnalysis, Module>(),
                  PMW);
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
      LiveDebugValuesPass(
          getTM<TargetMachine>().Options.ShouldEmitDebugEntryValues()),
      PMW);
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

  derived().addPostBBSections(PMW);

  addMachineFunctionPass(StackFrameLayoutAnalysisPass(), PMW);

  // Add passes that directly emit MI after all other MI passes.
  derived().addPreEmitPass2(PMW);

  return Error::success();
}

/// Add passes that optimize machine instructions in SSA form.
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addMachineSSAOptimization(
    PassManagerWrapper &PMW) const {
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
  derived().addILPOpts(PMW);

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
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addTargetRegisterAllocator(
    PassManagerWrapper &PMW, bool Optimized) const {
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
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addRegAllocPass(
    PassManagerWrapper &PMW, bool Optimized) const {
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
  derived().addTargetRegisterAllocator(PMW, Optimized);
}

template <typename Derived, typename TargetMachineT>
Error CodeGenPassBuilder<Derived, TargetMachineT>::addRegAssignmentFast(
    PassManagerWrapper &PMW) const {
  // TODO: Ensure allocator is default or fast.
  addRegAllocPass(PMW, false);
  return Error::success();
}

template <typename Derived, typename TargetMachineT>
Error CodeGenPassBuilder<Derived, TargetMachineT>::addRegAssignmentOptimized(
    PassManagerWrapper &PMW) const {
  // Add the selected register allocation pass.
  addRegAllocPass(PMW, true);

  // Allow targets to change the register assignments before rewriting.
  derived().addPreRewrite(PMW);

  // Finally rewrite virtual registers.
  addMachineFunctionPass(VirtRegRewriterPass(), PMW);
  // Perform stack slot coloring and post-ra machine LICM.
  //
  // FIXME: Re-enable coloring with register when it's capable of adding
  // kill markers.
  addMachineFunctionPass(StackSlotColoringPass(), PMW);

  return Error::success();
}

/// Add the minimum set of target-independent passes that are required for
/// register allocation. No coalescing or scheduling.
template <typename Derived, typename TargetMachineT>
Error CodeGenPassBuilder<Derived, TargetMachineT>::addFastRegAlloc(
    PassManagerWrapper &PMW) const {
  addMachineFunctionPass(PHIEliminationPass(), PMW);
  addMachineFunctionPass(TwoAddressInstructionPass(), PMW);
  return derived().addRegAssignmentFast(PMW);
}

/// Add standard target-independent passes that are tightly coupled with
/// optimized register allocation, including coalescing, machine instruction
/// scheduling, and register allocation itself.
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addOptimizedRegAlloc(
    PassManagerWrapper &PMW) const {
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

  if (auto E = derived().addRegAssignmentOptimized(PMW)) {
    // addRegAssignmentOptimized did not add a reg alloc pass, so do nothing.
    return;
  }

  addMachineFunctionPass(StackSlotColoringPass(), PMW);

  // Allow targets to expand pseudo instructions depending on the choice of
  // registers before MachineCopyPropagation.
  derived().addPostRewrite(PMW);

  // Copy propagate to forward register uses and try to eliminate COPYs that
  // were not coalesced.
  addMachineFunctionPass(MachineCopyPropagationPass(), PMW);

  // Run post-ra machine LICM to hoist reloads / remats.
  //
  // FIXME: can this move into MachineLateOptimization?
  addMachineFunctionPass(MachineLICMPass(), PMW);
}

//===---------------------------------------------------------------------===//
/// Post RegAlloc Pass Configuration
//===---------------------------------------------------------------------===//

/// Add passes that optimize machine instructions after register allocation.
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addMachineLateOptimization(
    PassManagerWrapper &PMW) const {
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
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addBlockPlacement(
    PassManagerWrapper &PMW) const {
  addMachineFunctionPass(MachineBlockPlacementPass(Opt.EnableTailMerge), PMW);
  // Run a separate pass to collect block placement statistics.
  if (Opt.EnableBlockPlacementStats)
    addMachineFunctionPass(MachineBlockPlacementStatsPass(), PMW);
}

} // namespace llvm

#endif // LLVM_PASSES_CODEGENPASSBUILDER_H
