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

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "llvm/Target/CGPassBuilderOption.h"
#include "llvm/Target/TargetMachine.h"
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

class PassManagerWrapper {
private:
  PassManagerWrapper(ModulePassManager &ModulePM) : MPM(ModulePM) {};

  ModulePassManager &MPM;
  FunctionPassManager FPM;
  MachineFunctionPassManager MFPM;

  friend class CodeGenPassBuilder;
};

/// This class provides access to building LLVM's passes.
///
/// Its members provide the baseline state available to passes during their
/// construction. The \c MachinePassRegistry.def file specifies how to construct
/// all of the built-in passes, and those may reference these members during
/// construction.
///
/// Targets customize the pipeline by deriving from this class and overriding
/// the virtual add* hooks below: the add%Stage hooks replace a whole stage of
/// the pipeline, while the addPre%Stage / addPost%Stage hooks inject passes
/// around one. See addMachinePasses for how they fit together.
///
/// Dispatch is virtual rather than templated on the derived builder so that the
/// target-independent pipeline is emitted once for the whole build instead of
/// once per target.
class LLVM_ABI CodeGenPassBuilder {
public:
  CodeGenPassBuilder(TargetMachine &TM, const CGPassBuilderOption &Opts,
                     PassInstrumentationCallbacks *PIC);
  CodeGenPassBuilder(const CodeGenPassBuilder &) = delete;
  CodeGenPassBuilder &operator=(const CodeGenPassBuilder &) = delete;
  virtual ~CodeGenPassBuilder();

  Error buildPipeline(ModulePassManager &MPM, ModuleAnalysisManager &MAM,
                      raw_pwrite_stream &Out, raw_pwrite_stream *DwoOut,
                      CodeGenFileType FileType, MCContext &Ctx);

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
                       bool Force = false, StringRef Name = PassT::name()) {
    static_assert(is_detected<is_function_pass_t, PassT>::value &&
                  "Only function passes are supported.");
    if (!Force && !runBeforeAdding(Name))
      return;
    PMW.FPM.addPass(std::forward<PassT>(Pass));
  }

  template <typename PassT>
  void addModulePass(PassT &&Pass, PassManagerWrapper &PMW, bool Force = false,
                     StringRef Name = PassT::name()) {
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
                              StringRef Name = PassT::name()) {
    static_assert(is_detected<is_machine_function_pass_t, PassT>::value &&
                  "Only machine function passes are supported.");

    if (!Force && !runBeforeAdding(Name))
      return;
    PMW.MFPM.addPass(std::forward<PassT>(Pass));
    for (auto &C : AfterCallbacks)
      C(Name, PMW.MFPM);
  }

  void flushFPMsToMPM(PassManagerWrapper &PMW,
                      bool FreeMachineFunctions = false);

  void requireCGSCCOrder(PassManagerWrapper &PMW) {
    assert(!AddInCGSCCOrder);
    assert(PMW.FPM.isEmpty() && PMW.MFPM.isEmpty() &&
           "Requiring CGSCC ordering requires flushing the current function "
           "pipelines to the MPM.");
    AddInCGSCCOrder = true;
  }

  void stopAddingInCGSCCOrder(PassManagerWrapper &PMW) {
    assert(AddInCGSCCOrder);
    assert(PMW.FPM.isEmpty() && PMW.MFPM.isEmpty() &&
           "Stopping CGSCC ordering requires flushing the current function "
           "pipelines to the MPM.");
    AddInCGSCCOrder = false;
  }

  TargetMachine &TM;
  CGPassBuilderOption Opt;
  PassInstrumentationCallbacks *PIC;

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
  virtual Error addInstSelector(PassManagerWrapper &PMW);

  /// Target can override this to add GlobalMergePass before all IR passes.
  virtual void addGlobalMergePass(PassManagerWrapper &PMW) {}

  /// Add passes that optimize instruction level parallelism for out-of-order
  /// targets. These passes are run while the machine code is still in SSA
  /// form, so they can use MachineTraceMetrics to control their heuristics.
  ///
  /// All passes added here should preserve the MachineDominatorTree,
  /// MachineLoopInfo, and MachineTraceMetrics analyses.
  virtual void addILPOpts(PassManagerWrapper &PMW) {}

  /// This method may be implemented by targets that want to run passes
  /// immediately before register allocation.
  virtual void addPreRegAlloc(PassManagerWrapper &PMW) {}

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
  virtual void addPreRewrite(PassManagerWrapper &PMW) {}

  /// Add passes to be run immediately after virtual registers are rewritten
  /// to physical registers.
  virtual void addPostRewrite(PassManagerWrapper &PMW) {}

  /// This method may be implemented by targets that want to run passes after
  /// register allocation pass pipeline but before prolog-epilog insertion.
  virtual void addPostRegAlloc(PassManagerWrapper &PMW) {}

  /// This method may be implemented by targets that want to run passes after
  /// prolog-epilog insertion and before the second instruction scheduling pass.
  virtual void addPreSched2(PassManagerWrapper &PMW) {}

  /// This pass may be implemented by targets that want to run passes
  /// immediately before machine code is emitted.
  virtual void addPreEmitPass(PassManagerWrapper &PMW) {}

  /// Targets may add passes immediately before machine code is emitted in this
  /// callback. This is called even later than `addPreEmitPass`.
  // FIXME: Rename `addPreEmitPass` to something more sensible given its actual
  // position and remove the `2` suffix here as this callback is what
  // `addPreEmitPass` *should* be but in reality isn't.
  virtual void addPreEmitPass2(PassManagerWrapper &PMW) {}

  /// {{@ For GlobalISel
  ///

  /// addPreISel - This method should add any "last minute" LLVM->LLVM
  /// passes (which are run just before instruction selector).
  virtual void addPreISel(PassManagerWrapper &PMW) {}

  /// This method should install an IR translator pass, which converts from
  /// LLVM code to machine instructions with possibly generic opcodes.
  virtual Error addIRTranslator(PassManagerWrapper &PMW);

  /// This method may be implemented by targets that want to run passes
  /// immediately before legalization.
  virtual void addPreLegalizeMachineIR(PassManagerWrapper &PMW) {}

  /// This method should install a legalize pass, which converts the instruction
  /// sequence into one that can be selected by the target.
  virtual Error addLegalizeMachineIR(PassManagerWrapper &PMW);

  /// This method may be implemented by targets that want to run passes
  /// immediately before the register bank selection.
  virtual void addPreRegBankSelect(PassManagerWrapper &PMW) {}

  /// This method should install a register bank selector pass, which
  /// assigns register banks to virtual registers without a register
  /// class or register banks.
  virtual Error addRegBankSelect(PassManagerWrapper &PMW);

  /// This method may be implemented by targets that want to run passes
  /// immediately before the (global) instruction selection.
  virtual void addPreGlobalInstructionSelect(PassManagerWrapper &PMW) {}

  /// This method should install a (global) instruction selector pass, which
  /// converts possibly generic instructions to fully target-specific
  /// instructions, thereby constraining all generic virtual registers to
  /// register classes.
  virtual Error addGlobalInstructionSelect(PassManagerWrapper &PMW);
  /// @}}

  /// High level function that adds all passes necessary to go from llvm IR
  /// representation to the MI representation.
  /// Adds IR based lowering and target specific optimization passes and finally
  /// the core instruction selection passes.
  void addISelPasses(PassManagerWrapper &PMW);

  /// Add the actual instruction selection passes. This does not include
  /// preparation passes on IR.
  Error addCoreISelPasses(PassManagerWrapper &PMW);

  /// Add the complete, standard set of LLVM CodeGen passes.
  /// Fully developed targets will not generally override this.
  virtual Error addMachinePasses(PassManagerWrapper &PMW);

  /// Add passes to lower exception handling for the code generator.
  void addPassesToHandleExceptions(PassManagerWrapper &PMW);

  /// Add common target configurable passes that perform LLVM IR to IR
  /// transforms following machine independent optimization.
  virtual void addIRPasses(PassManagerWrapper &PMW);

  /// Add pass to prepare the LLVM IR for code generation. This should be done
  /// before exception handling preparation passes.
  virtual void addCodeGenPrepare(PassManagerWrapper &PMW);

  /// Add common passes that perform LLVM IR to IR transforms in preparation for
  /// instruction selection.
  virtual void addISelPrepare(PassManagerWrapper &PMW);

  /// Methods with trivial inline returns are convenient points in the common
  /// codegen pass pipeline where targets may insert passes. Methods with
  /// out-of-line standard implementations are major CodeGen stages called by
  /// addMachinePasses. Some targets may override major stages when inserting
  /// passes is insufficient, but maintaining overriden stages is more work.
  ///

  /// addMachineSSAOptimization - Add standard passes that optimize machine
  /// instructions in SSA form.
  virtual void addMachineSSAOptimization(PassManagerWrapper &PMW);

  /// addFastRegAlloc - Add the minimum set of target-independent passes that
  /// are required for fast register allocation.
  virtual Error addFastRegAlloc(PassManagerWrapper &PMW);

  /// addOptimizedRegAlloc - Add passes related to register allocation.
  /// CodeGenTargetMachineImpl provides standard regalloc passes for most
  /// targets.
  virtual Error addOptimizedRegAlloc(PassManagerWrapper &PMW);

  /// Add passes that optimize machine instructions after register allocation.
  virtual void addMachineLateOptimization(PassManagerWrapper &PMW);

  /// addGCPasses - Add late codegen passes that analyze code for garbage
  /// collection. This should return true if GC info should be printed after
  /// these passes.
  virtual void addGCPasses(PassManagerWrapper &PMW) {}

  /// Add standard basic block placement passes.
  virtual void addBlockPlacement(PassManagerWrapper &PMW);

  virtual void addPostBBSections(PassManagerWrapper &PMW) {}

  virtual void addAsmPrinterBegin(PassManagerWrapper &PMW);

  virtual void addAsmPrinter(PassManagerWrapper &PMW);

  virtual void addAsmPrinterEnd(PassManagerWrapper &PMW);

  /// Utilities for targets to add passes to the pass manager.
  ///

  /// Create the register allocator pass for this target at the current
  /// optimization level.
  virtual void addTargetRegisterAllocator(PassManagerWrapper &PMW,
                                          bool Optimized);

  /// addMachinePasses helper to create the target-selected or overriden
  /// regalloc pass.
  void addRegAllocPass(PassManagerWrapper &PMW, bool Optimized);

  /// Add core register allocator passes which do the actual register assignment
  /// and rewriting. addRegAssignAndRewriteOptimized should return true if any
  /// passes were added.
  virtual Error addRegAssignAndRewriteFast(PassManagerWrapper &PMW);
  virtual Expected<bool>
  addRegAssignAndRewriteOptimized(PassManagerWrapper &PMW);

  /// Allow the target to disable a specific pass by default.
  /// Backend can declare unwanted passes in constructor.
  template <typename... PassTs> void disablePass() {
    BeforeCallbacks.emplace_back(
        [](StringRef Name) { return ((Name != PassTs::name()) && ...); });
  }

  /// Insert InsertedPass pass after TargetPass pass.
  /// Only machine function passes are supported.
  template <typename TargetPassT, typename InsertedPassT>
  void insertPass(InsertedPassT &&Pass) {
    AfterCallbacks.emplace_back(
        [&](StringRef Name, MachineFunctionPassManager &MFPM) mutable {
          if (Name == TargetPassT::name() &&
              runBeforeAdding(InsertedPassT::name())) {
            MFPM.addPass(std::forward<InsertedPassT>(Pass));
          }
        });
  }

private:
  bool runBeforeAdding(StringRef Name) {
    bool ShouldAdd = true;
    for (auto &C : BeforeCallbacks)
      ShouldAdd &= C(Name);
    return ShouldAdd;
  }

  void setStartStopPasses(const TargetPassConfig::StartStopInfo &Info);

  Error verifyStartStop(const TargetPassConfig::StartStopInfo &Info) const;

  SmallVector<llvm::unique_function<bool(StringRef)>, 4> BeforeCallbacks;
  SmallVector<
      llvm::unique_function<void(StringRef, MachineFunctionPassManager &)>, 4>
      AfterCallbacks;

  /// Helper variable for `-start-before/-start-after/-stop-before/-stop-after`
  bool Started = true;
  bool Stopped = true;
  bool AddInCGSCCOrder = false;
};

} // namespace llvm

#endif // LLVM_PASSES_CODEGENPASSBUILDER_H
