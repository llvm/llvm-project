//===- Construction of codegen pass pipelines ------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// Interfaces for registering analysis passes, producing common pass manager
/// configurations, and parsing of pass pipelines.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_CODEGENPASSBUILDER_H
#define LLVM_CODEGEN_CODEGENPASSBUILDER_H

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/CFLAndersAliasAnalysis.h"
#include "llvm/Analysis/CFLSteensAliasAnalysis.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/CodeGen/ExpandReductions.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/PreISelIntrinsicLowering.h"
#include "llvm/CodeGen/ReplaceWithVeclib.h"
#include "llvm/CodeGen/UnreachableBlockElim.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/CGPassBuilderOption.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/ConstantHoisting.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Scalar/LoopStrengthReduce.h"
#include "llvm/Transforms/Scalar/LowerConstantIntrinsics.h"
#include "llvm/Transforms/Scalar/MergeICmps.h"
#include "llvm/Transforms/Scalar/PartiallyInlineLibCalls.h"
#include "llvm/Transforms/Scalar/ScalarizeMaskedMemIntrin.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/EntryExitInstrumenter.h"
#include "llvm/Transforms/Utils/LowerInvoke.h"
#include <cassert>
#include <type_traits>
#include <utility>

namespace llvm {

// FIXME: Dummy target independent passes definitions that have not yet been
// ported to new pass manager. Once they do, remove these.
#define DUMMY_FUNCTION_PASS(NAME, PASS_NAME, CONSTRUCTOR)                      \
  struct PASS_NAME : public PassInfoMixin<PASS_NAME> {                         \
    template <typename... Ts> PASS_NAME(Ts &&...) {}                           \
    PreservedAnalyses run(Function &, FunctionAnalysisManager &) {             \
      return PreservedAnalyses::all();                                         \
    }                                                                          \
  };
#define DUMMY_MODULE_PASS(NAME, PASS_NAME, CONSTRUCTOR)                        \
  struct PASS_NAME : public PassInfoMixin<PASS_NAME> {                         \
    template <typename... Ts> PASS_NAME(Ts &&...) {}                           \
    PreservedAnalyses run(Module &, ModuleAnalysisManager &) {                 \
      return PreservedAnalyses::all();                                         \
    }                                                                          \
  };
#define DUMMY_MACHINE_MODULE_PASS(NAME, PASS_NAME, CONSTRUCTOR)                \
  struct PASS_NAME : public PassInfoMixin<PASS_NAME> {                         \
    template <typename... Ts> PASS_NAME(Ts &&...) {}                           \
    Error run(Module &, MachineFunctionAnalysisManager &) {                    \
      return Error::success();                                                 \
    }                                                                          \
    PreservedAnalyses run(MachineFunction &,                                   \
                          MachineFunctionAnalysisManager &) {                  \
      llvm_unreachable("this api is to make new PM api happy");                \
    }                                                                          \
    static AnalysisKey Key;                                                    \
  };
#define DUMMY_MACHINE_FUNCTION_PASS(NAME, PASS_NAME, CONSTRUCTOR)              \
  struct PASS_NAME : public PassInfoMixin<PASS_NAME> {                         \
    template <typename... Ts> PASS_NAME(Ts &&...) {}                           \
    PreservedAnalyses run(MachineFunction &,                                   \
                          MachineFunctionAnalysisManager &) {                  \
      return PreservedAnalyses::all();                                         \
    }                                                                          \
    static AnalysisKey Key;                                                    \
  };
#include "MachinePassRegistry.def"

/// This class provides access to building LLVM's passes.
///
/// Its members provide the baseline state available to passes during their
/// construction. The \c MachinePassRegistry.def file specifies how to construct
/// all of the built-in passes, and those may reference these members during
/// construction.
template <typename DerivedT> class CodeGenPassBuilder {
public:
  explicit CodeGenPassBuilder(LLVMTargetMachine &TM, CGPassBuilderOption Opts,
                              PassInstrumentationCallbacks *PIC)
      : TM(TM), Opt(Opts), PIC(PIC) {
    // Target could set CGPassBuilderOption::MISchedPostRA to true to achieve
    //     substitutePass(&PostRASchedulerID, &PostMachineSchedulerID)

    // Target should override TM.Options.EnableIPRA in their target-specific
    // LLVMTM ctor. See TargetMachine::setGlobalISel for example.
    if (Opt.EnableIPRA)
      TM.Options.EnableIPRA = *Opt.EnableIPRA;

    if (Opt.EnableGlobalISelAbort)
      TM.Options.GlobalISelAbort = *Opt.EnableGlobalISelAbort;

    if (!Opt.OptimizeRegAlloc)
      Opt.OptimizeRegAlloc = getOptLevel() != CodeGenOpt::None;
  }

  Error buildPipeline(ModulePassManager &MPM, MachineFunctionPassManager &MFPM,
                      raw_pwrite_stream &Out, raw_pwrite_stream *DwoOut,
                      CodeGenFileType FileType) const;

  void registerModuleAnalyses(ModuleAnalysisManager &) const;
  void registerFunctionAnalyses(FunctionAnalysisManager &) const;
  void registerMachineFunctionAnalyses(MachineFunctionAnalysisManager &) const;
  std::pair<StringRef, bool> getPassNameFromLegacyName(StringRef) const;

  void registerAnalyses(MachineFunctionAnalysisManager &MFAM) const {
    registerModuleAnalyses(*MFAM.MAM);
    registerFunctionAnalyses(*MFAM.FAM);
    registerMachineFunctionAnalyses(MFAM);
  }

  PassInstrumentationCallbacks *getPassInstrumentationCallbacks() const {
    return PIC;
  }

protected:
  template <typename PassT> using has_key_t = decltype(PassT::Key);

  template <typename PassT>
  using is_module_pass_t = decltype(std::declval<PassT &>().run(
      std::declval<Module &>(), std::declval<ModuleAnalysisManager &>()));

  template <typename PassT>
  using is_function_pass_t = decltype(std::declval<PassT &>().run(
      std::declval<Function &>(), std::declval<FunctionAnalysisManager &>()));

  // Function object to maintain state while adding codegen IR passes.
  class AddIRPass {
  public:
    AddIRPass(ModulePassManager &MPM, bool DebugPM, bool Check = true)
        : MPM(MPM) {
      if (Check)
        AddingFunctionPasses = false;
    }
    ~AddIRPass() {
      MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
    }

    // Add Function Pass
    template <typename PassT>
    std::enable_if_t<is_detected<is_function_pass_t, PassT>::value>
    operator()(PassT &&Pass) {
      if (AddingFunctionPasses && !*AddingFunctionPasses)
        AddingFunctionPasses = true;
      FPM.addPass(std::forward<PassT>(Pass));
    }

    // Add Module Pass
    template <typename PassT>
    std::enable_if_t<is_detected<is_module_pass_t, PassT>::value &&
                     !is_detected<is_function_pass_t, PassT>::value>
    operator()(PassT &&Pass) {
      assert((!AddingFunctionPasses || !*AddingFunctionPasses) &&
             "could not add module pass after adding function pass");
      MPM.addPass(std::forward<PassT>(Pass));
    }

  private:
    ModulePassManager &MPM;
    FunctionPassManager FPM;
    // The codegen IR pipeline are mostly function passes with the exceptions of
    // a few loop and module passes. `AddingFunctionPasses` make sures that
    // we could only add module passes at the beginning of the pipeline. Once
    // we begin adding function passes, we could no longer add module passes.
    // This special-casing introduces less adaptor passes. If we have the need
    // of adding module passes after function passes, we could change the
    // implementation to accommodate that.
    Optional<bool> AddingFunctionPasses;
  };

  // Function object to maintain state while adding codegen machine passes.
  class AddMachinePass {
  public:
    AddMachinePass(MachineFunctionPassManager &PM) : PM(PM) {}

    template <typename PassT> void operator()(PassT &&Pass) {
      static_assert(
          is_detected<has_key_t, PassT>::value,
          "Machine function pass must define a static member variable `Key`.");
      for (auto &C : BeforeCallbacks)
        if (!C(&PassT::Key))
          return;
      PM.addPass(std::forward<PassT>(Pass));
      for (auto &C : AfterCallbacks)
        C(&PassT::Key);
    }

    template <typename PassT> void insertPass(AnalysisKey *ID, PassT Pass) {
      AfterCallbacks.emplace_back(
          [this, ID, Pass = std::move(Pass)](AnalysisKey *PassID) {
            if (PassID == ID)
              this->PM.addPass(std::move(Pass));
          });
    }

    void disablePass(AnalysisKey *ID) {
      BeforeCallbacks.emplace_back(
          [ID](AnalysisKey *PassID) { return PassID != ID; });
    }

    MachineFunctionPassManager releasePM() { return std::move(PM); }

  private:
    MachineFunctionPassManager &PM;
    SmallVector<llvm::unique_function<bool(AnalysisKey *)>, 4> BeforeCallbacks;
    SmallVector<llvm::unique_function<void(AnalysisKey *)>, 4> AfterCallbacks;
  };

  LLVMTargetMachine &TM;
  CGPassBuilderOption Opt;
  PassInstrumentationCallbacks *PIC;

  /// Target override these hooks to parse target-specific analyses.
  void registerTargetAnalysis(ModuleAnalysisManager &) const {}
  void registerTargetAnalysis(FunctionAnalysisManager &) const {}
  void registerTargetAnalysis(MachineFunctionAnalysisManager &) const {}
  std::pair<StringRef, bool> getTargetPassNameFromLegacyName(StringRef) const {
    return {"", false};
  }

  template <typename TMC> TMC &getTM() const { return static_cast<TMC &>(TM); }
  CodeGenOpt::Level getOptLevel() const { return TM.getOptLevel(); }

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
  Error addInstSelector(AddMachinePass &) const {
    return make_error<StringError>("addInstSelector is not overridden",
                                   inconvertibleErrorCode());
  }

  /// Add passes that optimize instruction level parallelism for out-of-order
  /// targets. These passes are run while the machine code is still in SSA
  /// form, so they can use MachineTraceMetrics to control their heuristics.
  ///
  /// All passes added here should preserve the MachineDominatorTree,
  /// MachineLoopInfo, and MachineTraceMetrics analyses.
  void addILPOpts(AddMachinePass &) const {}

  /// This method may be implemented by targets that want to run passes
  /// immediately before register allocation.
  void addPreRegAlloc(AddMachinePass &) const {}

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
  /// be honored. This is also not generally used for the the fast variant,
  /// where the allocation and rewriting are done in one pass.
  void addPreRewrite(AddMachinePass &) const {}

  /// Add passes to be run immediately after virtual registers are rewritten
  /// to physical registers.
  void addPostRewrite(AddMachinePass &) const {}

  /// This method may be implemented by targets that want to run passes after
  /// register allocation pass pipeline but before prolog-epilog insertion.
  void addPostRegAlloc(AddMachinePass &) const {}

  /// This method may be implemented by targets that want to run passes after
  /// prolog-epilog insertion and before the second instruction scheduling pass.
  void addPreSched2(AddMachinePass &) const {}

  /// This pass may be implemented by targets that want to run passes
  /// immediately before machine code is emitted.
  void addPreEmitPass(AddMachinePass &) const {}

  /// Targets may add passes immediately before machine code is emitted in this
  /// callback. This is called even later than `addPreEmitPass`.
  // FIXME: Rename `addPreEmitPass` to something more sensible given its actual
  // position and remove the `2` suffix here as this callback is what
  // `addPreEmitPass` *should* be but in reality isn't.
  void addPreEmitPass2(AddMachinePass &) const {}

  /// {{@ For GlobalISel
  ///

  /// addPreISel - This method should add any "last minute" LLVM->LLVM
  /// passes (which are run just before instruction selector).
  void addPreISel(AddIRPass &) const {
    llvm_unreachable("addPreISel is not overridden");
  }

  /// This method should install an IR translator pass, which converts from
  /// LLVM code to machine instructions with possibly generic opcodes.
  Error addIRTranslator(AddMachinePass &) const {
    return make_error<StringError>("addIRTranslator is not overridden",
                                   inconvertibleErrorCode());
  }

  /// This method may be implemented by targets that want to run passes
  /// immediately before legalization.
  void addPreLegalizeMachineIR(AddMachinePass &) const {}

  /// This method should install a legalize pass, which converts the instruction
  /// sequence into one that can be selected by the target.
  Error addLegalizeMachineIR(AddMachinePass &) const {
    return make_error<StringError>("addLegalizeMachineIR is not overridden",
                                   inconvertibleErrorCode());
  }

  /// This method may be implemented by targets that want to run passes
  /// immediately before the register bank selection.
  void addPreRegBankSelect(AddMachinePass &) const {}

  /// This method should install a register bank selector pass, which
  /// assigns register banks to virtual registers without a register
  /// class or register banks.
  Error addRegBankSelect(AddMachinePass &) const {
    return make_error<StringError>("addRegBankSelect is not overridden",
                                   inconvertibleErrorCode());
  }

  /// This method may be implemented by targets that want to run passes
  /// immediately before the (global) instruction selection.
  void addPreGlobalInstructionSelect(AddMachinePass &) const {}

  /// This method should install a (global) instruction selector pass, which
  /// converts possibly generic instructions to fully target-specific
  /// instructions, thereby constraining all generic virtual registers to
  /// register classes.
  Error addGlobalInstructionSelect(AddMachinePass &) const {
    return make_error<StringError>(
        "addGlobalInstructionSelect is not overridden",
        inconvertibleErrorCode());
  }
  /// @}}

  /// High level function that adds all passes necessary to go from llvm IR
  /// representation to the MI representation.
  /// Adds IR based lowering and target specific optimization passes and finally
  /// the core instruction selection passes.
  void addISelPasses(AddIRPass &) const;

  /// Add the actual instruction selection passes. This does not include
  /// preparation passes on IR.
  Error addCoreISelPasses(AddMachinePass &) const;

  /// Add the complete, standard set of LLVM CodeGen passes.
  /// Fully developed targets will not generally override this.
  Error addMachinePasses(AddMachinePass &) const;

  /// Add passes to lower exception handling for the code generator.
  void addPassesToHandleExceptions(AddIRPass &) const;

  /// Add common target configurable passes that perform LLVM IR to IR
  /// transforms following machine independent optimization.
  void addIRPasses(AddIRPass &) const;

  /// Add pass to prepare the LLVM IR for code generation. This should be done
  /// before exception handling preparation passes.
  void addCodeGenPrepare(AddIRPass &) const;

  /// Add common passes that perform LLVM IR to IR transforms in preparation for
  /// instruction selection.
  void addISelPrepare(AddIRPass &) const;

  /// Methods with trivial inline returns are convenient points in the common
  /// codegen pass pipeline where targets may insert passes. Methods with
  /// out-of-line standard implementations are major CodeGen stages called by
  /// addMachinePasses. Some targets may override major stages when inserting
  /// passes is insufficient, but maintaining overriden stages is more work.
  ///

  /// addMachineSSAOptimization - Add standard passes that optimize machine
  /// instructions in SSA form.
  void addMachineSSAOptimization(AddMachinePass &) const;

  /// addFastRegAlloc - Add the minimum set of target-independent passes that
  /// are required for fast register allocation.
  Error addFastRegAlloc(AddMachinePass &) const;

  /// addOptimizedRegAlloc - Add passes related to register allocation.
  /// LLVMTargetMachine provides standard regalloc passes for most targets.
  void addOptimizedRegAlloc(AddMachinePass &) const;

  /// Add passes that optimize machine instructions after register allocation.
  void addMachineLateOptimization(AddMachinePass &) const;

  /// addGCPasses - Add late codegen passes that analyze code for garbage
  /// collection. This should return true if GC info should be printed after
  /// these passes.
  void addGCPasses(AddMachinePass &) const {}

  /// Add standard basic block placement passes.
  void addBlockPlacement(AddMachinePass &) const;

  using CreateMCStreamer =
      std::function<Expected<std::unique_ptr<MCStreamer>>(MCContext &)>;
  void addAsmPrinter(AddMachinePass &, CreateMCStreamer) const {
    llvm_unreachable("addAsmPrinter is not overridden");
  }

  /// Utilities for targets to add passes to the pass manager.
  ///

  /// createTargetRegisterAllocator - Create the register allocator pass for
  /// this target at the current optimization level.
  void addTargetRegisterAllocator(AddMachinePass &, bool Optimized) const;

  /// addMachinePasses helper to create the target-selected or overriden
  /// regalloc pass.
  void addRegAllocPass(AddMachinePass &, bool Optimized) const;

  /// Add core register alloator passes which do the actual register assignment
  /// and rewriting. \returns true if any passes were added.
  Error addRegAssignmentFast(AddMachinePass &) const;
  Error addRegAssignmentOptimized(AddMachinePass &) const;

private:
  DerivedT &derived() { return static_cast<DerivedT &>(*this); }
  const DerivedT &derived() const {
    return static_cast<const DerivedT &>(*this);
  }
};

template <typename Derived>
Error CodeGenPassBuilder<Derived>::buildPipeline(
    ModulePassManager &MPM, MachineFunctionPassManager &MFPM,
    raw_pwrite_stream &Out, raw_pwrite_stream *DwoOut,
    CodeGenFileType FileType) const {
  AddIRPass addIRPass(MPM, Opt.DebugPM);
  addISelPasses(addIRPass);

  AddMachinePass addPass(MFPM);
  if (auto Err = addCoreISelPasses(addPass))
    return std::move(Err);

  if (auto Err = derived().addMachinePasses(addPass))
    return std::move(Err);

  derived().addAsmPrinter(
      addPass, [this, &Out, DwoOut, FileType](MCContext &Ctx) {
        return this->TM.createMCStreamer(Out, DwoOut, FileType, Ctx);
      });

  addPass(FreeMachineFunctionPass());
  return Error::success();
}

static inline AAManager registerAAAnalyses(CFLAAType UseCFLAA) {
  AAManager AA;

  // The order in which these are registered determines their priority when
  // being queried.

  switch (UseCFLAA) {
  case CFLAAType::Steensgaard:
    AA.registerFunctionAnalysis<CFLSteensAA>();
    break;
  case CFLAAType::Andersen:
    AA.registerFunctionAnalysis<CFLAndersAA>();
    break;
  case CFLAAType::Both:
    AA.registerFunctionAnalysis<CFLAndersAA>();
    AA.registerFunctionAnalysis<CFLSteensAA>();
    break;
  default:
    break;
  }

  // Basic AliasAnalysis support.
  // Add TypeBasedAliasAnalysis before BasicAliasAnalysis so that
  // BasicAliasAnalysis wins if they disagree. This is intended to help
  // support "obvious" type-punning idioms.
  AA.registerFunctionAnalysis<TypeBasedAA>();
  AA.registerFunctionAnalysis<ScopedNoAliasAA>();
  AA.registerFunctionAnalysis<BasicAA>();

  return AA;
}

template <typename Derived>
void CodeGenPassBuilder<Derived>::registerModuleAnalyses(
    ModuleAnalysisManager &MAM) const {
#define MODULE_ANALYSIS(NAME, PASS_NAME, CONSTRUCTOR)                          \
  MAM.registerPass([&] { return PASS_NAME CONSTRUCTOR; });
#include "MachinePassRegistry.def"
  derived().registerTargetAnalysis(MAM);
}

template <typename Derived>
void CodeGenPassBuilder<Derived>::registerFunctionAnalyses(
    FunctionAnalysisManager &FAM) const {
  FAM.registerPass([this] { return registerAAAnalyses(this->Opt.UseCFLAA); });

#define FUNCTION_ANALYSIS(NAME, PASS_NAME, CONSTRUCTOR)                        \
  FAM.registerPass([&] { return PASS_NAME CONSTRUCTOR; });
#include "MachinePassRegistry.def"
  derived().registerTargetAnalysis(FAM);
}

template <typename Derived>
void CodeGenPassBuilder<Derived>::registerMachineFunctionAnalyses(
    MachineFunctionAnalysisManager &MFAM) const {
#define MACHINE_FUNCTION_ANALYSIS(NAME, PASS_NAME, CONSTRUCTOR)                \
  MFAM.registerPass([&] { return PASS_NAME CONSTRUCTOR; });
#include "MachinePassRegistry.def"
  derived().registerTargetAnalysis(MFAM);
}

// FIXME: For new PM, use pass name directly in commandline seems good.
// Translate stringfied pass name to its old commandline name. Returns the
// matching legacy name and a boolean value indicating if the pass is a machine
// pass.
template <typename Derived>
std::pair<StringRef, bool>
CodeGenPassBuilder<Derived>::getPassNameFromLegacyName(StringRef Name) const {
  std::pair<StringRef, bool> Ret;
  if (Name.empty())
    return Ret;

#define FUNCTION_PASS(NAME, PASS_NAME, CONSTRUCTOR)                            \
  if (Name == NAME)                                                            \
    Ret = {#PASS_NAME, false};
#define DUMMY_FUNCTION_PASS(NAME, PASS_NAME, CONSTRUCTOR)                      \
  if (Name == NAME)                                                            \
    Ret = {#PASS_NAME, false};
#define MODULE_PASS(NAME, PASS_NAME, CONSTRUCTOR)                              \
  if (Name == NAME)                                                            \
    Ret = {#PASS_NAME, false};
#define DUMMY_MODULE_PASS(NAME, PASS_NAME, CONSTRUCTOR)                        \
  if (Name == NAME)                                                            \
    Ret = {#PASS_NAME, false};
#define MACHINE_MODULE_PASS(NAME, PASS_NAME, CONSTRUCTOR)                      \
  if (Name == NAME)                                                            \
    Ret = {#PASS_NAME, true};
#define DUMMY_MACHINE_MODULE_PASS(NAME, PASS_NAME, CONSTRUCTOR)                \
  if (Name == NAME)                                                            \
    Ret = {#PASS_NAME, true};
#define MACHINE_FUNCTION_PASS(NAME, PASS_NAME, CONSTRUCTOR)                    \
  if (Name == NAME)                                                            \
    Ret = {#PASS_NAME, true};
#define DUMMY_MACHINE_FUNCTION_PASS(NAME, PASS_NAME, CONSTRUCTOR)              \
  if (Name == NAME)                                                            \
    Ret = {#PASS_NAME, true};
#include "llvm/CodeGen/MachinePassRegistry.def"

  if (Ret.first.empty())
    Ret = derived().getTargetPassNameFromLegacyName(Name);

  if (Ret.first.empty())
    report_fatal_error(Twine('\"') + Twine(Name) +
                       Twine("\" pass could not be found."));

  return Ret;
}

template <typename Derived>
void CodeGenPassBuilder<Derived>::addISelPasses(AddIRPass &addPass) const {
  if (TM.useEmulatedTLS())
    addPass(LowerEmuTLSPass());

  addPass(PreISelIntrinsicLoweringPass());

  derived().addIRPasses(addPass);
  derived().addCodeGenPrepare(addPass);
  addPassesToHandleExceptions(addPass);
  derived().addISelPrepare(addPass);
}

/// Add common target configurable passes that perform LLVM IR to IR transforms
/// following machine independent optimization.
template <typename Derived>
void CodeGenPassBuilder<Derived>::addIRPasses(AddIRPass &addPass) const {
  // Before running any passes, run the verifier to determine if the input
  // coming from the front-end and/or optimizer is valid.
  if (!Opt.DisableVerify)
    addPass(VerifierPass());

  // Run loop strength reduction before anything else.
  if (getOptLevel() != CodeGenOpt::None && !Opt.DisableLSR) {
    addPass(createFunctionToLoopPassAdaptor(
        LoopStrengthReducePass(), /*UseMemorySSA*/ true, Opt.DebugPM));
    // FIXME: use -stop-after so we could remove PrintLSR
    if (Opt.PrintLSR)
      addPass(PrintFunctionPass(dbgs(), "\n\n*** Code after LSR ***\n"));
  }

  if (getOptLevel() != CodeGenOpt::None) {
    // The MergeICmpsPass tries to create memcmp calls by grouping sequences of
    // loads and compares. ExpandMemCmpPass then tries to expand those calls
    // into optimally-sized loads and compares. The transforms are enabled by a
    // target lowering hook.
    if (!Opt.DisableMergeICmps)
      addPass(MergeICmpsPass());
    addPass(ExpandMemCmpPass());
  }

  // Run GC lowering passes for builtin collectors
  // TODO: add a pass insertion point here
  addPass(GCLoweringPass());
  addPass(ShadowStackGCLoweringPass());
  addPass(LowerConstantIntrinsicsPass());

  // Make sure that no unreachable blocks are instruction selected.
  addPass(UnreachableBlockElimPass());

  // Prepare expensive constants for SelectionDAG.
  if (getOptLevel() != CodeGenOpt::None && !Opt.DisableConstantHoisting)
    addPass(ConstantHoistingPass());

  // Replace calls to LLVM intrinsics (e.g., exp, log) operating on vector
  // operands with calls to the corresponding functions in a vector library.
  if (getOptLevel() != CodeGenOpt::None)
    addPass(ReplaceWithVeclib());

  if (getOptLevel() != CodeGenOpt::None && !Opt.DisablePartialLibcallInlining)
    addPass(PartiallyInlineLibCallsPass());

  // Instrument function entry and exit, e.g. with calls to mcount().
  addPass(EntryExitInstrumenterPass(/*PostInlining=*/true));

  // Add scalarization of target's unsupported masked memory intrinsics pass.
  // the unsupported intrinsic will be replaced with a chain of basic blocks,
  // that stores/loads element one-by-one if the appropriate mask bit is set.
  addPass(ScalarizeMaskedMemIntrinPass());

  // Expand reduction intrinsics into shuffle sequences if the target wants to.
  addPass(ExpandReductionsPass());
}

/// Turn exception handling constructs into something the code generators can
/// handle.
template <typename Derived>
void CodeGenPassBuilder<Derived>::addPassesToHandleExceptions(
    AddIRPass &addPass) const {
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
    addPass(SjLjEHPreparePass());
    LLVM_FALLTHROUGH;
  case ExceptionHandling::DwarfCFI:
  case ExceptionHandling::ARM:
  case ExceptionHandling::AIX:
    addPass(DwarfEHPass(getOptLevel()));
    break;
  case ExceptionHandling::WinEH:
    // We support using both GCC-style and MSVC-style exceptions on Windows, so
    // add both preparation passes. Each pass will only actually run if it
    // recognizes the personality function.
    addPass(WinEHPass());
    addPass(DwarfEHPass(getOptLevel()));
    break;
  case ExceptionHandling::Wasm:
    // Wasm EH uses Windows EH instructions, but it does not need to demote PHIs
    // on catchpads and cleanuppads because it does not outline them into
    // funclets. Catchswitch blocks are not lowered in SelectionDAG, so we
    // should remove PHIs there.
    addPass(WinEHPass(/*DemoteCatchSwitchPHIOnly=*/false));
    addPass(WasmEHPass());
    break;
  case ExceptionHandling::None:
    addPass(LowerInvokePass());

    // The lower invoke pass may create unreachable code. Remove it.
    addPass(UnreachableBlockElimPass());
    break;
  }
}

/// Add pass to prepare the LLVM IR for code generation. This should be done
/// before exception handling preparation passes.
template <typename Derived>
void CodeGenPassBuilder<Derived>::addCodeGenPrepare(AddIRPass &addPass) const {
  if (getOptLevel() != CodeGenOpt::None && !Opt.DisableCGP)
    addPass(CodeGenPreparePass());
  // TODO: Default ctor'd RewriteSymbolPass is no-op.
  // addPass(RewriteSymbolPass());
}

/// Add common passes that perform LLVM IR to IR transforms in preparation for
/// instruction selection.
template <typename Derived>
void CodeGenPassBuilder<Derived>::addISelPrepare(AddIRPass &addPass) const {
  derived().addPreISel(addPass);

  // Add both the safe stack and the stack protection passes: each of them will
  // only protect functions that have corresponding attributes.
  addPass(SafeStackPass());
  addPass(StackProtectorPass());

  if (Opt.PrintISelInput)
    addPass(PrintFunctionPass(dbgs(),
                              "\n\n*** Final LLVM Code input to ISel ***\n"));

  // All passes which modify the LLVM IR are now complete; run the verifier
  // to ensure that the IR is valid.
  if (!Opt.DisableVerify)
    addPass(VerifierPass());
}

template <typename Derived>
Error CodeGenPassBuilder<Derived>::addCoreISelPasses(
    AddMachinePass &addPass) const {
  // Enable FastISel with -fast-isel, but allow that to be overridden.
  TM.setO0WantsFastISel(Opt.EnableFastISelOption.getValueOr(true));

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
  else if (TM.getOptLevel() == CodeGenOpt::None && TM.getO0WantsFastISel())
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
    if (auto Err = derived().addIRTranslator(addPass))
      return std::move(Err);

    derived().addPreLegalizeMachineIR(addPass);

    if (auto Err = derived().addLegalizeMachineIR(addPass))
      return std::move(Err);

    // Before running the register bank selector, ask the target if it
    // wants to run some passes.
    derived().addPreRegBankSelect(addPass);

    if (auto Err = derived().addRegBankSelect(addPass))
      return std::move(Err);

    derived().addPreGlobalInstructionSelect(addPass);

    if (auto Err = derived().addGlobalInstructionSelect(addPass))
      return std::move(Err);

    // Pass to reset the MachineFunction if the ISel failed.
    addPass(ResetMachineFunctionPass(reportDiagnosticWhenGlobalISelFallback(),
                                     isGlobalISelAbortEnabled()));

    // Provide a fallback path when we do not want to abort on
    // not-yet-supported input.
    if (!isGlobalISelAbortEnabled())
      if (auto Err = derived().addInstSelector(addPass))
        return std::move(Err);

  } else if (auto Err = derived().addInstSelector(addPass))
    return std::move(Err);

  // Expand pseudo-instructions emitted by ISel. Don't run the verifier before
  // FinalizeISel.
  addPass(FinalizeISelPass());

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
/// Any CodeGenPassBuilder<Derived>::addXX routine may be overriden by the
/// Target. The addPre/Post methods with empty header implementations allow
/// injecting target-specific fixups just before or after major stages.
/// Additionally, targets have the flexibility to change pass order within a
/// stage by overriding default implementation of add%Stage routines below. Each
/// technique has maintainability tradeoffs because alternate pass orders are
/// not well supported. addPre/Post works better if the target pass is easily
/// tied to a common pass. But if it has subtle dependencies on multiple passes,
/// the target should override the stage instead.
template <typename Derived>
Error CodeGenPassBuilder<Derived>::addMachinePasses(
    AddMachinePass &addPass) const {
  // Add passes that optimize machine instructions in SSA form.
  if (getOptLevel() != CodeGenOpt::None) {
    derived().addMachineSSAOptimization(addPass);
  } else {
    // If the target requests it, assign local variables to stack slots relative
    // to one another and simplify frame index references where possible.
    addPass(LocalStackSlotPass());
  }

  if (TM.Options.EnableIPRA)
    addPass(RegUsageInfoPropagationPass());

  // Run pre-ra passes.
  derived().addPreRegAlloc(addPass);

  // Run register allocation and passes that are tightly coupled with it,
  // including phi elimination and scheduling.
  if (*Opt.OptimizeRegAlloc) {
    derived().addOptimizedRegAlloc(addPass);
  } else {
    if (auto Err = derived().addFastRegAlloc(addPass))
      return Err;
  }

  // Run post-ra passes.
  derived().addPostRegAlloc(addPass);

  addPass(RemoveRedundantDebugValuesPass());

  // Insert prolog/epilog code.  Eliminate abstract frame index references...
  if (getOptLevel() != CodeGenOpt::None) {
    addPass(PostRAMachineSinkingPass());
    addPass(ShrinkWrapPass());
  }

  addPass(PrologEpilogInserterPass());

  /// Add passes that optimize machine instructions after register allocation.
  if (getOptLevel() != CodeGenOpt::None)
    derived().addMachineLateOptimization(addPass);

  // Expand pseudo instructions before second scheduling pass.
  addPass(ExpandPostRAPseudosPass());

  // Run pre-sched2 passes.
  derived().addPreSched2(addPass);

  if (Opt.EnableImplicitNullChecks)
    addPass(ImplicitNullChecksPass());

  // Second pass scheduler.
  // Let Target optionally insert this pass by itself at some other
  // point.
  if (getOptLevel() != CodeGenOpt::None &&
      !TM.targetSchedulesPostRAScheduling()) {
    if (Opt.MISchedPostRA)
      addPass(PostMachineSchedulerPass());
    else
      addPass(PostRASchedulerPass());
  }

  // GC
  derived().addGCPasses(addPass);

  // Basic block placement.
  if (getOptLevel() != CodeGenOpt::None)
    derived().addBlockPlacement(addPass);

  // Insert before XRay Instrumentation.
  addPass(FEntryInserterPass());

  addPass(XRayInstrumentationPass());
  addPass(PatchableFunctionPass());

  derived().addPreEmitPass(addPass);

  if (TM.Options.EnableIPRA)
    // Collect register usage information and produce a register mask of
    // clobbered registers, to be used to optimize call sites.
    addPass(RegUsageInfoCollectorPass());

  addPass(FuncletLayoutPass());

  addPass(StackMapLivenessPass());
  addPass(LiveDebugValuesPass());

  if (TM.Options.EnableMachineOutliner && getOptLevel() != CodeGenOpt::None &&
      Opt.EnableMachineOutliner != RunOutliner::NeverOutline) {
    bool RunOnAllFunctions =
        (Opt.EnableMachineOutliner == RunOutliner::AlwaysOutline);
    bool AddOutliner = RunOnAllFunctions || TM.Options.SupportsDefaultOutlining;
    if (AddOutliner)
      addPass(MachineOutlinerPass(RunOnAllFunctions));
  }

  // Add passes that directly emit MI after all other MI passes.
  derived().addPreEmitPass2(addPass);

  return Error::success();
}

/// Add passes that optimize machine instructions in SSA form.
template <typename Derived>
void CodeGenPassBuilder<Derived>::addMachineSSAOptimization(
    AddMachinePass &addPass) const {
  // Pre-ra tail duplication.
  addPass(EarlyTailDuplicatePass());

  // Optimize PHIs before DCE: removing dead PHI cycles may make more
  // instructions dead.
  addPass(OptimizePHIsPass());

  // This pass merges large allocas. StackSlotColoring is a different pass
  // which merges spill slots.
  addPass(StackColoringPass());

  // If the target requests it, assign local variables to stack slots relative
  // to one another and simplify frame index references where possible.
  addPass(LocalStackSlotPass());

  // With optimization, dead code should already be eliminated. However
  // there is one known exception: lowered code for arguments that are only
  // used by tail calls, where the tail calls reuse the incoming stack
  // arguments directly (see t11 in test/CodeGen/X86/sibcall.ll).
  addPass(DeadMachineInstructionElimPass());

  // Allow targets to insert passes that improve instruction level parallelism,
  // like if-conversion. Such passes will typically need dominator trees and
  // loop info, just like LICM and CSE below.
  derived().addILPOpts(addPass);

  addPass(EarlyMachineLICMPass());
  addPass(MachineCSEPass());

  addPass(MachineSinkingPass());

  addPass(PeepholeOptimizerPass());
  // Clean-up the dead code that may have been generated by peephole
  // rewriting.
  addPass(DeadMachineInstructionElimPass());
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
/// selection. But -regalloc=... always takes precedence.
template <typename Derived>
void CodeGenPassBuilder<Derived>::addTargetRegisterAllocator(
    AddMachinePass &addPass, bool Optimized) const {
  if (Optimized)
    addPass(RAGreedyPass());
  else
    addPass(RAFastPass());
}

/// Find and instantiate the register allocation pass requested by this target
/// at the current optimization level.  Different register allocators are
/// defined as separate passes because they may require different analysis.
template <typename Derived>
void CodeGenPassBuilder<Derived>::addRegAllocPass(AddMachinePass &addPass,
                                                  bool Optimized) const {
  if (Opt.RegAlloc == RegAllocType::Default)
    // With no -regalloc= override, ask the target for a regalloc pass.
    derived().addTargetRegisterAllocator(addPass, Optimized);
  else if (Opt.RegAlloc == RegAllocType::Basic)
    addPass(RABasicPass());
  else if (Opt.RegAlloc == RegAllocType::Fast)
    addPass(RAFastPass());
  else if (Opt.RegAlloc == RegAllocType::Greedy)
    addPass(RAGreedyPass());
  else if (Opt.RegAlloc == RegAllocType::PBQP)
    addPass(RAPBQPPass());
  else
    llvm_unreachable("unknonwn register allocator type");
}

template <typename Derived>
Error CodeGenPassBuilder<Derived>::addRegAssignmentFast(
    AddMachinePass &addPass) const {
  if (Opt.RegAlloc != RegAllocType::Default &&
      Opt.RegAlloc != RegAllocType::Fast)
    return make_error<StringError>(
        "Must use fast (default) register allocator for unoptimized regalloc.",
        inconvertibleErrorCode());

  addRegAllocPass(addPass, false);
  return Error::success();
}

template <typename Derived>
Error CodeGenPassBuilder<Derived>::addRegAssignmentOptimized(
    AddMachinePass &addPass) const {
  // Add the selected register allocation pass.
  addRegAllocPass(addPass, true);

  // Allow targets to change the register assignments before rewriting.
  derived().addPreRewrite(addPass);

  // Finally rewrite virtual registers.
  addPass(VirtRegRewriterPass());
  // Perform stack slot coloring and post-ra machine LICM.
  //
  // FIXME: Re-enable coloring with register when it's capable of adding
  // kill markers.
  addPass(StackSlotColoringPass());

  return Error::success();
}

/// Add the minimum set of target-independent passes that are required for
/// register allocation. No coalescing or scheduling.
template <typename Derived>
Error CodeGenPassBuilder<Derived>::addFastRegAlloc(
    AddMachinePass &addPass) const {
  addPass(PHIEliminationPass());
  addPass(TwoAddressInstructionPass());
  return derived().addRegAssignmentFast(addPass);
}

/// Add standard target-independent passes that are tightly coupled with
/// optimized register allocation, including coalescing, machine instruction
/// scheduling, and register allocation itself.
template <typename Derived>
void CodeGenPassBuilder<Derived>::addOptimizedRegAlloc(
    AddMachinePass &addPass) const {
  addPass(DetectDeadLanesPass());

  addPass(ProcessImplicitDefsPass());

  // Edge splitting is smarter with machine loop info.
  addPass(PHIEliminationPass());

  // Eventually, we want to run LiveIntervals before PHI elimination.
  if (Opt.EarlyLiveIntervals)
    addPass(LiveIntervalsPass());

  addPass(TwoAddressInstructionPass());
  addPass(RegisterCoalescerPass());

  // The machine scheduler may accidentally create disconnected components
  // when moving subregister definitions around, avoid this by splitting them to
  // separate vregs before. Splitting can also improve reg. allocation quality.
  addPass(RenameIndependentSubregsPass());

  // PreRA instruction scheduling.
  addPass(MachineSchedulerPass());

  if (derived().addRegAssignmentOptimized(addPass)) {
    // Allow targets to expand pseudo instructions depending on the choice of
    // registers before MachineCopyPropagation.
    derived().addPostRewrite(addPass);

    // Copy propagate to forward register uses and try to eliminate COPYs that
    // were not coalesced.
    addPass(MachineCopyPropagationPass());

    // Run post-ra machine LICM to hoist reloads / remats.
    //
    // FIXME: can this move into MachineLateOptimization?
    addPass(MachineLICMPass());
  }
}

//===---------------------------------------------------------------------===//
/// Post RegAlloc Pass Configuration
//===---------------------------------------------------------------------===//

/// Add passes that optimize machine instructions after register allocation.
template <typename Derived>
void CodeGenPassBuilder<Derived>::addMachineLateOptimization(
    AddMachinePass &addPass) const {
  // Branch folding must be run after regalloc and prolog/epilog insertion.
  addPass(BranchFolderPass());

  // Tail duplication.
  // Note that duplicating tail just increases code size and degrades
  // performance for targets that require Structured Control Flow.
  // In addition it can also make CFG irreducible. Thus we disable it.
  if (!TM.requiresStructuredCFG())
    addPass(TailDuplicatePass());

  // Copy propagation.
  addPass(MachineCopyPropagationPass());
}

/// Add standard basic block placement passes.
template <typename Derived>
void CodeGenPassBuilder<Derived>::addBlockPlacement(
    AddMachinePass &addPass) const {
  addPass(MachineBlockPlacementPass());
  // Run a separate pass to collect block placement statistics.
  if (Opt.EnableBlockPlacementStats)
    addPass(MachineBlockPlacementStatsPass());
}

} // namespace llvm

#endif // LLVM_CODEGEN_CODEGENPASSBUILDER_H
