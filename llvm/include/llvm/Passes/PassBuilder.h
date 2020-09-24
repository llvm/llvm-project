//===- Parsing, selection, and construction of pass pipelines --*- C++ -*--===//
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

#ifndef LLVM_PASSES_PASSBUILDER_H
#define LLVM_PASSES_PASSBUILDER_H

#include "llvm/ADT/Optional.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Error.h"
#include "llvm/Transforms/IPO/Inliner.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include <vector>

namespace llvm {
class StringRef;
class AAManager;
class TargetMachine;
class ModuleSummaryIndex;

/// A struct capturing PGO tunables.
struct PGOOptions {
  enum PGOAction { NoAction, IRInstr, IRUse, SampleUse };
  enum CSPGOAction { NoCSAction, CSIRInstr, CSIRUse };
  PGOOptions(std::string ProfileFile = "", std::string CSProfileGenFile = "",
             std::string ProfileRemappingFile = "", PGOAction Action = NoAction,
             CSPGOAction CSAction = NoCSAction, bool SamplePGOSupport = false)
      : ProfileFile(ProfileFile), CSProfileGenFile(CSProfileGenFile),
        ProfileRemappingFile(ProfileRemappingFile), Action(Action),
        CSAction(CSAction),
        SamplePGOSupport(SamplePGOSupport || Action == SampleUse) {
    // Note, we do allow ProfileFile.empty() for Action=IRUse LTO can
    // callback with IRUse action without ProfileFile.

    // If there is a CSAction, PGOAction cannot be IRInstr or SampleUse.
    assert(this->CSAction == NoCSAction ||
           (this->Action != IRInstr && this->Action != SampleUse));

    // For CSIRInstr, CSProfileGenFile also needs to be nonempty.
    assert(this->CSAction != CSIRInstr || !this->CSProfileGenFile.empty());

    // If CSAction is CSIRUse, PGOAction needs to be IRUse as they share
    // a profile.
    assert(this->CSAction != CSIRUse || this->Action == IRUse);

    // If neither Action nor CSAction, SamplePGOSupport needs to be true.
    assert(this->Action != NoAction || this->CSAction != NoCSAction ||
           this->SamplePGOSupport);
  }
  std::string ProfileFile;
  std::string CSProfileGenFile;
  std::string ProfileRemappingFile;
  PGOAction Action;
  CSPGOAction CSAction;
  bool SamplePGOSupport;
};

/// Tunable parameters for passes in the default pipelines.
class PipelineTuningOptions {
public:
  /// Constructor sets pipeline tuning defaults based on cl::opts. Each option
  /// can be set in the PassBuilder when using a LLVM as a library.
  PipelineTuningOptions();

  /// Tuning option to set loop interleaving on/off, set based on opt level.
  bool LoopInterleaving;

  /// Tuning option to enable/disable loop vectorization, set based on opt
  /// level.
  bool LoopVectorization;

  /// Tuning option to enable/disable slp loop vectorization, set based on opt
  /// level.
  bool SLPVectorization;

  /// Tuning option to enable/disable loop unrolling. Its default value is true.
  bool LoopUnrolling;

  /// Tuning option to forget all SCEV loops in LoopUnroll. Its default value
  /// is that of the flag: `-forget-scev-loop-unroll`.
  bool ForgetAllSCEVInLoopUnroll;

  /// Tuning option to enable/disable coroutine intrinsic lowering. Its default
  /// value is false. Frontends such as Clang may enable this conditionally. For
  /// example, Clang enables this option if the flags `-std=c++2a` or above, or
  /// `-fcoroutines-ts`, have been specified.
  bool Coroutines;

  /// Tuning option to cap the number of calls to retrive clobbering accesses in
  /// MemorySSA, in LICM.
  unsigned LicmMssaOptCap;

  /// Tuning option to disable promotion to scalars in LICM with MemorySSA, if
  /// the number of access is too large.
  unsigned LicmMssaNoAccForPromotionCap;

  /// Tuning option to enable/disable call graph profile. Its default value is
  /// that of the flag: `-enable-npm-call-graph-profile`.
  bool CallGraphProfile;
};

/// This class provides access to building LLVM's passes.
///
/// Its members provide the baseline state available to passes during their
/// construction. The \c PassRegistry.def file specifies how to construct all
/// of the built-in passes, and those may reference these members during
/// construction.
class PassBuilder {
  TargetMachine *TM;
  PipelineTuningOptions PTO;
  Optional<PGOOptions> PGOOpt;
  PassInstrumentationCallbacks *PIC;

public:
  /// A struct to capture parsed pass pipeline names.
  ///
  /// A pipeline is defined as a series of names, each of which may in itself
  /// recursively contain a nested pipeline. A name is either the name of a pass
  /// (e.g. "instcombine") or the name of a pipeline type (e.g. "cgscc"). If the
  /// name is the name of a pass, the InnerPipeline is empty, since passes
  /// cannot contain inner pipelines. See parsePassPipeline() for a more
  /// detailed description of the textual pipeline format.
  struct PipelineElement {
    StringRef Name;
    std::vector<PipelineElement> InnerPipeline;
  };

  /// ThinLTO phase.
  ///
  /// This enumerates the LLVM ThinLTO optimization phases.
  enum class ThinLTOPhase {
    /// No ThinLTO behavior needed.
    None,
    /// ThinLTO prelink (summary) phase.
    PreLink,
    /// ThinLTO postlink (backend compile) phase.
    PostLink
  };

  /// LLVM-provided high-level optimization levels.
  ///
  /// This enumerates the LLVM-provided high-level optimization levels. Each
  /// level has a specific goal and rationale.
  class OptimizationLevel final {
    unsigned SpeedLevel = 2;
    unsigned SizeLevel = 0;
    OptimizationLevel(unsigned SpeedLevel, unsigned SizeLevel)
        : SpeedLevel(SpeedLevel), SizeLevel(SizeLevel) {
      // Check that only valid combinations are passed.
      assert(SpeedLevel <= 3 &&
             "Optimization level for speed should be 0, 1, 2, or 3");
      assert(SizeLevel <= 2 &&
             "Optimization level for size should be 0, 1, or 2");
      assert((SizeLevel == 0 || SpeedLevel == 2) &&
             "Optimize for size should be encoded with speedup level == 2");
    }

  public:
    OptimizationLevel() = default;
    /// Disable as many optimizations as possible. This doesn't completely
    /// disable the optimizer in all cases, for example always_inline functions
    /// can be required to be inlined for correctness.
    static const OptimizationLevel O0;

    /// Optimize quickly without destroying debuggability.
    ///
    /// This level is tuned to produce a result from the optimizer as quickly
    /// as possible and to avoid destroying debuggability. This tends to result
    /// in a very good development mode where the compiled code will be
    /// immediately executed as part of testing. As a consequence, where
    /// possible, we would like to produce efficient-to-execute code, but not
    /// if it significantly slows down compilation or would prevent even basic
    /// debugging of the resulting binary.
    ///
    /// As an example, complex loop transformations such as versioning,
    /// vectorization, or fusion don't make sense here due to the degree to
    /// which the executed code differs from the source code, and the compile
    /// time cost.
    static const OptimizationLevel O1;
    /// Optimize for fast execution as much as possible without triggering
    /// significant incremental compile time or code size growth.
    ///
    /// The key idea is that optimizations at this level should "pay for
    /// themselves". So if an optimization increases compile time by 5% or
    /// increases code size by 5% for a particular benchmark, that benchmark
    /// should also be one which sees a 5% runtime improvement. If the compile
    /// time or code size penalties happen on average across a diverse range of
    /// LLVM users' benchmarks, then the improvements should as well.
    ///
    /// And no matter what, the compile time needs to not grow superlinearly
    /// with the size of input to LLVM so that users can control the runtime of
    /// the optimizer in this mode.
    ///
    /// This is expected to be a good default optimization level for the vast
    /// majority of users.
    static const OptimizationLevel O2;
    /// Optimize for fast execution as much as possible.
    ///
    /// This mode is significantly more aggressive in trading off compile time
    /// and code size to get execution time improvements. The core idea is that
    /// this mode should include any optimization that helps execution time on
    /// balance across a diverse collection of benchmarks, even if it increases
    /// code size or compile time for some benchmarks without corresponding
    /// improvements to execution time.
    ///
    /// Despite being willing to trade more compile time off to get improved
    /// execution time, this mode still tries to avoid superlinear growth in
    /// order to make even significantly slower compile times at least scale
    /// reasonably. This does not preclude very substantial constant factor
    /// costs though.
    static const OptimizationLevel O3;
    /// Similar to \c O2 but tries to optimize for small code size instead of
    /// fast execution without triggering significant incremental execution
    /// time slowdowns.
    ///
    /// The logic here is exactly the same as \c O2, but with code size and
    /// execution time metrics swapped.
    ///
    /// A consequence of the different core goal is that this should in general
    /// produce substantially smaller executables that still run in
    /// a reasonable amount of time.
    static const OptimizationLevel Os;
    /// A very specialized mode that will optimize for code size at any and all
    /// costs.
    ///
    /// This is useful primarily when there are absolute size limitations and
    /// any effort taken to reduce the size is worth it regardless of the
    /// execution time impact. You should expect this level to produce rather
    /// slow, but very small, code.
    static const OptimizationLevel Oz;

    bool isOptimizingForSpeed() const {
      return SizeLevel == 0 && SpeedLevel > 0;
    }

    bool isOptimizingForSize() const { return SizeLevel > 0; }

    bool operator==(const OptimizationLevel &Other) const {
      return SizeLevel == Other.SizeLevel && SpeedLevel == Other.SpeedLevel;
    }
    bool operator!=(const OptimizationLevel &Other) const {
      return SizeLevel != Other.SizeLevel || SpeedLevel != Other.SpeedLevel;
    }

    unsigned getSpeedupLevel() const { return SpeedLevel; }

    unsigned getSizeLevel() const { return SizeLevel; }
  };

  explicit PassBuilder(TargetMachine *TM = nullptr,
                       PipelineTuningOptions PTO = PipelineTuningOptions(),
                       Optional<PGOOptions> PGOOpt = None,
                       PassInstrumentationCallbacks *PIC = nullptr)
      : TM(TM), PTO(PTO), PGOOpt(PGOOpt), PIC(PIC) {}

  /// Cross register the analysis managers through their proxies.
  ///
  /// This is an interface that can be used to cross register each
  /// AnalysisManager with all the others analysis managers.
  void crossRegisterProxies(LoopAnalysisManager &LAM,
                            FunctionAnalysisManager &FAM,
                            CGSCCAnalysisManager &CGAM,
                            ModuleAnalysisManager &MAM);

  /// Registers all available module analysis passes.
  ///
  /// This is an interface that can be used to populate a \c
  /// ModuleAnalysisManager with all registered module analyses. Callers can
  /// still manually register any additional analyses. Callers can also
  /// pre-register analyses and this will not override those.
  void registerModuleAnalyses(ModuleAnalysisManager &MAM);

  /// Registers all available CGSCC analysis passes.
  ///
  /// This is an interface that can be used to populate a \c CGSCCAnalysisManager
  /// with all registered CGSCC analyses. Callers can still manually register any
  /// additional analyses. Callers can also pre-register analyses and this will
  /// not override those.
  void registerCGSCCAnalyses(CGSCCAnalysisManager &CGAM);

  /// Registers all available function analysis passes.
  ///
  /// This is an interface that can be used to populate a \c
  /// FunctionAnalysisManager with all registered function analyses. Callers can
  /// still manually register any additional analyses. Callers can also
  /// pre-register analyses and this will not override those.
  void registerFunctionAnalyses(FunctionAnalysisManager &FAM);

  /// Registers all available loop analysis passes.
  ///
  /// This is an interface that can be used to populate a \c LoopAnalysisManager
  /// with all registered loop analyses. Callers can still manually register any
  /// additional analyses.
  void registerLoopAnalyses(LoopAnalysisManager &LAM);

  /// Construct the core LLVM function canonicalization and simplification
  /// pipeline.
  ///
  /// This is a long pipeline and uses most of the per-function optimization
  /// passes in LLVM to canonicalize and simplify the IR. It is suitable to run
  /// repeatedly over the IR and is not expected to destroy important
  /// information about the semantics of the IR.
  ///
  /// Note that \p Level cannot be `O0` here. The pipelines produced are
  /// only intended for use when attempting to optimize code. If frontends
  /// require some transformations for semantic reasons, they should explicitly
  /// build them.
  ///
  /// \p Phase indicates the current ThinLTO phase.
  FunctionPassManager
  buildFunctionSimplificationPipeline(OptimizationLevel Level,
                                      ThinLTOPhase Phase,
                                      bool DebugLogging = false);

  /// Construct the core LLVM module canonicalization and simplification
  /// pipeline.
  ///
  /// This pipeline focuses on canonicalizing and simplifying the entire module
  /// of IR. Much like the function simplification pipeline above, it is
  /// suitable to run repeatedly over the IR and is not expected to destroy
  /// important information. It does, however, perform inlining and other
  /// heuristic based simplifications that are not strictly reversible.
  ///
  /// Note that \p Level cannot be `O0` here. The pipelines produced are
  /// only intended for use when attempting to optimize code. If frontends
  /// require some transformations for semantic reasons, they should explicitly
  /// build them.
  ///
  /// \p Phase indicates the current ThinLTO phase.
  ModulePassManager
  buildModuleSimplificationPipeline(OptimizationLevel Level,
                                    ThinLTOPhase Phase,
                                    bool DebugLogging = false);

  /// Construct the module pipeline that performs inlining as well as
  /// the inlining-driven cleanups.
  ModuleInlinerWrapperPass buildInlinerPipeline(OptimizationLevel Level,
                                                ThinLTOPhase Phase,
                                                bool DebugLogging = false);

  /// Construct the core LLVM module optimization pipeline.
  ///
  /// This pipeline focuses on optimizing the execution speed of the IR. It
  /// uses cost modeling and thresholds to balance code growth against runtime
  /// improvements. It includes vectorization and other information destroying
  /// transformations. It also cannot generally be run repeatedly on a module
  /// without potentially seriously regressing either runtime performance of
  /// the code or serious code size growth.
  ///
  /// Note that \p Level cannot be `O0` here. The pipelines produced are
  /// only intended for use when attempting to optimize code. If frontends
  /// require some transformations for semantic reasons, they should explicitly
  /// build them.
  ModulePassManager buildModuleOptimizationPipeline(OptimizationLevel Level,
                                                    bool DebugLogging = false,
                                                    bool LTOPreLink = false);

  /// Build a per-module default optimization pipeline.
  ///
  /// This provides a good default optimization pipeline for per-module
  /// optimization and code generation without any link-time optimization. It
  /// typically correspond to frontend "-O[123]" options for optimization
  /// levels \c O1, \c O2 and \c O3 resp.
  ///
  /// Note that \p Level cannot be `O0` here. The pipelines produced are
  /// only intended for use when attempting to optimize code. If frontends
  /// require some transformations for semantic reasons, they should explicitly
  /// build them.
  ModulePassManager buildPerModuleDefaultPipeline(OptimizationLevel Level,
                                                  bool DebugLogging = false,
                                                  bool LTOPreLink = false);

  /// Build a pre-link, ThinLTO-targeting default optimization pipeline to
  /// a pass manager.
  ///
  /// This adds the pre-link optimizations tuned to prepare a module for
  /// a ThinLTO run. It works to minimize the IR which needs to be analyzed
  /// without making irreversible decisions which could be made better during
  /// the LTO run.
  ///
  /// Note that \p Level cannot be `O0` here. The pipelines produced are
  /// only intended for use when attempting to optimize code. If frontends
  /// require some transformations for semantic reasons, they should explicitly
  /// build them.
  ModulePassManager
  buildThinLTOPreLinkDefaultPipeline(OptimizationLevel Level,
                                     bool DebugLogging = false);

  /// Build an ThinLTO default optimization pipeline to a pass manager.
  ///
  /// This provides a good default optimization pipeline for link-time
  /// optimization and code generation. It is particularly tuned to fit well
  /// when IR coming into the LTO phase was first run through \c
  /// addPreLinkLTODefaultPipeline, and the two coordinate closely.
  ///
  /// Note that \p Level cannot be `O0` here. The pipelines produced are
  /// only intended for use when attempting to optimize code. If frontends
  /// require some transformations for semantic reasons, they should explicitly
  /// build them.
  ModulePassManager
  buildThinLTODefaultPipeline(OptimizationLevel Level, bool DebugLogging,
                              const ModuleSummaryIndex *ImportSummary);

  /// Build a pre-link, LTO-targeting default optimization pipeline to a pass
  /// manager.
  ///
  /// This adds the pre-link optimizations tuned to work well with a later LTO
  /// run. It works to minimize the IR which needs to be analyzed without
  /// making irreversible decisions which could be made better during the LTO
  /// run.
  ///
  /// Note that \p Level cannot be `O0` here. The pipelines produced are
  /// only intended for use when attempting to optimize code. If frontends
  /// require some transformations for semantic reasons, they should explicitly
  /// build them.
  ModulePassManager buildLTOPreLinkDefaultPipeline(OptimizationLevel Level,
                                                   bool DebugLogging = false);

  /// Build an LTO default optimization pipeline to a pass manager.
  ///
  /// This provides a good default optimization pipeline for link-time
  /// optimization and code generation. It is particularly tuned to fit well
  /// when IR coming into the LTO phase was first run through \c
  /// addPreLinkLTODefaultPipeline, and the two coordinate closely.
  ///
  /// Note that \p Level cannot be `O0` here. The pipelines produced are
  /// only intended for use when attempting to optimize code. If frontends
  /// require some transformations for semantic reasons, they should explicitly
  /// build them.
  ModulePassManager buildLTODefaultPipeline(OptimizationLevel Level,
                                            bool DebugLogging,
                                            ModuleSummaryIndex *ExportSummary);

  /// Build the default `AAManager` with the default alias analysis pipeline
  /// registered.
  AAManager buildDefaultAAPipeline();

  /// Parse a textual pass pipeline description into a \c
  /// ModulePassManager.
  ///
  /// The format of the textual pass pipeline description looks something like:
  ///
  ///   module(function(instcombine,sroa),dce,cgscc(inliner,function(...)),...)
  ///
  /// Pass managers have ()s describing the nest structure of passes. All passes
  /// are comma separated. As a special shortcut, if the very first pass is not
  /// a module pass (as a module pass manager is), this will automatically form
  /// the shortest stack of pass managers that allow inserting that first pass.
  /// So, assuming function passes 'fpassN', CGSCC passes 'cgpassN', and loop
  /// passes 'lpassN', all of these are valid:
  ///
  ///   fpass1,fpass2,fpass3
  ///   cgpass1,cgpass2,cgpass3
  ///   lpass1,lpass2,lpass3
  ///
  /// And they are equivalent to the following (resp.):
  ///
  ///   module(function(fpass1,fpass2,fpass3))
  ///   module(cgscc(cgpass1,cgpass2,cgpass3))
  ///   module(function(loop(lpass1,lpass2,lpass3)))
  ///
  /// This shortcut is especially useful for debugging and testing small pass
  /// combinations.
  ///
  /// The sequence of passes aren't necessarily the exact same kind of pass.
  /// You can mix different levels implicitly if adaptor passes are defined to
  /// make them work. For example,
  ///
  ///   mpass1,fpass1,fpass2,mpass2,lpass1
  ///
  /// This pipeline uses only one pass manager: the top-level module manager.
  /// fpass1,fpass2 and lpass1 are added into the the top-level module manager
  /// using only adaptor passes. No nested function/loop pass managers are
  /// added. The purpose is to allow easy pass testing when the user
  /// specifically want the pass to run under a adaptor directly. This is
  /// preferred when a pipeline is largely of one type, but one or just a few
  /// passes are of different types(See PassBuilder.cpp for examples).
  Error parsePassPipeline(ModulePassManager &MPM, StringRef PipelineText,
                          bool VerifyEachPass = true,
                          bool DebugLogging = false);

  /// {{@ Parse a textual pass pipeline description into a specific PassManager
  ///
  /// Automatic deduction of an appropriate pass manager stack is not supported.
  /// For example, to insert a loop pass 'lpass' into a FunctionPassManager,
  /// this is the valid pipeline text:
  ///
  ///   function(lpass)
  Error parsePassPipeline(CGSCCPassManager &CGPM, StringRef PipelineText,
                          bool VerifyEachPass = true,
                          bool DebugLogging = false);
  Error parsePassPipeline(FunctionPassManager &FPM, StringRef PipelineText,
                          bool VerifyEachPass = true,
                          bool DebugLogging = false);
  Error parsePassPipeline(LoopPassManager &LPM, StringRef PipelineText,
                          bool VerifyEachPass = true,
                          bool DebugLogging = false);
  /// @}}

  /// Parse a textual alias analysis pipeline into the provided AA manager.
  ///
  /// The format of the textual AA pipeline is a comma separated list of AA
  /// pass names:
  ///
  ///   basic-aa,globals-aa,...
  ///
  /// The AA manager is set up such that the provided alias analyses are tried
  /// in the order specified. See the \c AAManaager documentation for details
  /// about the logic used. This routine just provides the textual mapping
  /// between AA names and the analyses to register with the manager.
  ///
  /// Returns false if the text cannot be parsed cleanly. The specific state of
  /// the \p AA manager is unspecified if such an error is encountered and this
  /// returns false.
  Error parseAAPipeline(AAManager &AA, StringRef PipelineText);

  /// Returns true if the pass name is the name of an alias analysis pass.
  bool isAAPassName(StringRef PassName);

  /// Returns true if the pass name is the name of a (non-alias) analysis pass.
  bool isAnalysisPassName(StringRef PassName);

  /// Register a callback for a default optimizer pipeline extension
  /// point
  ///
  /// This extension point allows adding passes that perform peephole
  /// optimizations similar to the instruction combiner. These passes will be
  /// inserted after each instance of the instruction combiner pass.
  void registerPeepholeEPCallback(
      const std::function<void(FunctionPassManager &, OptimizationLevel)> &C) {
    PeepholeEPCallbacks.push_back(C);
  }

  /// Register a callback for a default optimizer pipeline extension
  /// point
  ///
  /// This extension point allows adding late loop canonicalization and
  /// simplification passes. This is the last point in the loop optimization
  /// pipeline before loop deletion. Each pass added
  /// here must be an instance of LoopPass.
  /// This is the place to add passes that can remove loops, such as target-
  /// specific loop idiom recognition.
  void registerLateLoopOptimizationsEPCallback(
      const std::function<void(LoopPassManager &, OptimizationLevel)> &C) {
    LateLoopOptimizationsEPCallbacks.push_back(C);
  }

  /// Register a callback for a default optimizer pipeline extension
  /// point
  ///
  /// This extension point allows adding loop passes to the end of the loop
  /// optimizer.
  void registerLoopOptimizerEndEPCallback(
      const std::function<void(LoopPassManager &, OptimizationLevel)> &C) {
    LoopOptimizerEndEPCallbacks.push_back(C);
  }

  /// Register a callback for a default optimizer pipeline extension
  /// point
  ///
  /// This extension point allows adding optimization passes after most of the
  /// main optimizations, but before the last cleanup-ish optimizations.
  void registerScalarOptimizerLateEPCallback(
      const std::function<void(FunctionPassManager &, OptimizationLevel)> &C) {
    ScalarOptimizerLateEPCallbacks.push_back(C);
  }

  /// Register a callback for a default optimizer pipeline extension
  /// point
  ///
  /// This extension point allows adding CallGraphSCC passes at the end of the
  /// main CallGraphSCC passes and before any function simplification passes run
  /// by CGPassManager.
  void registerCGSCCOptimizerLateEPCallback(
      const std::function<void(CGSCCPassManager &, OptimizationLevel)> &C) {
    CGSCCOptimizerLateEPCallbacks.push_back(C);
  }

  /// Register a callback for a default optimizer pipeline extension
  /// point
  ///
  /// This extension point allows adding optimization passes before the
  /// vectorizer and other highly target specific optimization passes are
  /// executed.
  void registerVectorizerStartEPCallback(
      const std::function<void(FunctionPassManager &, OptimizationLevel)> &C) {
    VectorizerStartEPCallbacks.push_back(C);
  }

  /// Register a callback for a default optimizer pipeline extension point.
  ///
  /// This extension point allows adding optimization once at the start of the
  /// pipeline. This does not apply to 'backend' compiles (LTO and ThinLTO
  /// link-time pipelines).
  void registerPipelineStartEPCallback(
      const std::function<void(ModulePassManager &)> &C) {
    PipelineStartEPCallbacks.push_back(C);
  }

  /// Register a callback for a default optimizer pipeline extension point
  ///
  /// This extension point allows adding optimizations at the very end of the
  /// function optimization pipeline. A key difference between this and the
  /// legacy PassManager's OptimizerLast callback is that this extension point
  /// is not triggered at O0. Extensions to the O0 pipeline should append their
  /// passes to the end of the overall pipeline.
  void registerOptimizerLastEPCallback(
      const std::function<void(ModulePassManager &, OptimizationLevel)> &C) {
    OptimizerLastEPCallbacks.push_back(C);
  }

  /// Register a callback for parsing an AliasAnalysis Name to populate
  /// the given AAManager \p AA
  void registerParseAACallback(
      const std::function<bool(StringRef Name, AAManager &AA)> &C) {
    AAParsingCallbacks.push_back(C);
  }

  /// {{@ Register callbacks for analysis registration with this PassBuilder
  /// instance.
  /// Callees register their analyses with the given AnalysisManager objects.
  void registerAnalysisRegistrationCallback(
      const std::function<void(CGSCCAnalysisManager &)> &C) {
    CGSCCAnalysisRegistrationCallbacks.push_back(C);
  }
  void registerAnalysisRegistrationCallback(
      const std::function<void(FunctionAnalysisManager &)> &C) {
    FunctionAnalysisRegistrationCallbacks.push_back(C);
  }
  void registerAnalysisRegistrationCallback(
      const std::function<void(LoopAnalysisManager &)> &C) {
    LoopAnalysisRegistrationCallbacks.push_back(C);
  }
  void registerAnalysisRegistrationCallback(
      const std::function<void(ModuleAnalysisManager &)> &C) {
    ModuleAnalysisRegistrationCallbacks.push_back(C);
  }
  /// @}}

  /// {{@ Register pipeline parsing callbacks with this pass builder instance.
  /// Using these callbacks, callers can parse both a single pass name, as well
  /// as entire sub-pipelines, and populate the PassManager instance
  /// accordingly.
  void registerPipelineParsingCallback(
      const std::function<bool(StringRef Name, CGSCCPassManager &,
                               ArrayRef<PipelineElement>)> &C) {
    CGSCCPipelineParsingCallbacks.push_back(C);
  }
  void registerPipelineParsingCallback(
      const std::function<bool(StringRef Name, FunctionPassManager &,
                               ArrayRef<PipelineElement>)> &C) {
    FunctionPipelineParsingCallbacks.push_back(C);
  }
  void registerPipelineParsingCallback(
      const std::function<bool(StringRef Name, LoopPassManager &,
                               ArrayRef<PipelineElement>)> &C) {
    LoopPipelineParsingCallbacks.push_back(C);
  }
  void registerPipelineParsingCallback(
      const std::function<bool(StringRef Name, ModulePassManager &,
                               ArrayRef<PipelineElement>)> &C) {
    ModulePipelineParsingCallbacks.push_back(C);
  }
  /// @}}

  /// Register a callback for a top-level pipeline entry.
  ///
  /// If the PassManager type is not given at the top level of the pipeline
  /// text, this Callback should be used to determine the appropriate stack of
  /// PassManagers and populate the passed ModulePassManager.
  void registerParseTopLevelPipelineCallback(
      const std::function<bool(ModulePassManager &, ArrayRef<PipelineElement>,
                               bool VerifyEachPass, bool DebugLogging)> &C) {
    TopLevelPipelineParsingCallbacks.push_back(C);
  }

  /// Add PGOInstrumenation passes for O0 only.
  void addPGOInstrPassesForO0(ModulePassManager &MPM, bool DebugLogging,
                              bool RunProfileGen, bool IsCS,
                              std::string ProfileFile,
                              std::string ProfileRemappingFile);


  /// Returns PIC. External libraries can use this to register pass
  /// instrumentation callbacks.
  PassInstrumentationCallbacks *getPassInstrumentationCallbacks() const {
    return PIC;
  }

private:
  // O1 pass pipeline
  FunctionPassManager buildO1FunctionSimplificationPipeline(
      OptimizationLevel Level, ThinLTOPhase Phase, bool DebugLogging = false);

  static Optional<std::vector<PipelineElement>>
  parsePipelineText(StringRef Text);

  Error parseModulePass(ModulePassManager &MPM, const PipelineElement &E,
                        bool VerifyEachPass, bool DebugLogging);
  Error parseCGSCCPass(CGSCCPassManager &CGPM, const PipelineElement &E,
                       bool VerifyEachPass, bool DebugLogging);
  Error parseFunctionPass(FunctionPassManager &FPM, const PipelineElement &E,
                          bool VerifyEachPass, bool DebugLogging);
  Error parseLoopPass(LoopPassManager &LPM, const PipelineElement &E,
                      bool VerifyEachPass, bool DebugLogging);
  bool parseAAPassName(AAManager &AA, StringRef Name);

  Error parseLoopPassPipeline(LoopPassManager &LPM,
                              ArrayRef<PipelineElement> Pipeline,
                              bool VerifyEachPass, bool DebugLogging);
  Error parseFunctionPassPipeline(FunctionPassManager &FPM,
                                  ArrayRef<PipelineElement> Pipeline,
                                  bool VerifyEachPass, bool DebugLogging);
  Error parseCGSCCPassPipeline(CGSCCPassManager &CGPM,
                               ArrayRef<PipelineElement> Pipeline,
                               bool VerifyEachPass, bool DebugLogging);
  Error parseModulePassPipeline(ModulePassManager &MPM,
                                ArrayRef<PipelineElement> Pipeline,
                                bool VerifyEachPass, bool DebugLogging);

  void addPGOInstrPasses(ModulePassManager &MPM, bool DebugLogging,
                         OptimizationLevel Level, bool RunProfileGen, bool IsCS,
                         std::string ProfileFile,
                         std::string ProfileRemappingFile);
  void invokePeepholeEPCallbacks(FunctionPassManager &, OptimizationLevel);

  // Extension Point callbacks
  SmallVector<std::function<void(FunctionPassManager &, OptimizationLevel)>, 2>
      PeepholeEPCallbacks;
  SmallVector<std::function<void(LoopPassManager &, OptimizationLevel)>, 2>
      LateLoopOptimizationsEPCallbacks;
  SmallVector<std::function<void(LoopPassManager &, OptimizationLevel)>, 2>
      LoopOptimizerEndEPCallbacks;
  SmallVector<std::function<void(FunctionPassManager &, OptimizationLevel)>, 2>
      ScalarOptimizerLateEPCallbacks;
  SmallVector<std::function<void(CGSCCPassManager &, OptimizationLevel)>, 2>
      CGSCCOptimizerLateEPCallbacks;
  SmallVector<std::function<void(FunctionPassManager &, OptimizationLevel)>, 2>
      VectorizerStartEPCallbacks;
  SmallVector<std::function<void(ModulePassManager &, OptimizationLevel)>, 2>
      OptimizerLastEPCallbacks;
  // Module callbacks
  SmallVector<std::function<void(ModulePassManager &)>, 2>
      PipelineStartEPCallbacks;
  SmallVector<std::function<void(ModuleAnalysisManager &)>, 2>
      ModuleAnalysisRegistrationCallbacks;
  SmallVector<std::function<bool(StringRef, ModulePassManager &,
                                 ArrayRef<PipelineElement>)>,
              2>
      ModulePipelineParsingCallbacks;
  SmallVector<std::function<bool(ModulePassManager &, ArrayRef<PipelineElement>,
                                 bool VerifyEachPass, bool DebugLogging)>,
              2>
      TopLevelPipelineParsingCallbacks;
  // CGSCC callbacks
  SmallVector<std::function<void(CGSCCAnalysisManager &)>, 2>
      CGSCCAnalysisRegistrationCallbacks;
  SmallVector<std::function<bool(StringRef, CGSCCPassManager &,
                                 ArrayRef<PipelineElement>)>,
              2>
      CGSCCPipelineParsingCallbacks;
  // Function callbacks
  SmallVector<std::function<void(FunctionAnalysisManager &)>, 2>
      FunctionAnalysisRegistrationCallbacks;
  SmallVector<std::function<bool(StringRef, FunctionPassManager &,
                                 ArrayRef<PipelineElement>)>,
              2>
      FunctionPipelineParsingCallbacks;
  // Loop callbacks
  SmallVector<std::function<void(LoopAnalysisManager &)>, 2>
      LoopAnalysisRegistrationCallbacks;
  SmallVector<std::function<bool(StringRef, LoopPassManager &,
                                 ArrayRef<PipelineElement>)>,
              2>
      LoopPipelineParsingCallbacks;
  // AA callbacks
  SmallVector<std::function<bool(StringRef Name, AAManager &AA)>, 2>
      AAParsingCallbacks;
};

/// This utility template takes care of adding require<> and invalidate<>
/// passes for an analysis to a given \c PassManager. It is intended to be used
/// during parsing of a pass pipeline when parsing a single PipelineName.
/// When registering a new function analysis FancyAnalysis with the pass
/// pipeline name "fancy-analysis", a matching ParsePipelineCallback could look
/// like this:
///
/// static bool parseFunctionPipeline(StringRef Name, FunctionPassManager &FPM,
///                                   ArrayRef<PipelineElement> P) {
///   if (parseAnalysisUtilityPasses<FancyAnalysis>("fancy-analysis", Name,
///                                                 FPM))
///     return true;
///   return false;
/// }
template <typename AnalysisT, typename IRUnitT, typename AnalysisManagerT,
          typename... ExtraArgTs>
bool parseAnalysisUtilityPasses(
    StringRef AnalysisName, StringRef PipelineName,
    PassManager<IRUnitT, AnalysisManagerT, ExtraArgTs...> &PM) {
  if (!PipelineName.endswith(">"))
    return false;
  // See if this is an invalidate<> pass name
  if (PipelineName.startswith("invalidate<")) {
    PipelineName = PipelineName.substr(11, PipelineName.size() - 12);
    if (PipelineName != AnalysisName)
      return false;
    PM.addPass(InvalidateAnalysisPass<AnalysisT>());
    return true;
  }

  // See if this is a require<> pass name
  if (PipelineName.startswith("require<")) {
    PipelineName = PipelineName.substr(8, PipelineName.size() - 9);
    if (PipelineName != AnalysisName)
      return false;
    PM.addPass(RequireAnalysisPass<AnalysisT, IRUnitT, AnalysisManagerT,
                                   ExtraArgTs...>());
    return true;
  }

  return false;
}
}

#endif
