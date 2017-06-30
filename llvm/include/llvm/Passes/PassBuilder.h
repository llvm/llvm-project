//===- Parsing, selection, and construction of pass pipelines --*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include <vector>

namespace llvm {
class StringRef;
class AAManager;
class TargetMachine;

/// A struct capturing PGO tunables.
struct PGOOptions {
  std::string ProfileGenFile = "";
  std::string ProfileUseFile = "";
  std::string SampleProfileFile = "";
  bool RunProfileGen = false;
};

/// \brief This class provides access to building LLVM's passes.
///
/// It's members provide the baseline state available to passes during their
/// construction. The \c PassRegistry.def file specifies how to construct all
/// of the built-in passes, and those may reference these members during
/// construction.
class PassBuilder {
  TargetMachine *TM;
  Optional<PGOOptions> PGOOpt;

public:
  /// \brief LLVM-provided high-level optimization levels.
  ///
  /// This enumerates the LLVM-provided high-level optimization levels. Each
  /// level has a specific goal and rationale.
  enum OptimizationLevel {
    /// Disable as many optimizations as possible. This doesn't completely
    /// disable the optimizer in all cases, for example always_inline functions
    /// can be required to be inlined for correctness.
    O0,

    /// Optimize quickly without destroying debuggability.
    ///
    /// FIXME: The current and historical behavior of this level does *not*
    /// agree with this goal, but we would like to move toward this goal in the
    /// future.
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
    /// vectorization, or fusion might not make sense here due to the degree to
    /// which the executed code would differ from the source code, and the
    /// potential compile time cost.
    O1,

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
    O2,

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
    O3,

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
    Os,

    /// A very specialized mode that will optimize for code size at any and all
    /// costs.
    ///
    /// This is useful primarily when there are absolute size limitations and
    /// any effort taken to reduce the size is worth it regardless of the
    /// execution time impact. You should expect this level to produce rather
    /// slow, but very small, code.
    Oz
  };

  explicit PassBuilder(TargetMachine *TM = nullptr,
                       Optional<PGOOptions> PGOOpt = None)
      : TM(TM), PGOOpt(PGOOpt) {}

  /// \brief Cross register the analysis managers through their proxies.
  ///
  /// This is an interface that can be used to cross register each
  // AnalysisManager with all the others analysis managers.
  void crossRegisterProxies(LoopAnalysisManager &LAM,
                            FunctionAnalysisManager &FAM,
                            CGSCCAnalysisManager &CGAM,
                            ModuleAnalysisManager &MAM);

  /// \brief Registers all available module analysis passes.
  ///
  /// This is an interface that can be used to populate a \c
  /// ModuleAnalysisManager with all registered module analyses. Callers can
  /// still manually register any additional analyses. Callers can also
  /// pre-register analyses and this will not override those.
  void registerModuleAnalyses(ModuleAnalysisManager &MAM);

  /// \brief Registers all available CGSCC analysis passes.
  ///
  /// This is an interface that can be used to populate a \c CGSCCAnalysisManager
  /// with all registered CGSCC analyses. Callers can still manually register any
  /// additional analyses. Callers can also pre-register analyses and this will
  /// not override those.
  void registerCGSCCAnalyses(CGSCCAnalysisManager &CGAM);

  /// \brief Registers all available function analysis passes.
  ///
  /// This is an interface that can be used to populate a \c
  /// FunctionAnalysisManager with all registered function analyses. Callers can
  /// still manually register any additional analyses. Callers can also
  /// pre-register analyses and this will not override those.
  void registerFunctionAnalyses(FunctionAnalysisManager &FAM);

  /// \brief Registers all available loop analysis passes.
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
  FunctionPassManager
  buildFunctionSimplificationPipeline(OptimizationLevel Level,
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
  ModulePassManager
  buildModuleSimplificationPipeline(OptimizationLevel Level,
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
                                                    bool DebugLogging = false);

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
                                                  bool DebugLogging = false);

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
  ModulePassManager buildThinLTODefaultPipeline(OptimizationLevel Level,
                                                bool DebugLogging = false);

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
                                            bool DebugLogging = false);

  /// Build the default `AAManager` with the default alias analysis pipeline
  /// registered.
  AAManager buildDefaultAAPipeline();

  /// \brief Parse a textual pass pipeline description into a \c ModulePassManager.
  ///
  /// The format of the textual pass pipeline description looks something like:
  ///
  ///   module(function(instcombine,sroa),dce,cgscc(inliner,function(...)),...)
  ///
  /// Pass managers have ()s describing the nest structure of passes. All passes
  /// are comma separated. As a special shortcut, if the very first pass is not
  /// a module pass (as a module pass manager is), this will automatically form
  /// the shortest stack of pass managers that allow inserting that first pass.
  /// So, assuming function passes 'fpassN', CGSCC passes 'cgpassN', and loop passes
  /// 'lpassN', all of these are valid:
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
  /// combinations. Note that these shortcuts don't introduce any other magic. If
  /// the sequence of passes aren't all the exact same kind of pass, it will be
  /// an error. You cannot mix different levels implicitly, you must explicitly
  /// form a pass manager in which to nest passes.
  bool parsePassPipeline(ModulePassManager &MPM, StringRef PipelineText,
                         bool VerifyEachPass = true, bool DebugLogging = false);

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
  bool parseAAPipeline(AAManager &AA, StringRef PipelineText);

private:
  /// A struct to capture parsed pass pipeline names.
  struct PipelineElement {
    StringRef Name;
    std::vector<PipelineElement> InnerPipeline;
  };

  static Optional<std::vector<PipelineElement>>
  parsePipelineText(StringRef Text);

  bool parseModulePass(ModulePassManager &MPM, const PipelineElement &E,
                       bool VerifyEachPass, bool DebugLogging);
  bool parseCGSCCPass(CGSCCPassManager &CGPM, const PipelineElement &E,
                      bool VerifyEachPass, bool DebugLogging);
  bool parseFunctionPass(FunctionPassManager &FPM, const PipelineElement &E,
                     bool VerifyEachPass, bool DebugLogging);
  bool parseLoopPass(LoopPassManager &LPM, const PipelineElement &E,
                     bool VerifyEachPass, bool DebugLogging);
  bool parseAAPassName(AAManager &AA, StringRef Name);

  bool parseLoopPassPipeline(LoopPassManager &LPM,
                             ArrayRef<PipelineElement> Pipeline,
                             bool VerifyEachPass, bool DebugLogging);
  bool parseFunctionPassPipeline(FunctionPassManager &FPM,
                                 ArrayRef<PipelineElement> Pipeline,
                                 bool VerifyEachPass, bool DebugLogging);
  bool parseCGSCCPassPipeline(CGSCCPassManager &CGPM,
                              ArrayRef<PipelineElement> Pipeline,
                              bool VerifyEachPass, bool DebugLogging);
  bool parseModulePassPipeline(ModulePassManager &MPM,
                               ArrayRef<PipelineElement> Pipeline,
                               bool VerifyEachPass, bool DebugLogging);
};
}

#endif
