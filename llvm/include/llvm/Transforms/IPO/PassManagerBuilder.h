// llvm/Transforms/IPO/PassManagerBuilder.h - Build Standard Pass -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the PassManagerBuilder class, which is used to set up a
// "standard" optimization sequence suitable for languages like C and C++.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_PASSMANAGERBUILDER_H
#define LLVM_TRANSFORMS_IPO_PASSMANAGERBUILDER_H

#include "llvm-c/Transforms/PassManagerBuilder.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace llvm {
class ModuleSummaryIndex;
class Pass;
class TargetLibraryInfoImpl;
class TargetMachine;

// The old pass manager infrastructure is hidden in a legacy namespace now.
namespace legacy {
class FunctionPassManager;
class PassManagerBase;
}

/// PassManagerBuilder - This class is used to set up a standard optimization
/// sequence for languages like C and C++, allowing some APIs to customize the
/// pass sequence in various ways. A simple example of using it would be:
///
///  PassManagerBuilder Builder;
///  Builder.OptLevel = 2;
///  Builder.populateFunctionPassManager(FPM);
///  Builder.populateModulePassManager(MPM);
///
/// In addition to setting up the basic passes, PassManagerBuilder allows
/// frontends to vend a plugin API, where plugins are allowed to add extensions
/// to the default pass manager.  They do this by specifying where in the pass
/// pipeline they want to be added, along with a callback function that adds
/// the pass(es).  For example, a plugin that wanted to add a loop optimization
/// could do something like this:
///
/// static void addMyLoopPass(const PMBuilder &Builder, PassManagerBase &PM) {
///   if (Builder.getOptLevel() > 2 && Builder.getOptSizeLevel() == 0)
///     PM.add(createMyAwesomePass());
/// }
///   ...
///   Builder.addExtension(PassManagerBuilder::EP_LoopOptimizerEnd,
///                        addMyLoopPass);
///   ...
class PassManagerBuilder {
public:
  /// Extensions are passed to the builder itself (so they can see how it is
  /// configured) as well as the pass manager to add stuff to.
  typedef std::function<void(const PassManagerBuilder &Builder,
                             legacy::PassManagerBase &PM)>
      ExtensionFn;
  typedef int GlobalExtensionID;

  enum ExtensionPointTy {
    /// EP_EarlyAsPossible - This extension point allows adding passes before
    /// any other transformations, allowing them to see the code as it is coming
    /// out of the frontend.
    EP_EarlyAsPossible,

    /// EP_ModuleOptimizerEarly - This extension point allows adding passes
    /// just before the main module-level optimization passes.
    EP_ModuleOptimizerEarly,

    /// EP_LoopOptimizerEnd - This extension point allows adding loop passes to
    /// the end of the loop optimizer.
    EP_LoopOptimizerEnd,

    /// EP_ScalarOptimizerLate - This extension point allows adding optimization
    /// passes after most of the main optimizations, but before the last
    /// cleanup-ish optimizations.
    EP_ScalarOptimizerLate,

    /// EP_OptimizerLast -- This extension point allows adding passes that
    /// run after everything else.
    EP_OptimizerLast,

    /// EP_VectorizerStart - This extension point allows adding optimization
    /// passes before the vectorizer and other highly target specific
    /// optimization passes are executed.
    EP_VectorizerStart,

    /// EP_EnabledOnOptLevel0 - This extension point allows adding passes that
    /// should not be disabled by O0 optimization level. The passes will be
    /// inserted after the inlining pass.
    EP_EnabledOnOptLevel0,

    /// EP_Peephole - This extension point allows adding passes that perform
    /// peephole optimizations similar to the instruction combiner. These passes
    /// will be inserted after each instance of the instruction combiner pass.
    EP_Peephole,

    /// EP_LateLoopOptimizations - This extension point allows adding late loop
    /// canonicalization and simplification passes. This is the last point in
    /// the loop optimization pipeline before loop deletion. Each pass added
    /// here must be an instance of LoopPass.
    /// This is the place to add passes that can remove loops, such as target-
    /// specific loop idiom recognition.
    EP_LateLoopOptimizations,

    /// EP_CGSCCOptimizerLate - This extension point allows adding CallGraphSCC
    /// passes at the end of the main CallGraphSCC passes and before any
    /// function simplification passes run by CGPassManager.
    EP_CGSCCOptimizerLate,

    /// EP_FullLinkTimeOptimizationEarly - This extensions point allow adding
    /// passes that
    /// run at Link Time, before Full Link Time Optimization.
    EP_FullLinkTimeOptimizationEarly,

    /// EP_FullLinkTimeOptimizationLast - This extensions point allow adding
    /// passes that
    /// run at Link Time, after Full Link Time Optimization.
    EP_FullLinkTimeOptimizationLast,
  };

  /// The Optimization Level - Specify the basic optimization level.
  ///    0 = -O0, 1 = -O1, 2 = -O2, 3 = -O3
  unsigned OptLevel;

  /// SizeLevel - How much we're optimizing for size.
  ///    0 = none, 1 = -Os, 2 = -Oz
  unsigned SizeLevel;

  /// LibraryInfo - Specifies information about the runtime library for the
  /// optimizer.  If this is non-null, it is added to both the function and
  /// per-module pass pipeline.
  TargetLibraryInfoImpl *LibraryInfo;

  /// Inliner - Specifies the inliner to use.  If this is non-null, it is
  /// added to the per-module passes.
  Pass *Inliner;

  /// The module summary index to use for exporting information from the
  /// regular LTO phase, for example for the CFI and devirtualization type
  /// tests.
  ModuleSummaryIndex *ExportSummary = nullptr;

  /// The module summary index to use for importing information to the
  /// thin LTO backends, for example for the CFI and devirtualization type
  /// tests.
  const ModuleSummaryIndex *ImportSummary = nullptr;

  bool DisableTailCalls;
  bool DisableUnrollLoops;
  bool CallGraphProfile;
  bool SLPVectorize;
  bool LoopVectorize;
  bool LoopsInterleaved;
  bool RerollLoops;
  bool NewGVN;
  bool DisableGVNLoadPRE;
  bool ForgetAllSCEVInLoopUnroll;
  bool VerifyInput;
  bool VerifyOutput;
  bool MergeFunctions;
  bool PrepareForLTO;
  bool PrepareForThinLTO;
  bool PerformThinLTO;
  bool DivergentTarget;
  unsigned LicmMssaOptCap;
  unsigned LicmMssaNoAccForPromotionCap;

  /// Enable profile instrumentation pass.
  bool EnablePGOInstrGen;
  /// Enable profile context sensitive instrumentation pass.
  bool EnablePGOCSInstrGen;
  /// Enable profile context sensitive profile use pass.
  bool EnablePGOCSInstrUse;
  /// Profile data file name that the instrumentation will be written to.
  std::string PGOInstrGen;
  /// Path of the profile data file.
  std::string PGOInstrUse;
  /// Path of the sample Profile data file.
  std::string PGOSampleUse;

private:
  /// ExtensionList - This is list of all of the extensions that are registered.
  std::vector<std::pair<ExtensionPointTy, ExtensionFn>> Extensions;

public:
  PassManagerBuilder();
  ~PassManagerBuilder();
  /// Adds an extension that will be used by all PassManagerBuilder instances.
  /// This is intended to be used by plugins, to register a set of
  /// optimisations to run automatically.
  ///
  /// \returns A global extension identifier that can be used to remove the
  /// extension.
  static GlobalExtensionID addGlobalExtension(ExtensionPointTy Ty,
                                              ExtensionFn Fn);
  /// Removes an extension that was previously added using addGlobalExtension.
  /// This is also intended to be used by plugins, to remove any extension that
  /// was previously registered before being unloaded.
  ///
  /// \param ExtensionID Identifier of the extension to be removed.
  static void removeGlobalExtension(GlobalExtensionID ExtensionID);
  void addExtension(ExtensionPointTy Ty, ExtensionFn Fn);

private:
  void addExtensionsToPM(ExtensionPointTy ETy,
                         legacy::PassManagerBase &PM) const;
  void addInitialAliasAnalysisPasses(legacy::PassManagerBase &PM) const;
  void addLTOOptimizationPasses(legacy::PassManagerBase &PM);
  void addLateLTOOptimizationPasses(legacy::PassManagerBase &PM);
  void addPGOInstrPasses(legacy::PassManagerBase &MPM, bool IsCS);
  void addFunctionSimplificationPasses(legacy::PassManagerBase &MPM);
  void addVectorPasses(legacy::PassManagerBase &PM, bool IsFullLTO);

public:
  /// populateFunctionPassManager - This fills in the function pass manager,
  /// which is expected to be run on each function immediately as it is
  /// generated.  The idea is to reduce the size of the IR in memory.
  void populateFunctionPassManager(legacy::FunctionPassManager &FPM);

  /// populateModulePassManager - This sets up the primary pass manager.
  void populateModulePassManager(legacy::PassManagerBase &MPM);
  void populateLTOPassManager(legacy::PassManagerBase &PM);
  void populateThinLTOPassManager(legacy::PassManagerBase &PM);
};

/// Registers a function for adding a standard set of passes.  This should be
/// used by optimizer plugins to allow all front ends to transparently use
/// them.  Create a static instance of this class in your plugin, providing a
/// private function that the PassManagerBuilder can use to add your passes.
class RegisterStandardPasses {
  PassManagerBuilder::GlobalExtensionID ExtensionID;

public:
  RegisterStandardPasses(PassManagerBuilder::ExtensionPointTy Ty,
                         PassManagerBuilder::ExtensionFn Fn) {
    ExtensionID = PassManagerBuilder::addGlobalExtension(Ty, std::move(Fn));
  }

  ~RegisterStandardPasses() {
    // If the collection holding the global extensions is destroyed after the
    // plugin is unloaded, the extension has to be removed here. Indeed, the
    // destructor of the ExtensionFn may reference code in the plugin.
    PassManagerBuilder::removeGlobalExtension(ExtensionID);
  }
};

inline PassManagerBuilder *unwrap(LLVMPassManagerBuilderRef P) {
    return reinterpret_cast<PassManagerBuilder*>(P);
}

inline LLVMPassManagerBuilderRef wrap(PassManagerBuilder *P) {
  return reinterpret_cast<LLVMPassManagerBuilderRef>(P);
}

} // end namespace llvm
#endif
