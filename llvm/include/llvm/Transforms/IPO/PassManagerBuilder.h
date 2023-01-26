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
#include <string>
#include <vector>

namespace llvm {
class ModuleSummaryIndex;
class Pass;
class TargetLibraryInfoImpl;

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

  bool DisableUnrollLoops;
  bool CallGraphProfile;
  bool SLPVectorize;
  bool LoopVectorize;
  bool LoopsInterleaved;
  bool DisableGVNLoadPRE;
  bool ForgetAllSCEVInLoopUnroll;
  bool VerifyInput;
  bool VerifyOutput;
  bool MergeFunctions;
  bool DivergentTarget;
  unsigned LicmMssaOptCap;
  unsigned LicmMssaNoAccForPromotionCap;

public:
  PassManagerBuilder();
  ~PassManagerBuilder();

private:
  void addInitialAliasAnalysisPasses(legacy::PassManagerBase &PM) const;
  void addFunctionSimplificationPasses(legacy::PassManagerBase &MPM);
  void addVectorPasses(legacy::PassManagerBase &PM, bool IsFullLTO);

public:
  /// populateFunctionPassManager - This fills in the function pass manager,
  /// which is expected to be run on each function immediately as it is
  /// generated.  The idea is to reduce the size of the IR in memory.
  void populateFunctionPassManager(legacy::FunctionPassManager &FPM);

  /// populateModulePassManager - This sets up the primary pass manager.
  void populateModulePassManager(legacy::PassManagerBase &MPM);
};

inline PassManagerBuilder *unwrap(LLVMPassManagerBuilderRef P) {
    return reinterpret_cast<PassManagerBuilder*>(P);
}

inline LLVMPassManagerBuilderRef wrap(PassManagerBuilder *P) {
  return reinterpret_cast<LLVMPassManagerBuilderRef>(P);
}

} // end namespace llvm
#endif
