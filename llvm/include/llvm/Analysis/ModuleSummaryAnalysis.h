//===- ModuleSummaryAnalysis.h - Module summary index builder ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This is the interface to build a ModuleSummaryIndex for a module.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_MODULESUMMARYANALYSIS_H
#define LLVM_ANALYSIS_MODULESUMMARYANALYSIS_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/Pass.h"

namespace llvm {

class BlockFrequencyInfo;

/// Class to build a module summary index for the given Module, possibly from
/// a Pass.
class ModuleSummaryIndexBuilder {
  /// The index being built
  std::unique_ptr<ModuleSummaryIndex> Index;
  /// The module for which we are building an index
  const Module *M;

public:
  /// Default constructor
  ModuleSummaryIndexBuilder() = default;

  /// Constructor that builds an index for the given Module. An optional
  /// callback can be supplied to obtain the frequency info for a function.
  ModuleSummaryIndexBuilder(
      const Module *M,
      std::function<BlockFrequencyInfo *(const Function &F)> Ftor = nullptr);

  /// Get a reference to the index owned by builder
  ModuleSummaryIndex &getIndex() const { return *Index; }

  /// Take ownership of the built index
  std::unique_ptr<ModuleSummaryIndex> takeIndex() { return std::move(Index); }

private:
  /// Compute summary for given function with optional frequency information
  void computeFunctionSummary(const Function &F,
                              BlockFrequencyInfo *BFI = nullptr);

  /// Compute summary for given variable with optional frequency information
  void computeVariableSummary(const GlobalVariable &V);
};

/// Legacy wrapper pass to provide the ModuleSummaryIndex object.
class ModuleSummaryIndexWrapperPass : public ModulePass {
  std::unique_ptr<ModuleSummaryIndexBuilder> IndexBuilder;

public:
  static char ID;

  ModuleSummaryIndexWrapperPass();

  /// Get the index built by pass
  ModuleSummaryIndex &getIndex() { return IndexBuilder->getIndex(); }
  const ModuleSummaryIndex &getIndex() const {
    return IndexBuilder->getIndex();
  }

  bool runOnModule(Module &M) override;
  bool doFinalization(Module &M) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

//===--------------------------------------------------------------------===//
//
// createModuleSummaryIndexWrapperPass - This pass builds a ModuleSummaryIndex
// object for the module, to be written to bitcode or LLVM assembly.
//
ModulePass *createModuleSummaryIndexWrapperPass();

/// Returns true if \p M is eligible for ThinLTO promotion.
///
/// Currently we check if it has any any InlineASM that uses an internal symbol.
bool moduleCanBeRenamedForThinLTO(const Module &M);
}

#endif
