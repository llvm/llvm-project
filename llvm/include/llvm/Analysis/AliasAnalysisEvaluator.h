//===- AliasAnalysisEvaluator.h - Alias Analysis Accuracy Evaluator -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple N^2 alias analysis accuracy evaluator.
// Basically, for each function in the program, it simply queries to see how the
// alias analysis implementation answers alias queries between each pair of
// pointers in the function.
//
// This is inspired and adapted from code by: Naveen Neelakantam, Francesco
// Spadini, and Wojciech Stryjewski.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_ALIASANALYSISEVALUATOR_H
#define LLVM_ANALYSIS_ALIASANALYSISEVALUATOR_H

#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
class AAResults;

class AAEvaluator {
  int64_t FunctionCount;
  int64_t NoAliasCount, MayAliasCount, PartialAliasCount, MustAliasCount;
  int64_t NoModRefCount, ModCount, RefCount, ModRefCount;

public:
  AAEvaluator()
      : FunctionCount(), NoAliasCount(), MayAliasCount(), PartialAliasCount(),
        MustAliasCount(), NoModRefCount(), ModCount(), RefCount(),
        ModRefCount() {}
  AAEvaluator(AAEvaluator &&Arg)
      : FunctionCount(Arg.FunctionCount), NoAliasCount(Arg.NoAliasCount),
        MayAliasCount(Arg.MayAliasCount),
        PartialAliasCount(Arg.PartialAliasCount),
        MustAliasCount(Arg.MustAliasCount), NoModRefCount(Arg.NoModRefCount),
        ModCount(Arg.ModCount), RefCount(Arg.RefCount),
        ModRefCount(Arg.ModRefCount) {
    Arg.FunctionCount = 0;
  }
  ~AAEvaluator();

  static StringRef name() { return "AAEvaluator"; }

  /// \brief Run the pass over the function.
  PreservedAnalyses run(Function &F, AnalysisManager<Function> *AM);

private:
  // Allow the legacy pass to run this using an internal API.
  friend class AAEvalLegacyPass;

  void runInternal(Function &F, AAResults &AA);
};

/// Create a wrapper of the above for the legacy pass manager.
FunctionPass *createAAEvalPass();

}

#endif
