//===- SROA.h - Scalar Replacement Of Aggregates ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file provides the interface for LLVM's Scalar Replacement of
/// Aggregates pass. This pass provides both aggregate splitting and the
/// primary SSA formation used in the compiler.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_SROA_H
#define LLVM_TRANSFORMS_SCALAR_SROA_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Function;

struct SROAOptions {
  enum PreserveCFGOption : bool { ModifyCFG, PreserveCFG };
  enum DecomposeStructsOption : bool { NoDecomposeStructs, DecomposeStructs };
  PreserveCFGOption PCFGOption;
  DecomposeStructsOption DSOption;
  SROAOptions(PreserveCFGOption PCFGOption)
      : PCFGOption(PCFGOption), DSOption(NoDecomposeStructs) {}
  SROAOptions(PreserveCFGOption PCFGOption, DecomposeStructsOption DSOption)
      : PCFGOption(PCFGOption), DSOption(DSOption) {}
};

class SROAPass : public PassInfoMixin<SROAPass> {
  const SROAOptions Options;

public:
  /// If \p PreserveCFG is set, then the pass is not allowed to modify CFG
  /// in any way, even if it would update CFG analyses.
  SROAPass(SROAOptions::PreserveCFGOption PreserveCFG);

  /// If \p Options.PreserveCFG is set, then the pass is not allowed to modify
  /// CFG in any way, even if it would update CFG analyses.
  /// If \p Options.DecomposeStructs is set, then the pass will decompose
  /// structs allocas into its constituent components regardless of whether or
  /// not pointer offsets into them are known at compile time.
  SROAPass(const SROAOptions &Options);

  /// Run the pass over the function.
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  void printPipeline(raw_ostream &OS,
                     function_ref<StringRef(StringRef)> MapClassName2PassName);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_SROA_H
