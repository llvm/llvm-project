//===------ GlobalMergeFunctions.h - Global merge functions -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass defines the implementation of a function merging mechanism
// that utilizes a stable function hash to track differences in constants and
// identify potential merge candidates. The process involves two rounds:
// 1. The first round collects stable function hashes and identifies merge
//    candidates with matching hashes. It also computes the set of parameters
//    that point to different constants during the stable function merge.
// 2. The second round leverages this collected global function information to
//    optimistically create a merged function in each module context, ensuring
//    correct transformation.
// Similar to the global outliner, this approach uses the linker's deduplication
// (ICF) to fold identical merged functions, thereby reducing the final binary
// size. The work is inspired by the concepts discussed in the following paper:
// https://dl.acm.org/doi/pdf/10.1145/3652032.3657575.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALMERGEFUNCTIONS_H
#define LLVM_CODEGEN_GLOBALMERGEFUNCTIONS_H

#include "llvm/CGData/StableFunctionMap.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

enum class HashFunctionMode {
  Local,
  BuildingHashFuncion,
  UsingHashFunction,
};

namespace llvm {

// A vector of locations (the pair of (instruction, operand) indices) reachable
// from a parameter.
using ParamLocs = SmallVector<IndexPair, 4>;
// A vector of parameters
using ParamLocsVecTy = SmallVector<ParamLocs, 8>;

/// GlobalMergeFunc is a ModulePass that implements a function merging mechanism
/// using stable function hashes. It identifies and merges functions with
/// matching hashes across modules to optimize binary size.
class GlobalMergeFunc {
  HashFunctionMode MergerMode = HashFunctionMode::Local;

  std::unique_ptr<StableFunctionMap> LocalFunctionMap;

  const ModuleSummaryIndex *Index;

public:
  /// The suffix used to identify the merged function that parameterizes
  /// the constant values. Note that the original function, without this suffix,
  /// becomes a thunk supplying contexts to the merged function via parameters.
  static constexpr const char MergingInstanceSuffix[] = ".Tgm";

  GlobalMergeFunc(const ModuleSummaryIndex *Index) : Index(Index) {};

  void initializeMergerMode(const Module &M);

  bool run(Module &M);

  /// Analyze module to create stable function into LocalFunctionMap.
  void analyze(Module &M);

  /// Emit LocalFunctionMap into __llvm_merge section.
  void emitFunctionMap(Module &M);

  /// Merge functions in the module using the given function map.
  bool merge(Module &M, const StableFunctionMap *FunctionMap);
};

/// Global function merging pass for new pass manager.
struct GlobalMergeFuncPass : public PassInfoMixin<GlobalMergeFuncPass> {
  const ModuleSummaryIndex *ImportSummary = nullptr;
  GlobalMergeFuncPass() = default;
  GlobalMergeFuncPass(const ModuleSummaryIndex *ImportSummary)
      : ImportSummary(ImportSummary) {}
  PreservedAnalyses run(Module &M, AnalysisManager<Module> &);
};

} // end namespace llvm
#endif // LLVM_CODEGEN_GLOBALMERGEFUNCTIONS_H
