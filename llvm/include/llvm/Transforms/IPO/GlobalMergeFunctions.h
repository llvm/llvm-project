//===------ GlobalMergeFunctions.h - Global merge functions -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file defines global merge functions pass and related data structure.
///
//===----------------------------------------------------------------------===//

#ifndef PIKA_TRANSFORMS_UTILS_GLOBALMERGEFUNCTIONS_H
#define PIKA_TRANSFORMS_UTILS_GLOBALMERGEFUNCTIONS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StableHashing.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CGData/StableFunctionMap.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include <map>
#include <mutex>

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
// A map of stable hash to a vector of stable functions

/// GlobalMergeFunc finds functions which only differ by constants in
/// certain instructions, e.g. resulting from specialized functions of layout
/// compatible types.
/// Unlike PikaMergeFunc that directly compares IRs, this uses stable function
/// hash to find the merge candidate. Similar to the global outliner, we can run
/// codegen twice to collect function merge candidate in the first round, and
/// merge functions globally in the second round.
class GlobalMergeFunc : public ModulePass {
  HashFunctionMode MergerMode = HashFunctionMode::Local;

  std::unique_ptr<StableFunctionMap> LocalFunctionMap;

public:
  static char ID;

  GlobalMergeFunc();

  StringRef getPassName() const override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  void initializeMergerMode(const Module &M);

  bool runOnModule(Module &M) override;

  /// Analyze module to create stable function into LocalFunctionMap.
  void analyze(Module &M);

  /// Emit LocalFunctionMap into __llvm_merge section.
  void emitFunctionMap(Module &M);

  /// Merge functions in the module using the global function map.
  bool merge(Module &M, const StableFunctionMap *FunctionMap);
};

} // end namespace llvm
#endif // PIKA_TRANSFORMS_UTILS_GLOBALMERGEFUNCTIONS_H
