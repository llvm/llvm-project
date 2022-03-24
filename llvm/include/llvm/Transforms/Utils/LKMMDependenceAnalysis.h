//===- llvm/Transforms/LKMMDependenceAnalysis.h - LKMM Deps -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains all declarations / definitions required for LKMM
/// dependence analysis. Implementations live in LKMMDependenceAnalysis.cpp.
///
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Casting.h"
#include <list>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>

#ifndef LLVM_TRANSFORMS_UTILS_LKMMDEPENDENCEANALYSIS_H
#define LLVM_TRANSFORMS_UTILS_LKMMDEPENDENCEANALYSIS_H

namespace llvm {
namespace {
// FIXME Is there a more elegant way of dealing with duplicate IDs (preferably
//  getting eliminating the problem all together)?

// The IDReMap type alias represents the map of IDs to sets of alias IDs which
// verification contexts use for remapping duplicate IDs. Duplicate IDs appear
// when an annotated instruction is duplicated as part of optimizations.
using IDReMap =
    std::unordered_map<std::string, std::unordered_set<std::string>>;

struct VerBFSResult;
class VerDepHalf;
} // namespace

//===----------------------------------------------------------------------===//
// The Annotation Pass
//===----------------------------------------------------------------------===//

class AnnotateLKMMDeps : public PassInfoMixin<AnnotateLKMMDeps> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

//===----------------------------------------------------------------------===//
// The Verification Pass
//===----------------------------------------------------------------------===//

class VerifyLKMMDeps : public PassInfoMixin<VerifyLKMMDeps> {
public:
  VerifyLKMMDeps()
      : RemappedIDs(std::make_shared<IDReMap>()),
        VerifiedIDs(std::make_shared<std::unordered_set<std::string>>()),
        PrintedBrokenIDs(), PrintedModules() {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

private:
  std::shared_ptr<IDReMap> RemappedIDs;

  std::shared_ptr<std::unordered_set<std::string>> VerifiedIDs;

  std::unordered_set<std::string> PrintedBrokenIDs;

  std::unordered_set<Module *> PrintedModules;

  /// Prints broken dependencies.
  ///
  /// \param BFSRes
  void printBrokenDeps(VerBFSResult *IBFSRes);

  void printBrokenDep(VerDepHalf &Beg, VerDepHalf &End, const std::string &ID);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_CUSTOMMEMDEP_H