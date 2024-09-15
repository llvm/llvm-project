//===- llvm/Analysis/DebugInfoCache.h - debug info cache --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains an analysis that builds a cache of debug info for each
// DICompileUnit in a module.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DEBUGINFOCACHE_H
#define LLVM_ANALYSIS_DEBUGINFOCACHE_H

#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

/// Processes and caches debug info for each DICompileUnit in a module.
///
/// The result of the analysis is a set of DebugInfoFinders primed on their
/// respective DICompileUnit. Such DebugInfoFinders can be used to speed up
/// function cloning which otherwise requires an expensive traversal of
/// DICompileUnit-level debug info. See an example usage in CoroSplit.
class DebugInfoCache {
public:
  using DIFinderCache = SmallDenseMap<const DICompileUnit *, DebugInfoFinder>;
  DIFinderCache Result;

  DebugInfoCache(const Module &M);

  bool invalidate(Module &, const PreservedAnalyses &,
                  ModuleAnalysisManager::Invalidator &);
};

class DebugInfoCacheAnalysis
    : public AnalysisInfoMixin<DebugInfoCacheAnalysis> {
  friend AnalysisInfoMixin<DebugInfoCacheAnalysis>;
  static AnalysisKey Key;

public:
  using Result = DebugInfoCache;
  Result run(Module &M, ModuleAnalysisManager &);
};
} // namespace llvm

#endif
