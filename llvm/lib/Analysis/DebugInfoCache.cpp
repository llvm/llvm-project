//===- llvm/Analysis/DebugInfoCache.cpp - debug info cache ----------------===//
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

#include "llvm/Analysis/DebugInfoCache.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace {
DebugInfoFinder processCompileUnit(DICompileUnit *CU) {
  DebugInfoFinder DIFinder;
  DIFinder.processCompileUnit(CU);

  return DIFinder;
}
} // namespace

DebugInfoCache::DebugInfoCache(const Module &M) {
  for (const auto CU : M.debug_compile_units()) {
    auto DIFinder = processCompileUnit(CU);
    Result[CU] = std::move(DIFinder);
  }
}

bool DebugInfoCache::invalidate(Module &M, const PreservedAnalyses &PA,
                                ModuleAnalysisManager::Invalidator &) {
  // Check whether the analysis has been explicitly invalidated. Otherwise, it's
  // stateless and remains preserved.
  auto PAC = PA.getChecker<DebugInfoCacheAnalysis>();
  return !PAC.preservedWhenStateless();
}

AnalysisKey DebugInfoCacheAnalysis::Key;

DebugInfoCache DebugInfoCacheAnalysis::run(Module &M, ModuleAnalysisManager &) {
  return DebugInfoCache(M);
}
