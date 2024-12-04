//===- ExtraFunctionPassManager.h - Run Optimizations on Demand -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file provides a pass manager that only runs its passes if the
/// provided marker analysis has been preserved, together with a class to
/// define such a marker analysis.
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_EXTRAPASSMANAGER_H
#define LLVM_TRANSFORMS_UTILS_EXTRAPASSMANAGER_H

#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"

namespace llvm {

/// A marker analysis to determine if extra passes should be run on demand.
/// Passes requesting extra transformations to run need to request and preserve
/// this analysis.
template <typename MarkerTy> struct ShouldRunExtraPasses {
  struct Result {
    bool invalidate(Function &F, const PreservedAnalyses &PA,
                    FunctionAnalysisManager::Invalidator &) {
      // Check whether the analysis has been explicitly invalidated. Otherwise,
      // it remains preserved.
      auto PAC = PA.getChecker<MarkerTy>();
      return !PAC.preservedWhenStateless();
    }

    bool invalidate(Loop &L, const PreservedAnalyses &PA,
                    LoopAnalysisManager::Invalidator &) {
      // Check whether the analysis has been explicitly invalidated. Otherwise,
      // it remains preserved.
      auto PAC = PA.getChecker<MarkerTy>();
      return !PAC.preservedWhenStateless();
    }
  };

  Result run(Function &F, FunctionAnalysisManager &FAM) { return Result(); }

  Result run(Loop &L, LoopAnalysisManager &AM,
             LoopStandardAnalysisResults &AR) {
    return Result();
  }
};

/// A pass manager to run a set of extra function passes if the
/// ShouldRunExtraPasses marker analysis is present. This allows passes to
/// request additional transformations on demand. An example is extra
/// simplifications after loop-vectorization, if runtime checks have been added.
template <typename MarkerTy>
struct ExtraFunctionPassManager : public FunctionPassManager {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    auto PA = PreservedAnalyses::all();
    if (AM.getCachedResult<MarkerTy>(F))
      PA.intersect(FunctionPassManager::run(F, AM));
    PA.abandon<MarkerTy>();
    return PA;
  }
};

/// A pass manager to run a set of extra loop passes if the MarkerTy analysis is
/// present. This allows passes to request additional transformations on demand.
/// An example is doing additional runs of SimpleLoopUnswitch.
template <typename MarkerTy>
struct ExtraLoopPassManager : public LoopPassManager {
  PreservedAnalyses run(Loop &L, LoopAnalysisManager &AM,
                        LoopStandardAnalysisResults &AR, LPMUpdater &U) {
    auto PA = PreservedAnalyses::all();
    if (AM.getCachedResult<MarkerTy>(L))
      PA.intersect(LoopPassManager::run(L, AM, AR, U));
    PA.abandon<MarkerTy>();
    return PA;
  }
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_EXTRAPASSMANAGER_H
