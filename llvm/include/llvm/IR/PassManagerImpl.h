//===- PassManagerImpl.h - Pass management infrastructure -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Provides implementations for PassManager and AnalysisManager template
/// methods. These classes should be explicitly instantiated for any IR unit,
/// and files doing the explicit instantiation should include this header.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_PASSMANAGERIMPL_H
#define LLVM_IR_PASSMANAGERIMPL_H

#include "llvm/IR/PassManager.h"

namespace llvm {

template <typename IRUnitT, typename... ExtraArgTs>
inline AnalysisManager<IRUnitT, ExtraArgTs...>::AnalysisManager(
    bool DebugLogging)
    : DebugLogging(DebugLogging) {}

template <typename IRUnitT, typename... ExtraArgTs>
inline AnalysisManager<IRUnitT, ExtraArgTs...>::AnalysisManager(
    AnalysisManager &&) = default;

template <typename IRUnitT, typename... ExtraArgTs>
inline AnalysisManager<IRUnitT, ExtraArgTs...> &
AnalysisManager<IRUnitT, ExtraArgTs...>::operator=(AnalysisManager &&) =
    default;

template <typename IRUnitT, typename... ExtraArgTs>
inline void
AnalysisManager<IRUnitT, ExtraArgTs...>::clear(IRUnitT &IR,
                                               llvm::StringRef Name) {
  if (DebugLogging)
    dbgs() << "Clearing all analysis results for: " << Name << "\n";

  auto ResultsListI = AnalysisResultLists.find(&IR);
  if (ResultsListI == AnalysisResultLists.end())
    return;
  // Delete the map entries that point into the results list.
  for (auto &IDAndResult : ResultsListI->second)
    AnalysisResults.erase({IDAndResult.first, &IR});

  // And actually destroy and erase the results associated with this IR.
  AnalysisResultLists.erase(ResultsListI);
}

template <typename IRUnitT, typename... ExtraArgTs>
inline typename AnalysisManager<IRUnitT, ExtraArgTs...>::ResultConceptT &
AnalysisManager<IRUnitT, ExtraArgTs...>::getResultImpl(
    AnalysisKey *ID, IRUnitT &IR, ExtraArgTs... ExtraArgs) {
  typename AnalysisResultMapT::iterator RI;
  bool Inserted;
  std::tie(RI, Inserted) = AnalysisResults.insert(std::make_pair(
      std::make_pair(ID, &IR), typename AnalysisResultListT::iterator()));

  // If we don't have a cached result for this function, look up the pass and
  // run it to produce a result, which we then add to the cache.
  if (Inserted) {
    auto &P = this->lookUpPass(ID);
    if (DebugLogging)
      dbgs() << "Running analysis: " << P.name() << " on " << IR.getName()
             << "\n";

    PassInstrumentation PI;
    if (ID != PassInstrumentationAnalysis::ID()) {
      PI = getResult<PassInstrumentationAnalysis>(IR, ExtraArgs...);
      PI.runBeforeAnalysis(P, IR);
    }

    AnalysisResultListT &ResultList = AnalysisResultLists[&IR];
    ResultList.emplace_back(ID, P.run(IR, *this, ExtraArgs...));

    PI.runAfterAnalysis(P, IR);

    // P.run may have inserted elements into AnalysisResults and invalidated
    // RI.
    RI = AnalysisResults.find({ID, &IR});
    assert(RI != AnalysisResults.end() && "we just inserted it!");

    RI->second = std::prev(ResultList.end());
  }

  return *RI->second->second;
}

template <typename IRUnitT, typename... ExtraArgTs>
inline void AnalysisManager<IRUnitT, ExtraArgTs...>::invalidate(
    IRUnitT &IR, const PreservedAnalyses &PA) {
  // We're done if all analyses on this IR unit are preserved.
  if (PA.allAnalysesInSetPreserved<AllAnalysesOn<IRUnitT>>())
    return;

  if (DebugLogging)
    dbgs() << "Invalidating all non-preserved analyses for: " << IR.getName()
           << "\n";

  // Track whether each analysis's result is invalidated in
  // IsResultInvalidated.
  SmallDenseMap<AnalysisKey *, bool, 8> IsResultInvalidated;
  Invalidator Inv(IsResultInvalidated, AnalysisResults);
  AnalysisResultListT &ResultsList = AnalysisResultLists[&IR];
  for (auto &AnalysisResultPair : ResultsList) {
    // This is basically the same thing as Invalidator::invalidate, but we
    // can't call it here because we're operating on the type-erased result.
    // Moreover if we instead called invalidate() directly, it would do an
    // unnecessary look up in ResultsList.
    AnalysisKey *ID = AnalysisResultPair.first;
    auto &Result = *AnalysisResultPair.second;

    auto IMapI = IsResultInvalidated.find(ID);
    if (IMapI != IsResultInvalidated.end())
      // This result was already handled via the Invalidator.
      continue;

    // Try to invalidate the result, giving it the Invalidator so it can
    // recursively query for any dependencies it has and record the result.
    // Note that we cannot reuse 'IMapI' here or pre-insert the ID, as
    // Result.invalidate may insert things into the map, invalidating our
    // iterator.
    bool Inserted =
        IsResultInvalidated.insert({ID, Result.invalidate(IR, PA, Inv)}).second;
    (void)Inserted;
    assert(Inserted && "Should never have already inserted this ID, likely "
                       "indicates a cycle!");
  }

  // Now erase the results that were marked above as invalidated.
  if (!IsResultInvalidated.empty()) {
    for (auto I = ResultsList.begin(), E = ResultsList.end(); I != E;) {
      AnalysisKey *ID = I->first;
      if (!IsResultInvalidated.lookup(ID)) {
        ++I;
        continue;
      }

      if (DebugLogging)
        dbgs() << "Invalidating analysis: " << this->lookUpPass(ID).name()
               << " on " << IR.getName() << "\n";

      I = ResultsList.erase(I);
      AnalysisResults.erase({ID, &IR});
    }
  }

  if (ResultsList.empty())
    AnalysisResultLists.erase(&IR);
}
} // end namespace llvm

#endif // LLVM_IR_PASSMANAGERIMPL_H
