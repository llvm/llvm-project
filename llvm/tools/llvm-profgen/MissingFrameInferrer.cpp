//===-- MissingFrameInferrer.cpp - Missing frame inferrer --------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MissingFrameInferrer.h"
#include "Options.h"
#include "PerfReader.h"
#include "ProfiledBinary.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/Statistic.h"
#include <algorithm>
#include <cstdint>
#include <queue>
#include <sys/types.h>

#define DEBUG_TYPE "missing-frame-inferrer"

using namespace llvm;
using namespace sampleprof;

STATISTIC(TailCallUniReachable,
          "Number of frame pairs reachable via a unique tail call path");
STATISTIC(TailCallMultiReachable,
          "Number of frame pairs reachable via a multiple tail call paths");
STATISTIC(TailCallUnreachable,
          "Number of frame pairs unreachable via any tail call path");
STATISTIC(TailCallFuncSingleTailCalls,
          "Number of functions with single tail call site");
STATISTIC(TailCallFuncMultipleTailCalls,
          "Number of functions with multiple tail call sites");
STATISTIC(TailCallMaxTailCallPath, "Length of the longest tail call path");

static cl::opt<uint32_t>
    MaximumSearchDepth("max-search-depth", cl::init(UINT32_MAX - 1),
                       cl::desc("The maximum number of tail calls in a missing "
                                "frame path"),
                       cl::cat(ProfGenCategory));

void MissingFrameInferrer::initialize(
    const ContextSampleCounterMap *SampleCounters) {
  // Refine call edges based on LBR samples.
  if (SampleCounters) {
    DenseMap<uint64_t, DenseSet<uint64_t>> SampledCalls;
    DenseMap<uint64_t, DenseSet<uint64_t>> SampledTailCalls;

    // Populate SampledCalls based on static call sites. Similarly to
    // SampledTailCalls.
    for (const auto &CI : *SampleCounters) {
      for (auto Item : CI.second.BranchCounter) {
        auto From = Item.first.first;
        auto To = Item.first.second;
        if (CallEdges.count(From)) {
          assert(CallEdges[From].size() == 1 &&
                 "A callsite should only appear once with either a known or a "
                 "zero (unknown) target value at this point");
          SampledCalls[From].insert(To);
        }
        if (TailCallEdges.count(From)) {
          assert(TailCallEdges[From].size() == 1 &&
                 "A callsite should only appear once with either a known or a "
                 "zero (unknown) target value at this point");
          FuncRange *FromFRange = Binary->findFuncRange(From);
          FuncRange *ToFRange = Binary->findFuncRange(To);
          if (FromFRange != ToFRange)
            SampledTailCalls[From].insert(To);
        }
      }
    }

    // Replace static edges with dynamic edges.
    CallEdges = std::move(SampledCalls);
    TailCallEdges = std::move(SampledTailCalls);
  }

  // Populate function-based edges. This is to speed up address to function
  // translation.
  for (auto Call : CallEdges)
    for (auto Target : Call.second)
      if (FuncRange *ToFRange = Binary->findFuncRange(Target))
        CallEdgesF[Call.first].insert(ToFRange->Func);

  for (auto Call : TailCallEdges) {
    for (auto Target : Call.second) {
      if (FuncRange *ToFRange = Binary->findFuncRange(Target)) {
        TailCallEdgesF[Call.first].insert(ToFRange->Func);
        TailCallTargetFuncs.insert(ToFRange->Func);
      }
    }
    if (FuncRange *FromFRange = Binary->findFuncRange(Call.first))
      FuncToTailCallMap[FromFRange->Func].push_back(Call.first);
  }

#if LLVM_ENABLE_STATS
  for (auto F : FuncToTailCallMap) {
    assert(F.second.size() > 0 && "");
    if (F.second.size() > 1)
      TailCallFuncMultipleTailCalls++;
    else
      TailCallFuncSingleTailCalls++;
  }
#endif

#ifndef NDEBUG
  auto PrintCallTargets =
      [&](const DenseMap<uint64_t, DenseSet<uint64_t>> &CallTargets,
          bool IsTailCall) {
        for (const auto &Targets : CallTargets) {
          for (const auto &Target : Targets.second) {
            dbgs() << (IsTailCall ? "TailCall" : "Call");
            dbgs() << " From " << format("%8" PRIx64, Targets.first) << " to "
                   << format("%8" PRIx64, Target) << "\n";
          }
        }
      };

  LLVM_DEBUG({
    dbgs() << "============================\n ";
    dbgs() << "Call targets:\n";
    PrintCallTargets(CallEdges, false);
    dbgs() << "\nTail call targets:\n";
    PrintCallTargets(TailCallEdges, true);
    dbgs() << "============================\n";
  });
#endif
}

uint64_t MissingFrameInferrer::computeUniqueTailCallPath(
    BinaryFunction *From, BinaryFunction *To, SmallVectorImpl<uint64_t> &Path) {
  // Search for a unique path comprised of only tail call edges for a given
  // source and target frame address on the a tail call graph that consists of
  // only tail call edges. Note that only a unique path counts. Multiple paths
  // are treated unreachable.
  if (From == To)
    return 1;

  // If already computed, return the cached result.
  auto I = UniquePaths.find({From, To});
  if (I != UniquePaths.end()) {
    Path.append(I->second.begin(), I->second.end());
    return 1;
  }

  auto J = NonUniquePaths.find({From, To});
  if (J != NonUniquePaths.end()) {
    return J->second;
  }

  uint64_t Pos = Path.size();
  uint64_t NumPaths = countTailCallPaths(From, To, Path);

  // Cache the result.
  if (NumPaths == 1) {
    UniquePaths[{From, To}].assign(Path.begin() + Pos, Path.end());
#if LLVM_ENABLE_STATS
    auto &LocalPath = UniquePaths[{From, To}];
    assert((LocalPath.size() <= MaximumSearchDepth + 1) &&
           "Path should not be longer than the maximum searching depth");
    TailCallMaxTailCallPath = std::max(uint64_t(LocalPath.size()),
                                       TailCallMaxTailCallPath.getValue());
#endif
  } else {
    NonUniquePaths[{From, To}] = NumPaths;
  }

  return NumPaths;
}

uint64_t MissingFrameInferrer::countTailCallPaths(
    BinaryFunction *From, BinaryFunction *To,
    SmallVectorImpl<uint64_t> &UniquePath) {
  struct TailCall {
    uint64_t Site = 0;
    BinaryFunction *Caller = nullptr;
    BinaryFunction *Callee = nullptr;
  };
  auto ForEachTailCall = [&](BinaryFunction *Caller, auto Callback) {
    auto Sites = FuncToTailCallMap.find(Caller);
    if (Sites == FuncToTailCallMap.end())
      return;
    for (uint64_t Site : Sites->second) {
      auto Callees = TailCallEdgesF.find(Site);
      if (Callees == TailCallEdgesF.end())
        continue;
      for (BinaryFunction *Callee : Callees->second)
        Callback(TailCall{Site, Caller, Callee});
    }
  };

  // Search breadth first for a shortest path of tail calls to To, without
  // the tail call Skipped, recording the tail call reaching each function.
  DenseMap<BinaryFunction *, TailCall> ReachedBy;
  auto FindPath = [&](const TailCall *Skipped) {
    ReachedBy.clear();
    ReachedBy.try_emplace(From);
    SmallVector<BinaryFunction *, 16> Level = {From}, NextLevel;
    for (uint32_t Depth = 0; Depth < MaximumSearchDepth && !Level.empty();
         ++Depth) {
      for (BinaryFunction *Caller : Level)
        ForEachTailCall(Caller, [&](const TailCall &Call) {
          if (Skipped && Call.Site == Skipped->Site &&
              Call.Callee == Skipped->Callee)
            return;
          if (ReachedBy.try_emplace(Call.Callee, Call).second)
            NextLevel.push_back(Call.Callee);
        });
      if (ReachedBy.contains(To))
        return true;
      std::swap(Level, NextLevel);
      NextLevel.clear();
    }
    return false;
  };

  if (!FindPath(nullptr))
    return 0;
  SmallVector<TailCall, 8> Path;
  for (BinaryFunction *Func = To; Func != From; Func = Path.back().Caller)
    Path.push_back(ReachedBy.lookup(Func));
  std::reverse(Path.begin(), Path.end());

  // Another simple path leaves this one at a function through another of its
  // tail calls, so it does not take the tail call of this path there, and a
  // shortest path without that tail call is simple. Only functions with
  // several tail calls can be left that way.
  for (const TailCall &Call : Path) {
    unsigned NumTailCalls = 0;
    ForEachTailCall(Call.Caller, [&](const TailCall &) { ++NumTailCalls; });
    if (NumTailCalls > 1 && FindPath(&Call))
      return 2;
  }

  for (const TailCall &Call : Path)
    UniquePath.push_back(Call.Site);
  return 1;
}

bool MissingFrameInferrer::inferMissingFrames(
    uint64_t From, uint64_t To, SmallVectorImpl<uint64_t> &UniquePath) {
  assert(!TailCallEdgesF.count(From) &&
         "transition between From and To cannot be via a tailcall otherwise "
         "they would not show up at the same time");
  UniquePath.push_back(From);
  uint64_t Pos = UniquePath.size();

  FuncRange *ToFRange = Binary->findFuncRange(To);
  if (!ToFRange)
    return false;

  // Bail out if caller has no known outgoing call edges.
  auto It = CallEdgesF.find(From);
  if (It == CallEdgesF.end())
    return false;

  // Done with the inference if the calle is reachable via a single callsite.
  // This may not be accurate but it improves the search throughput.
  if (llvm::is_contained(It->second, ToFRange->Func))
    return true;

  // Bail out if callee is not tailcall reachable at all.
  if (!TailCallTargetFuncs.contains(ToFRange->Func))
    return false;

  uint64_t NumPaths = 0;
  for (auto Target : It->second) {
    NumPaths +=
        computeUniqueTailCallPath(Target, ToFRange->Func, UniquePath);
    // Stop analyzing the remaining if we are already seeing more than one
    // reachable paths.
    if (NumPaths > 1)
      break;
  }

  // Undo already-computed path if it is not unique.
  if (NumPaths != 1) {
    UniquePath.pop_back_n(UniquePath.size() - Pos);
    assert(UniquePath.back() == From && "broken path");
  }

#if LLVM_ENABLE_STATS
  if (NumPaths == 1) {
    if (ReachableViaUniquePaths.insert({From, ToFRange->StartAddress}).second)
      TailCallUniReachable++;
  } else if (NumPaths == 0) {
    if (Unreachables.insert({From, ToFRange->StartAddress}).second) {
      TailCallUnreachable++;
      LLVM_DEBUG(dbgs() << "No path found from "
                        << format("%8" PRIx64 ":", From) << " to "
                        << format("%8" PRIx64 ":", ToFRange->StartAddress)
                        << "\n");
    }
  } else if (NumPaths > 1) {
    if (ReachableViaMultiPaths.insert({From, ToFRange->StartAddress})
            .second) {
      TailCallMultiReachable++;
      LLVM_DEBUG(dbgs() << "Multiple paths found from "
                        << format("%8" PRIx64 ":", From) << " to "
                        << format("%8" PRIx64 ":", ToFRange->StartAddress)
                        << "\n");
    }
  }
#endif

  return NumPaths == 1;
}

void MissingFrameInferrer::inferMissingFrames(
    const SmallVectorImpl<uint64_t> &Context,
    SmallVectorImpl<uint64_t> &NewContext) {
  if (Context.size() == 1) {
    NewContext = Context;
    return;
  }

  NewContext.clear();
  for (uint64_t I = 1; I < Context.size(); I++) {
    inferMissingFrames(Context[I - 1], Context[I], NewContext);
  }
  NewContext.push_back(Context.back());

  assert((NewContext.size() >= Context.size()) &&
         "Inferred context should include all frames in the original context");
  assert((NewContext.size() > Context.size() || NewContext == Context) &&
         "Inferred context should be exactly the same "
         "with the original context");
}
