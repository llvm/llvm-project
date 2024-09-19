//===- PGOCtxProfFlattening.cpp - Contextual Instr. Flattening ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Flattens the contextual profile and lowers it to MD_prof.
// This should happen after all IPO (which is assumed to have maintained the
// contextual profile) happened. Flattening consists of summing the values at
// the same index of the counters belonging to all the contexts of a function.
// The lowering consists of materializing the counter values to function
// entrypoint counts and branch probabilities.
//
// This pass also removes contextual instrumentation, which has been kept around
// to facilitate its functionality.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/PGOCtxProfFlattening.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Analysis/CtxProfAnalysis.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ProfileSummary.h"
#include "llvm/ProfileData/ProfileCommon.h"
#include "llvm/Transforms/Instrumentation/PGOInstrumentation.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <deque>

using namespace llvm;

namespace {

class ProfileAnnotator final {
  class BBInfo;
  struct EdgeInfo {
    BBInfo *const Src;
    BBInfo *const Dest;
    std::optional<uint64_t> Count;

    explicit EdgeInfo(BBInfo &Src, BBInfo &Dest) : Src(&Src), Dest(&Dest) {}
  };

  class BBInfo {
    std::optional<uint64_t> Count;
    // OutEdges is dimensioned to match the number of terminator operands.
    // Entries in the vector match the index in the terminator operand list. In
    // some cases - see `shouldExcludeEdge` and its implementation - an entry
    // will be nullptr.
    // InEdges doesn't have the above constraint.
    SmallVector<EdgeInfo *> OutEdges;
    SmallVector<EdgeInfo *> InEdges;
    size_t UnknownCountOutEdges = 0;
    size_t UnknownCountInEdges = 0;

    // Pass AssumeAllKnown when we try to propagate counts from edges to BBs -
    // because all the edge counters must be known.
    // Return std::nullopt if there were no edges to sum. The user can decide
    // how to interpret that.
    std::optional<uint64_t> getEdgeSum(const SmallVector<EdgeInfo *> &Edges,
                                       bool AssumeAllKnown) const {
      std::optional<uint64_t> Sum;
      for (const auto *E : Edges) {
        // `Edges` may be `OutEdges`, case in which `E` could be nullptr.
        if (E) {
          if (!Sum.has_value())
            Sum = 0;
          *Sum += (AssumeAllKnown ? *E->Count : E->Count.value_or(0U));
        }
      }
      return Sum;
    }

    bool computeCountFrom(const SmallVector<EdgeInfo *> &Edges) {
      assert(!Count.has_value());
      Count = getEdgeSum(Edges, true);
      return Count.has_value();
    }

    void setSingleUnknownEdgeCount(SmallVector<EdgeInfo *> &Edges) {
      uint64_t KnownSum = getEdgeSum(Edges, false).value_or(0U);
      uint64_t EdgeVal = *Count > KnownSum ? *Count - KnownSum : 0U;
      EdgeInfo *E = nullptr;
      for (auto *I : Edges)
        if (I && !I->Count.has_value()) {
          E = I;
#ifdef NDEBUG
          break;
#else
          assert((!E || E == I) &&
                 "Expected exactly one edge to have an unknown count, "
                 "found a second one");
          continue;
#endif
        }
      assert(E && "Expected exactly one edge to have an unknown count");
      assert(!E->Count.has_value());
      E->Count = EdgeVal;
      assert(E->Src->UnknownCountOutEdges > 0);
      assert(E->Dest->UnknownCountInEdges > 0);
      --E->Src->UnknownCountOutEdges;
      --E->Dest->UnknownCountInEdges;
    }

  public:
    BBInfo(size_t NumInEdges, size_t NumOutEdges, std::optional<uint64_t> Count)
        : Count(Count) {
      // For in edges, we just want to pre-allocate enough space, since we know
      // it at this stage. For out edges, we will insert edges at the indices
      // corresponding to positions in this BB's terminator instruction, so we
      // construct a default (nullptr values)-initialized vector. A nullptr edge
      // corresponds to those that are excluded (see shouldExcludeEdge).
      InEdges.reserve(NumInEdges);
      OutEdges.resize(NumOutEdges);
    }

    bool tryTakeCountFromKnownOutEdges(const BasicBlock &BB) {
      if (!UnknownCountOutEdges) {
        return computeCountFrom(OutEdges);
      }
      return false;
    }

    bool tryTakeCountFromKnownInEdges(const BasicBlock &BB) {
      if (!UnknownCountInEdges) {
        return computeCountFrom(InEdges);
      }
      return false;
    }

    void addInEdge(EdgeInfo &Info) {
      InEdges.push_back(&Info);
      ++UnknownCountInEdges;
    }

    // For the out edges, we care about the position we place them in, which is
    // the position in terminator instruction's list (at construction). Later,
    // we build branch_weights metadata with edge frequency values matching
    // these positions.
    void addOutEdge(size_t Index, EdgeInfo &Info) {
      OutEdges[Index] = &Info;
      ++UnknownCountOutEdges;
    }

    bool hasCount() const { return Count.has_value(); }

    bool trySetSingleUnknownInEdgeCount() {
      if (UnknownCountInEdges == 1) {
        setSingleUnknownEdgeCount(InEdges);
        return true;
      }
      return false;
    }

    bool trySetSingleUnknownOutEdgeCount() {
      if (UnknownCountOutEdges == 1) {
        setSingleUnknownEdgeCount(OutEdges);
        return true;
      }
      return false;
    }
    size_t getNumOutEdges() const { return OutEdges.size(); }

    uint64_t getEdgeCount(size_t Index) const {
      if (auto *E = OutEdges[Index])
        return *E->Count;
      return 0U;
    }
  };

  Function &F;
  const SmallVectorImpl<uint64_t> &Counters;
  // To be accessed through getBBInfo() after construction.
  std::map<const BasicBlock *, BBInfo> BBInfos;
  std::vector<EdgeInfo> EdgeInfos;
  InstrProfSummaryBuilder &PB;

  // This is an adaptation of PGOUseFunc::populateCounters.
  // FIXME(mtrofin): look into factoring the code to share one implementation.
  void propagateCounterValues(const SmallVectorImpl<uint64_t> &Counters) {
    bool KeepGoing = true;
    while (KeepGoing) {
      KeepGoing = false;
      for (const auto &BB : F) {
        auto &Info = getBBInfo(BB);
        if (!Info.hasCount())
          KeepGoing |= Info.tryTakeCountFromKnownOutEdges(BB) ||
                       Info.tryTakeCountFromKnownInEdges(BB);
        if (Info.hasCount()) {
          KeepGoing |= Info.trySetSingleUnknownOutEdgeCount();
          KeepGoing |= Info.trySetSingleUnknownInEdgeCount();
        }
      }
    }
  }
  // The only criteria for exclusion is faux suspend -> exit edges in presplit
  // coroutines. The API serves for readability, currently.
  bool shouldExcludeEdge(const BasicBlock &Src, const BasicBlock &Dest) const {
    return llvm::isPresplitCoroSuspendExitEdge(Src, Dest);
  }

  BBInfo &getBBInfo(const BasicBlock &BB) { return BBInfos.find(&BB)->second; }

  const BBInfo &getBBInfo(const BasicBlock &BB) const {
    return BBInfos.find(&BB)->second;
  }

  // validation function after we propagate the counters: all BBs and edges'
  // counters must have a value.
  bool allCountersAreAssigned() const {
    for (const auto &BBInfo : BBInfos)
      if (!BBInfo.second.hasCount())
        return false;
    for (const auto &EdgeInfo : EdgeInfos)
      if (!EdgeInfo.Count.has_value())
        return false;
    return true;
  }

  /// Check that all paths from the entry basic block that use edges with
  /// non-zero counts arrive at a basic block with no successors (i.e. "exit")
  bool allTakenPathsExit() const {
    std::deque<const BasicBlock *> Worklist;
    DenseSet<const BasicBlock *> Visited;
    Worklist.push_back(&F.getEntryBlock());
    bool HitExit = false;
    while (!Worklist.empty()) {
      const auto *BB = Worklist.front();
      Worklist.pop_front();
      if (!Visited.insert(BB).second)
        continue;
      if (succ_size(BB) == 0) {
        if (isa<UnreachableInst>(BB->getTerminator()))
          return false;
        HitExit = true;
        continue;
      }
      if (succ_size(BB) == 1) {
        Worklist.push_back(BB->getUniqueSuccessor());
        continue;
      }
      const auto &BBInfo = getBBInfo(*BB);
      bool HasAWayOut = false;
      for (auto I = 0U; I < BB->getTerminator()->getNumSuccessors(); ++I) {
        const auto *Succ = BB->getTerminator()->getSuccessor(I);
        if (!shouldExcludeEdge(*BB, *Succ)) {
          if (BBInfo.getEdgeCount(I) > 0) {
            HasAWayOut = true;
            Worklist.push_back(Succ);
          }
        }
      }
      if (!HasAWayOut)
        return false;
    }
    return HitExit;
  }

public:
  ProfileAnnotator(Function &F, const SmallVectorImpl<uint64_t> &Counters,
                   InstrProfSummaryBuilder &PB)
      : F(F), Counters(Counters), PB(PB) {
    assert(!F.isDeclaration());
    assert(!Counters.empty());
    size_t NrEdges = 0;
    for (const auto &BB : F) {
      std::optional<uint64_t> Count;
      if (auto *Ins = CtxProfAnalysis::getBBInstrumentation(
              const_cast<BasicBlock &>(BB))) {
        auto Index = Ins->getIndex()->getZExtValue();
        assert(Index < Counters.size() &&
               "The index must be inside the counters vector by construction - "
               "tripping this assertion indicates a bug in how the contextual "
               "profile is managed by IPO transforms");
        (void)Index;
        Count = Counters[Ins->getIndex()->getZExtValue()];
      } else if (isa<UnreachableInst>(BB.getTerminator())) {
        // The program presumably didn't crash.
        Count = 0;
      }
      auto [It, Ins] =
          BBInfos.insert({&BB, {pred_size(&BB), succ_size(&BB), Count}});
      (void)Ins;
      assert(Ins && "We iterate through the function's BBs, no reason to "
                    "insert one more than once");
      NrEdges += llvm::count_if(successors(&BB), [&](const auto *Succ) {
        return !shouldExcludeEdge(BB, *Succ);
      });
    }
    // Pre-allocate the vector, we want references to its contents to be stable.
    EdgeInfos.reserve(NrEdges);
    for (const auto &BB : F) {
      auto &Info = getBBInfo(BB);
      for (auto I = 0U; I < BB.getTerminator()->getNumSuccessors(); ++I) {
        const auto *Succ = BB.getTerminator()->getSuccessor(I);
        if (!shouldExcludeEdge(BB, *Succ)) {
          auto &EI = EdgeInfos.emplace_back(getBBInfo(BB), getBBInfo(*Succ));
          Info.addOutEdge(I, EI);
          getBBInfo(*Succ).addInEdge(EI);
        }
      }
    }
    assert(EdgeInfos.capacity() == NrEdges &&
           "The capacity of EdgeInfos should have stayed unchanged it was "
           "populated, because we need pointers to its contents to be stable");
  }

  /// Assign branch weights and function entry count. Also update the PSI
  /// builder.
  void assignProfileData() {
    assert(!Counters.empty());
    propagateCounterValues(Counters);
    F.setEntryCount(Counters[0]);
    PB.addEntryCount(Counters[0]);

    for (auto &BB : F) {
      if (succ_size(&BB) < 2)
        continue;
      auto *Term = BB.getTerminator();
      SmallVector<uint64_t, 2> EdgeCounts(Term->getNumSuccessors(), 0);
      uint64_t MaxCount = 0;
      const auto &BBInfo = getBBInfo(BB);
      for (unsigned SuccIdx = 0, Size = BBInfo.getNumOutEdges(); SuccIdx < Size;
           ++SuccIdx) {
        uint64_t EdgeCount = BBInfo.getEdgeCount(SuccIdx);
        if (EdgeCount > MaxCount)
          MaxCount = EdgeCount;
        EdgeCounts[SuccIdx] = EdgeCount;
        PB.addInternalCount(EdgeCount);
      }

      if (MaxCount != 0)
        setProfMetadata(F.getParent(), Term, EdgeCounts, MaxCount);
    }
    assert(allCountersAreAssigned() &&
           "Expected all counters have been assigned.");
    assert(allTakenPathsExit() &&
           "[ctx-prof] Encountered a BB with more than one successor, where "
           "all outgoing edges have a 0 count. This occurs in non-exiting "
           "functions (message pumps, usually) which are not supported in the "
           "contextual profiling case");
  }
};

[[maybe_unused]] bool areAllBBsReachable(const Function &F,
                                         FunctionAnalysisManager &FAM) {
  auto &DT = FAM.getResult<DominatorTreeAnalysis>(const_cast<Function &>(F));
  return llvm::all_of(
      F, [&](const BasicBlock &BB) { return DT.isReachableFromEntry(&BB); });
}

void clearColdFunctionProfile(Function &F) {
  for (auto &BB : F)
    BB.getTerminator()->setMetadata(LLVMContext::MD_prof, nullptr);
  F.setEntryCount(0U);
}

void removeInstrumentation(Function &F) {
  for (auto &BB : F)
    for (auto &I : llvm::make_early_inc_range(BB))
      if (isa<InstrProfCntrInstBase>(I))
        I.eraseFromParent();
}

} // namespace

PreservedAnalyses PGOCtxProfFlatteningPass::run(Module &M,
                                                ModuleAnalysisManager &MAM) {
  // Ensure in all cases the instrumentation is removed: if this module had no
  // roots, the contextual profile would evaluate to false, but there would
  // still be instrumentation.
  // Note: in such cases we leave as-is any other profile info (if present -
  // e.g. synthetic weights, etc) because it wouldn't interfere with the
  // contextual - based one (which would be in other modules)
  auto OnExit = llvm::make_scope_exit([&]() {
    for (auto &F : M)
      removeInstrumentation(F);
  });
  auto &CtxProf = MAM.getResult<CtxProfAnalysis>(M);
  if (!CtxProf)
    return PreservedAnalyses::none();

  const auto FlattenedProfile = CtxProf.flatten();

  InstrProfSummaryBuilder PB(ProfileSummaryBuilder::DefaultCutoffs);
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;

    assert(areAllBBsReachable(
               F, MAM.getResult<FunctionAnalysisManagerModuleProxy>(M)
                      .getManager()) &&
           "Function has unreacheable basic blocks. The expectation was that "
           "DCE was run before.");

    auto It = FlattenedProfile.find(AssignGUIDPass::getGUID(F));
    // If this function didn't appear in the contextual profile, it's cold.
    if (It == FlattenedProfile.end())
      clearColdFunctionProfile(F);
    else {
      ProfileAnnotator S(F, It->second, PB);
      S.assignProfileData();
    }
  }

  auto &PSI = MAM.getResult<ProfileSummaryAnalysis>(M);

  M.setProfileSummary(PB.getSummary()->getMD(M.getContext()),
                      ProfileSummary::Kind::PSK_Instr);
  PSI.refresh();
  return PreservedAnalyses::none();
}
