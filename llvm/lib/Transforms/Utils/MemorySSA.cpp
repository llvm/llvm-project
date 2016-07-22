//===-- MemorySSA.cpp - Memory SSA Builder---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------===//
//
// This file implements the MemorySSA class.
//
//===----------------------------------------------------------------===//
#include "llvm/Transforms/Utils/MemorySSA.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/IteratedDominanceFrontier.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/PHITransAddr.h"
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Transforms/Scalar.h"
#include <algorithm>

#define DEBUG_TYPE "memoryssa"
using namespace llvm;
STATISTIC(NumClobberCacheLookups, "Number of Memory SSA version cache lookups");
STATISTIC(NumClobberCacheHits, "Number of Memory SSA version cache hits");
STATISTIC(NumClobberCacheInserts, "Number of MemorySSA version cache inserts");

INITIALIZE_PASS_BEGIN(MemorySSAWrapperPass, "memoryssa", "Memory SSA", false,
                      true)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_END(MemorySSAWrapperPass, "memoryssa", "Memory SSA", false,
                    true)

INITIALIZE_PASS_BEGIN(MemorySSAPrinterLegacyPass, "print-memoryssa",
                      "Memory SSA Printer", false, false)
INITIALIZE_PASS_DEPENDENCY(MemorySSAWrapperPass)
INITIALIZE_PASS_END(MemorySSAPrinterLegacyPass, "print-memoryssa",
                    "Memory SSA Printer", false, false)

static cl::opt<bool>
    VerifyMemorySSA("verify-memoryssa", cl::init(false), cl::Hidden,
                    cl::desc("Verify MemorySSA in legacy printer pass."));

namespace llvm {
/// \brief An assembly annotator class to print Memory SSA information in
/// comments.
class MemorySSAAnnotatedWriter : public AssemblyAnnotationWriter {
  friend class MemorySSA;
  const MemorySSA *MSSA;

public:
  MemorySSAAnnotatedWriter(const MemorySSA *M) : MSSA(M) {}

  virtual void emitBasicBlockStartAnnot(const BasicBlock *BB,
                                        formatted_raw_ostream &OS) {
    if (MemoryAccess *MA = MSSA->getMemoryAccess(BB))
      OS << "; " << *MA << "\n";
  }

  virtual void emitInstructionAnnot(const Instruction *I,
                                    formatted_raw_ostream &OS) {
    if (MemoryAccess *MA = MSSA->getMemoryAccess(I))
      OS << "; " << *MA << "\n";
  }
};
}

namespace {
struct UpwardsMemoryQuery {
  // True if our original query started off as a call
  bool IsCall;
  // The pointer location we started the query with. This will be empty if
  // IsCall is true.
  MemoryLocation StartingLoc;
  // This is the instruction we were querying about.
  const Instruction *Inst;
  // The MemoryAccess we actually got called with, used to test local domination
  const MemoryAccess *OriginalAccess;

  UpwardsMemoryQuery()
      : IsCall(false), Inst(nullptr), OriginalAccess(nullptr) {}

  UpwardsMemoryQuery(const Instruction *Inst, const MemoryAccess *Access)
      : IsCall(ImmutableCallSite(Inst)), Inst(Inst), OriginalAccess(Access) {
    if (!IsCall)
      StartingLoc = MemoryLocation::get(Inst);
  }
};

static bool instructionClobbersQuery(MemoryDef *MD, const MemoryLocation &Loc,
                                     const UpwardsMemoryQuery &Query,
                                     AliasAnalysis &AA) {
  Instruction *DefMemoryInst = MD->getMemoryInst();
  assert(DefMemoryInst && "Defining instruction not actually an instruction");

  if (!Query.IsCall)
    return AA.getModRefInfo(DefMemoryInst, Loc) & MRI_Mod;

  ModRefInfo I = AA.getModRefInfo(DefMemoryInst, ImmutableCallSite(Query.Inst));
  return I != MRI_NoModRef;
}

/// Cache for our caching MemorySSA walker.
class WalkerCache {
  DenseMap<ConstMemoryAccessPair, MemoryAccess *> Accesses;
  DenseMap<const MemoryAccess *, MemoryAccess *> Calls;

public:
  MemoryAccess *lookup(const MemoryAccess *MA, const MemoryLocation &Loc,
                       bool IsCall) const {
    ++NumClobberCacheLookups;
    MemoryAccess *R = IsCall ? Calls.lookup(MA) : Accesses.lookup({MA, Loc});
    if (R)
      ++NumClobberCacheHits;
    return R;
  }

  bool insert(const MemoryAccess *MA, MemoryAccess *To,
              const MemoryLocation &Loc, bool IsCall) {
    // This is fine for Phis, since there are times where we can't optimize
    // them.  Making a def its own clobber is never correct, though.
    assert((MA != To || isa<MemoryPhi>(MA)) &&
           "Something can't clobber itself!");

    ++NumClobberCacheInserts;
    bool Inserted;
    if (IsCall)
      Inserted = Calls.insert({MA, To}).second;
    else
      Inserted = Accesses.insert({{MA, Loc}, To}).second;

    return Inserted;
  }

  bool remove(const MemoryAccess *MA, const MemoryLocation &Loc, bool IsCall) {
    return IsCall ? Calls.erase(MA) : Accesses.erase({MA, Loc});
  }

  void clear() {
    Accesses.clear();
    Calls.clear();
  }

  bool contains(const MemoryAccess *MA) const {
    for (auto &P : Accesses)
      if (P.first.first == MA || P.second == MA)
        return true;
    for (auto &P : Calls)
      if (P.first == MA || P.second == MA)
        return true;
    return false;
  }
};

/// Walks the defining uses of MemoryDefs. Stops after we hit something that has
/// no defining use (e.g. a MemoryPhi or liveOnEntry). Note that, when comparing
/// against a null def_chain_iterator, this will compare equal only after
/// walking said Phi/liveOnEntry.
struct def_chain_iterator
    : public iterator_facade_base<def_chain_iterator, std::forward_iterator_tag,
                                  MemoryAccess *> {
  def_chain_iterator() : MA(nullptr) {}
  def_chain_iterator(MemoryAccess *MA) : MA(MA) {}

  MemoryAccess *operator*() const { return MA; }

  def_chain_iterator &operator++() {
    // N.B. liveOnEntry has a null defining access.
    if (auto *MUD = dyn_cast<MemoryUseOrDef>(MA))
      MA = MUD->getDefiningAccess();
    else
      MA = nullptr;
    return *this;
  }

  bool operator==(const def_chain_iterator &O) const { return MA == O.MA; }

private:
  MemoryAccess *MA;
};

static iterator_range<def_chain_iterator>
def_chain(MemoryAccess *MA, MemoryAccess *UpTo = nullptr) {
#ifdef EXPENSIVE_CHECKS
  assert((!UpTo || find(def_chain(MA), UpTo) != def_chain_iterator()) &&
         "UpTo isn't in the def chain!");
#endif
  return make_range(def_chain_iterator(MA), def_chain_iterator(UpTo));
}

/// Verifies that `Start` is clobbered by `ClobberAt`, and that nothing
/// inbetween `Start` and `ClobberAt` can clobbers `Start`.
///
/// This is meant to be as simple and self-contained as possible. Because it
/// uses no cache, etc., it can be relatively expensive.
///
/// \param Start     The MemoryAccess that we want to walk from.
/// \param ClobberAt A clobber for Start.
/// \param StartLoc  The MemoryLocation for Start.
/// \param MSSA      The MemorySSA isntance that Start and ClobberAt belong to.
/// \param Query     The UpwardsMemoryQuery we used for our search.
/// \param AA        The AliasAnalysis we used for our search.
static void LLVM_ATTRIBUTE_UNUSED
checkClobberSanity(MemoryAccess *Start, MemoryAccess *ClobberAt,
                   const MemoryLocation &StartLoc, const MemorySSA &MSSA,
                   const UpwardsMemoryQuery &Query, AliasAnalysis &AA) {
  assert(MSSA.dominates(ClobberAt, Start) && "Clobber doesn't dominate start?");

  if (MSSA.isLiveOnEntryDef(Start)) {
    assert(MSSA.isLiveOnEntryDef(ClobberAt) &&
           "liveOnEntry must clobber itself");
    return;
  }

  assert((isa<MemoryPhi>(Start) || Start != ClobberAt) &&
         "Start can't clobber itself!");

  bool FoundClobber = false;
  DenseSet<MemoryAccessPair> VisitedPhis;
  SmallVector<MemoryAccessPair, 8> Worklist;
  Worklist.emplace_back(Start, StartLoc);
  // Walk all paths from Start to ClobberAt, while looking for clobbers. If one
  // is found, complain.
  while (!Worklist.empty()) {
    MemoryAccessPair MAP = Worklist.pop_back_val();
    // All we care about is that nothing from Start to ClobberAt clobbers Start.
    // We learn nothing from revisiting nodes.
    if (!VisitedPhis.insert(MAP).second)
      continue;

    for (MemoryAccess *MA : def_chain(MAP.first)) {
      if (MA == ClobberAt) {
        if (auto *MD = dyn_cast<MemoryDef>(MA)) {
          // instructionClobbersQuery isn't essentially free, so don't use `|=`,
          // since it won't let us short-circuit.
          //
          // Also, note that this can't be hoisted out of the `Worklist` loop,
          // since MD may only act as a clobber for 1 of N MemoryLocations.
          FoundClobber = FoundClobber || MSSA.isLiveOnEntryDef(MD) ||
                         instructionClobbersQuery(MD, MAP.second, Query, AA);
        }
        break;
      }

      // We should never hit liveOnEntry, unless it's the clobber.
      assert(!MSSA.isLiveOnEntryDef(MA) && "Hit liveOnEntry before clobber?");

      if (auto *MD = dyn_cast<MemoryDef>(MA)) {
        (void)MD;
        assert(!instructionClobbersQuery(MD, MAP.second, Query, AA) &&
               "Found clobber before reaching ClobberAt!");
        continue;
      }

      assert(isa<MemoryPhi>(MA));
      Worklist.append(upward_defs_begin({MA, MAP.second}), upward_defs_end());
    }
  }

  // If ClobberAt is a MemoryPhi, we can assume something above it acted as a
  // clobber. Otherwise, `ClobberAt` should've acted as a clobber at some point.
  assert((isa<MemoryPhi>(ClobberAt) || FoundClobber) &&
         "ClobberAt never acted as a clobber");
}

/// Our algorithm for walking (and trying to optimize) clobbers, all wrapped up
/// in one class.
class ClobberWalker {
  /// Save a few bytes by using unsigned instead of size_t.
  using ListIndex = unsigned;

  /// Represents a span of contiguous MemoryDefs, potentially ending in a
  /// MemoryPhi.
  struct DefPath {
    MemoryLocation Loc;
    // Note that, because we always walk in reverse, Last will always dominate
    // First. Also note that First and Last are inclusive.
    MemoryAccess *First;
    MemoryAccess *Last;
    // N.B. Blocker is currently basically unused. The goal is to use it to make
    // cache invalidation better, but we're not there yet.
    MemoryAccess *Blocker;
    Optional<ListIndex> Previous;

    DefPath(const MemoryLocation &Loc, MemoryAccess *First, MemoryAccess *Last,
            Optional<ListIndex> Previous)
        : Loc(Loc), First(First), Last(Last), Previous(Previous) {}

    DefPath(const MemoryLocation &Loc, MemoryAccess *Init,
            Optional<ListIndex> Previous)
        : DefPath(Loc, Init, Init, Previous) {}
  };

  const MemorySSA &MSSA;
  AliasAnalysis &AA;
  DominatorTree &DT;
  WalkerCache &WC;
  UpwardsMemoryQuery *Query;
  bool UseCache;

  // Phi optimization bookkeeping
  SmallVector<DefPath, 32> Paths;
  DenseSet<ConstMemoryAccessPair> VisitedPhis;
  DenseMap<const BasicBlock *, MemoryAccess *> WalkTargetCache;

  void setUseCache(bool Use) { UseCache = Use; }
  bool shouldIgnoreCache() const {
    // UseCache will only be false when we're debugging, or when expensive
    // checks are enabled. In either case, we don't care deeply about speed.
    return LLVM_UNLIKELY(!UseCache);
  }

  void addCacheEntry(const MemoryAccess *What, MemoryAccess *To,
                     const MemoryLocation &Loc) const {
// EXPENSIVE_CHECKS because most of these queries are redundant.
#ifdef EXPENSIVE_CHECKS
    assert(MSSA.dominates(To, What));
#endif
    if (shouldIgnoreCache())
      return;
    WC.insert(What, To, Loc, Query->IsCall);
  }

  MemoryAccess *lookupCache(const MemoryAccess *MA, const MemoryLocation &Loc) {
    return shouldIgnoreCache() ? nullptr : WC.lookup(MA, Loc, Query->IsCall);
  }

  void cacheDefPath(const DefPath &DN, MemoryAccess *Target) const {
    if (shouldIgnoreCache())
      return;

    for (MemoryAccess *MA : def_chain(DN.First, DN.Last))
      addCacheEntry(MA, Target, DN.Loc);

    // DefPaths only express the path we walked. So, DN.Last could either be a
    // thing we want to cache, or not.
    if (DN.Last != Target)
      addCacheEntry(DN.Last, Target, DN.Loc);
  }

  /// Find the nearest def or phi that `From` can legally be optimized to.
  ///
  /// FIXME: Deduplicate this with MSSA::findDominatingDef. Ideally, MSSA should
  /// keep track of this information for us, and allow us O(1) lookups of this
  /// info.
  MemoryAccess *getWalkTarget(const MemoryPhi *From) {
    assert(!MSSA.isLiveOnEntryDef(From) && "liveOnEntry has no target.");
    assert(From->getNumOperands() && "Phi with no operands?");

    BasicBlock *BB = From->getBlock();
    auto At = WalkTargetCache.find(BB);
    if (At != WalkTargetCache.end())
      return At->second;

    SmallVector<const BasicBlock *, 8> ToCache;
    ToCache.push_back(BB);

    MemoryAccess *Result = MSSA.getLiveOnEntryDef();
    DomTreeNode *Node = DT.getNode(BB);
    while ((Node = Node->getIDom())) {
      auto At = WalkTargetCache.find(BB);
      if (At != WalkTargetCache.end()) {
        Result = At->second;
        break;
      }

      auto *Accesses = MSSA.getBlockAccesses(Node->getBlock());
      if (Accesses) {
        auto Iter = find_if(reverse(*Accesses), [](const MemoryAccess &MA) {
          return !isa<MemoryUse>(MA);
        });
        if (Iter != Accesses->rend()) {
          Result = const_cast<MemoryAccess *>(&*Iter);
          break;
        }
      }

      ToCache.push_back(Node->getBlock());
    }

    for (const BasicBlock *BB : ToCache)
      WalkTargetCache.insert({BB, Result});
    return Result;
  }

  /// Result of calling walkToPhiOrClobber.
  struct UpwardsWalkResult {
    /// The "Result" of the walk. Either a clobber, the last thing we walked, or
    /// both.
    MemoryAccess *Result;
    bool IsKnownClobber;
    bool FromCache;
  };

  /// Walk to the next Phi or Clobber in the def chain starting at Desc.Last.
  /// This will update Desc.Last as it walks. It will (optionally) also stop at
  /// StopAt.
  ///
  /// This does not test for whether StopAt is a clobber
  UpwardsWalkResult walkToPhiOrClobber(DefPath &Desc,
                                       MemoryAccess *StopAt = nullptr) {
    assert(!isa<MemoryUse>(Desc.Last) && "Uses don't exist in my world");

    for (MemoryAccess *Current : def_chain(Desc.Last)) {
      Desc.Last = Current;
      if (Current == StopAt)
        return {Current, false, false};

      if (auto *MD = dyn_cast<MemoryDef>(Current))
        if (MSSA.isLiveOnEntryDef(MD) ||
            instructionClobbersQuery(MD, Desc.Loc, *Query, AA))
          return {MD, true, false};

      // Cache checks must be done last, because if Current is a clobber, the
      // cache will contain the clobber for Current.
      if (MemoryAccess *MA = lookupCache(Current, Desc.Loc))
        return {MA, true, true};
    }

    assert(isa<MemoryPhi>(Desc.Last) &&
           "Ended at a non-clobber that's not a phi?");
    return {Desc.Last, false, false};
  }

  void addSearches(MemoryPhi *Phi, SmallVectorImpl<ListIndex> &PausedSearches,
                   ListIndex PriorNode) {
    auto UpwardDefs = make_range(upward_defs_begin({Phi, Paths[PriorNode].Loc}),
                                 upward_defs_end());
    for (const MemoryAccessPair &P : UpwardDefs) {
      PausedSearches.push_back(Paths.size());
      Paths.emplace_back(P.second, P.first, PriorNode);
    }
  }

  /// Represents a search that terminated after finding a clobber. This clobber
  /// may or may not be present in the path of defs from LastNode..SearchStart,
  /// since it may have been retrieved from cache.
  struct TerminatedPath {
    MemoryAccess *Clobber;
    ListIndex LastNode;
  };

  /// Get an access that keeps us from optimizing to the given phi.
  ///
  /// PausedSearches is an array of indices into the Paths array. Its incoming
  /// value is the indices of searches that stopped at the last phi optimization
  /// target. It's left in an unspecified state.
  ///
  /// If this returns None, NewPaused is a vector of searches that terminated
  /// at StopWhere. Otherwise, NewPaused is left in an unspecified state.
  Optional<ListIndex>
  getBlockingAccess(MemoryAccess *StopWhere,
                    SmallVectorImpl<ListIndex> &PausedSearches,
                    SmallVectorImpl<ListIndex> &NewPaused,
                    SmallVectorImpl<TerminatedPath> &Terminated) {
    assert(!PausedSearches.empty() && "No searches to continue?");

    // BFS vs DFS really doesn't make a difference here, so just do a DFS with
    // PausedSearches as our stack.
    while (!PausedSearches.empty()) {
      ListIndex PathIndex = PausedSearches.pop_back_val();
      DefPath &Node = Paths[PathIndex];

      // If we've already visited this path with this MemoryLocation, we don't
      // need to do so again.
      //
      // NOTE: That we just drop these paths on the ground makes caching
      // behavior sporadic. e.g. given a diamond:
      //  A
      // B C
      //  D
      //
      // ...If we walk D, B, A, C, we'll only cache the result of phi
      // optimization for A, B, and D; C will be skipped because it dies here.
      // This arguably isn't the worst thing ever, since:
      //   - We generally query things in a top-down order, so if we got below D
      //     without needing cache entries for {C, MemLoc}, then chances are
      //     that those cache entries would end up ultimately unused.
      //   - We still cache things for A, so C only needs to walk up a bit.
      // If this behavior becomes problematic, we can fix without a ton of extra
      // work.
      if (!VisitedPhis.insert({Node.Last, Node.Loc}).second)
        continue;

      UpwardsWalkResult Res = walkToPhiOrClobber(Node, /*StopAt=*/StopWhere);
      if (Res.IsKnownClobber) {
        assert(Res.Result != StopWhere || Res.FromCache);
        // If this wasn't a cache hit, we hit a clobber when walking. That's a
        // failure.
        if (!Res.FromCache || !MSSA.dominates(Res.Result, StopWhere))
          return PathIndex;

        // Otherwise, it's a valid thing to potentially optimize to.
        Terminated.push_back({Res.Result, PathIndex});
        continue;
      }

      if (Res.Result == StopWhere) {
        // We've hit our target. Save this path off for if we want to continue
        // walking.
        NewPaused.push_back(PathIndex);
        continue;
      }

      assert(!MSSA.isLiveOnEntryDef(Res.Result) && "liveOnEntry is a clobber");
      addSearches(cast<MemoryPhi>(Res.Result), PausedSearches, PathIndex);
    }

    return None;
  }

  template <typename T, typename Walker>
  struct generic_def_path_iterator
      : public iterator_facade_base<generic_def_path_iterator<T, Walker>,
                                    std::forward_iterator_tag, T *> {
    generic_def_path_iterator() : W(nullptr), N(None) {}
    generic_def_path_iterator(Walker *W, ListIndex N) : W(W), N(N) {}

    T &operator*() const { return curNode(); }

    generic_def_path_iterator &operator++() {
      N = curNode().Previous;
      return *this;
    }

    bool operator==(const generic_def_path_iterator &O) const {
      if (N.hasValue() != O.N.hasValue())
        return false;
      return !N.hasValue() || *N == *O.N;
    }

  private:
    T &curNode() const { return W->Paths[*N]; }

    Walker *W;
    Optional<ListIndex> N;
  };

  using def_path_iterator = generic_def_path_iterator<DefPath, ClobberWalker>;
  using const_def_path_iterator =
      generic_def_path_iterator<const DefPath, const ClobberWalker>;

  iterator_range<def_path_iterator> def_path(ListIndex From) {
    return make_range(def_path_iterator(this, From), def_path_iterator());
  }

  iterator_range<const_def_path_iterator> const_def_path(ListIndex From) const {
    return make_range(const_def_path_iterator(this, From),
                      const_def_path_iterator());
  }

  struct OptznResult {
    /// The path that contains our result.
    TerminatedPath PrimaryClobber;
    /// The paths that we can legally cache back from, but that aren't
    /// necessarily the result of the Phi optimization.
    SmallVector<TerminatedPath, 4> OtherClobbers;
  };

  ListIndex defPathIndex(const DefPath &N) const {
    // The assert looks nicer if we don't need to do &N
    const DefPath *NP = &N;
    assert(!Paths.empty() && NP >= &Paths.front() && NP <= &Paths.back() &&
           "Out of bounds DefPath!");
    return NP - &Paths.front();
  }

  /// Try to optimize a phi as best as we can. Returns a SmallVector of Paths
  /// that act as legal clobbers. Note that this won't return *all* clobbers.
  ///
  /// Phi optimization algorithm tl;dr:
  ///   - Find the earliest def/phi, A, we can optimize to
  ///   - Find if all paths from the starting memory access ultimately reach A
  ///     - If not, optimization isn't possible.
  ///     - Otherwise, walk from A to another clobber or phi, A'.
  ///       - If A' is a def, we're done.
  ///       - If A' is a phi, try to optimize it.
  ///
  /// A path is a series of {MemoryAccess, MemoryLocation} pairs. A path
  /// terminates when a MemoryAccess that clobbers said MemoryLocation is found.
  OptznResult tryOptimizePhi(MemoryPhi *Phi, MemoryAccess *Start,
                             const MemoryLocation &Loc) {
    assert(Paths.empty() && VisitedPhis.empty() &&
           "Reset the optimization state.");

    Paths.emplace_back(Loc, Start, Phi, None);
    // Stores how many "valid" optimization nodes we had prior to calling
    // addSearches/getBlockingAccess. Necessary for caching if we had a blocker.
    auto PriorPathsSize = Paths.size();

    SmallVector<ListIndex, 16> PausedSearches;
    SmallVector<ListIndex, 8> NewPaused;
    SmallVector<TerminatedPath, 4> TerminatedPaths;

    addSearches(Phi, PausedSearches, 0);

    // Moves the TerminatedPath with the "most dominated" Clobber to the end of
    // Paths.
    auto MoveDominatedPathToEnd = [&](SmallVectorImpl<TerminatedPath> &Paths) {
      assert(!Paths.empty() && "Need a path to move");
      auto Dom = Paths.begin();
      for (auto I = std::next(Dom), E = Paths.end(); I != E; ++I)
        if (!MSSA.dominates(I->Clobber, Dom->Clobber))
          Dom = I;
      auto Last = Paths.end() - 1;
      if (Last != Dom)
        std::iter_swap(Last, Dom);
    };

    MemoryPhi *Current = Phi;
    while (1) {
      assert(!MSSA.isLiveOnEntryDef(Current) &&
             "liveOnEntry wasn't treated as a clobber?");

      MemoryAccess *Target = getWalkTarget(Current);
      // If a TerminatedPath doesn't dominate Target, then it wasn't a legal
      // optimization for the prior phi.
      assert(all_of(TerminatedPaths, [&](const TerminatedPath &P) {
        return MSSA.dominates(P.Clobber, Target);
      }));

      // FIXME: This is broken, because the Blocker may be reported to be
      // liveOnEntry, and we'll happily wait for that to disappear (read: never)
      // For the moment, this is fine, since we do basically nothing with
      // blocker info.
      if (Optional<ListIndex> Blocker = getBlockingAccess(
              Target, PausedSearches, NewPaused, TerminatedPaths)) {
        MemoryAccess *BlockingAccess = Paths[*Blocker].Last;
        // Cache our work on the blocking node, since we know that's correct.
        cacheDefPath(Paths[*Blocker], BlockingAccess);

        // Find the node we started at. We can't search based on N->Last, since
        // we may have gone around a loop with a different MemoryLocation.
        auto Iter = find_if(def_path(*Blocker), [&](const DefPath &N) {
          return defPathIndex(N) < PriorPathsSize;
        });
        assert(Iter != def_path_iterator());

        DefPath &CurNode = *Iter;
        assert(CurNode.Last == Current);
        CurNode.Blocker = BlockingAccess;

        // Two things:
        // A. We can't reliably cache all of NewPaused back. Consider a case
        //    where we have two paths in NewPaused; one of which can't optimize
        //    above this phi, whereas the other can. If we cache the second path
        //    back, we'll end up with suboptimal cache entries. We can handle
        //    cases like this a bit better when we either try to find all
        //    clobbers that block phi optimization, or when our cache starts
        //    supporting unfinished searches.
        // B. We can't reliably cache TerminatedPaths back here without doing
        //    extra checks; consider a case like:
        //       T
        //      / \
        //     D   C
        //      \ /
        //       S
        //    Where T is our target, C is a node with a clobber on it, D is a
        //    diamond (with a clobber *only* on the left or right node, N), and
        //    S is our start. Say we walk to D, through the node opposite N
        //    (read: ignoring the clobber), and see a cache entry in the top
        //    node of D. That cache entry gets put into TerminatedPaths. We then
        //    walk up to C (N is later in our worklist), find the clobber, and
        //    quit. If we append TerminatedPaths to OtherClobbers, we'll cache
        //    the bottom part of D to the cached clobber, ignoring the clobber
        //    in N. Again, this problem goes away if we start tracking all
        //    blockers for a given phi optimization.
        TerminatedPath Result{CurNode.Last, defPathIndex(CurNode)};
        return {Result, {}};
      }

      // If there's nothing left to search, then all paths led to valid clobbers
      // that we got from our cache; pick the nearest to the start, and allow
      // the rest to be cached back.
      if (NewPaused.empty()) {
        MoveDominatedPathToEnd(TerminatedPaths);
        TerminatedPath Result = TerminatedPaths.pop_back_val();
        return {Result, std::move(TerminatedPaths)};
      }

      MemoryAccess *DefChainEnd = nullptr;
      SmallVector<TerminatedPath, 4> Clobbers;
      for (ListIndex Paused : NewPaused) {
        UpwardsWalkResult WR = walkToPhiOrClobber(Paths[Paused]);
        if (WR.IsKnownClobber)
          Clobbers.push_back({WR.Result, Paused});
        else
          // Micro-opt: If we hit the end of the chain, save it.
          DefChainEnd = WR.Result;
      }

      if (!TerminatedPaths.empty()) {
        // If we couldn't find the dominating phi/liveOnEntry in the above loop,
        // do it now.
        if (!DefChainEnd)
          for (MemoryAccess *MA : def_chain(Target))
            DefChainEnd = MA;

        // If any of the terminated paths don't dominate the phi we'll try to
        // optimize, we need to figure out what they are and quit.
        const BasicBlock *ChainBB = DefChainEnd->getBlock();
        for (const TerminatedPath &TP : TerminatedPaths) {
          // Because we know that DefChainEnd is as "high" as we can go, we
          // don't need local dominance checks; BB dominance is sufficient.
          if (DT.dominates(ChainBB, TP.Clobber->getBlock()))
            Clobbers.push_back(TP);
        }
      }

      // If we have clobbers in the def chain, find the one closest to Current
      // and quit.
      if (!Clobbers.empty()) {
        MoveDominatedPathToEnd(Clobbers);
        TerminatedPath Result = Clobbers.pop_back_val();
        return {Result, std::move(Clobbers)};
      }

      assert(all_of(NewPaused,
                    [&](ListIndex I) { return Paths[I].Last == DefChainEnd; }));

      // Because liveOnEntry is a clobber, this must be a phi.
      auto *DefChainPhi = cast<MemoryPhi>(DefChainEnd);

      PriorPathsSize = Paths.size();
      PausedSearches.clear();
      for (ListIndex I : NewPaused)
        addSearches(DefChainPhi, PausedSearches, I);
      NewPaused.clear();

      Current = DefChainPhi;
    }
  }

  /// Caches everything in an OptznResult.
  void cacheOptResult(const OptznResult &R) {
    if (R.OtherClobbers.empty()) {
      // If we're not going to be caching OtherClobbers, don't bother with
      // marking visited/etc.
      for (const DefPath &N : const_def_path(R.PrimaryClobber.LastNode))
        cacheDefPath(N, R.PrimaryClobber.Clobber);
      return;
    }

    // PrimaryClobber is our answer. If we can cache anything back, we need to
    // stop caching when we visit PrimaryClobber.
    SmallBitVector Visited(Paths.size());
    for (const DefPath &N : const_def_path(R.PrimaryClobber.LastNode)) {
      Visited[defPathIndex(N)] = true;
      cacheDefPath(N, R.PrimaryClobber.Clobber);
    }

    for (const TerminatedPath &P : R.OtherClobbers) {
      for (const DefPath &N : const_def_path(P.LastNode)) {
        ListIndex NIndex = defPathIndex(N);
        if (Visited[NIndex])
          break;
        Visited[NIndex] = true;
        cacheDefPath(N, P.Clobber);
      }
    }
  }

  void verifyOptResult(const OptznResult &R) const {
    assert(all_of(R.OtherClobbers, [&](const TerminatedPath &P) {
      return MSSA.dominates(P.Clobber, R.PrimaryClobber.Clobber);
    }));
  }

  void resetPhiOptznState() {
    Paths.clear();
    VisitedPhis.clear();
  }

public:
  ClobberWalker(const MemorySSA &MSSA, AliasAnalysis &AA, DominatorTree &DT,
                WalkerCache &WC)
      : MSSA(MSSA), AA(AA), DT(DT), WC(WC), UseCache(true) {}

  void reset() { WalkTargetCache.clear(); }

  /// Finds the nearest clobber for the given query, optimizing phis if
  /// possible.
  MemoryAccess *findClobber(MemoryAccess *Start, UpwardsMemoryQuery &Q,
                            bool UseWalkerCache = true) {
    setUseCache(UseWalkerCache);
    Query = &Q;

    MemoryAccess *Current = Start;
    // This walker pretends uses don't exist. If we're handed one, silently grab
    // its def. (This has the nice side-effect of ensuring we never cache uses)
    if (auto *MU = dyn_cast<MemoryUse>(Start))
      Current = MU->getDefiningAccess();

    DefPath FirstDesc(Q.StartingLoc, Current, Current, None);
    // Fast path for the overly-common case (no crazy phi optimization
    // necessary)
    UpwardsWalkResult WalkResult = walkToPhiOrClobber(FirstDesc);
    if (WalkResult.IsKnownClobber) {
      cacheDefPath(FirstDesc, WalkResult.Result);
      return WalkResult.Result;
    }

    OptznResult OptRes =
        tryOptimizePhi(cast<MemoryPhi>(FirstDesc.Last), Current, Q.StartingLoc);
    verifyOptResult(OptRes);
    cacheOptResult(OptRes);
    resetPhiOptznState();

#ifdef EXPENSIVE_CHECKS
    checkClobberSanity(Current, OptRes.PrimaryClobber.Clobber, Q.StartingLoc,
                       MSSA, Q, AA);
#endif
    return OptRes.PrimaryClobber.Clobber;
  }
};

struct RenamePassData {
  DomTreeNode *DTN;
  DomTreeNode::const_iterator ChildIt;
  MemoryAccess *IncomingVal;

  RenamePassData(DomTreeNode *D, DomTreeNode::const_iterator It,
                 MemoryAccess *M)
      : DTN(D), ChildIt(It), IncomingVal(M) {}
  void swap(RenamePassData &RHS) {
    std::swap(DTN, RHS.DTN);
    std::swap(ChildIt, RHS.ChildIt);
    std::swap(IncomingVal, RHS.IncomingVal);
  }
};
} // anonymous namespace

namespace llvm {
/// \brief A MemorySSAWalker that does AA walks and caching of lookups to
/// disambiguate accesses.
///
/// FIXME: The current implementation of this can take quadratic space in rare
/// cases. This can be fixed, but it is something to note until it is fixed.
///
/// In order to trigger this behavior, you need to store to N distinct locations
/// (that AA can prove don't alias), perform M stores to other memory
/// locations that AA can prove don't alias any of the initial N locations, and
/// then load from all of the N locations. In this case, we insert M cache
/// entries for each of the N loads.
///
/// For example:
/// define i32 @foo() {
///   %a = alloca i32, align 4
///   %b = alloca i32, align 4
///   store i32 0, i32* %a, align 4
///   store i32 0, i32* %b, align 4
///
///   ; Insert M stores to other memory that doesn't alias %a or %b here
///
///   %c = load i32, i32* %a, align 4 ; Caches M entries in
///                                   ; CachedUpwardsClobberingAccess for the
///                                   ; MemoryLocation %a
///   %d = load i32, i32* %b, align 4 ; Caches M entries in
///                                   ; CachedUpwardsClobberingAccess for the
///                                   ; MemoryLocation %b
///
///   ; For completeness' sake, loading %a or %b again would not cache *another*
///   ; M entries.
///   %r = add i32 %c, %d
///   ret i32 %r
/// }
class MemorySSA::CachingWalker final : public MemorySSAWalker {
  WalkerCache Cache;
  ClobberWalker Walker;
  bool AutoResetWalker;

  MemoryAccess *getClobberingMemoryAccess(MemoryAccess *, UpwardsMemoryQuery &);
  void verifyRemoved(MemoryAccess *);

public:
  CachingWalker(MemorySSA *, AliasAnalysis *, DominatorTree *);
  ~CachingWalker() override;

  using MemorySSAWalker::getClobberingMemoryAccess;
  MemoryAccess *getClobberingMemoryAccess(MemoryAccess *) override;
  MemoryAccess *getClobberingMemoryAccess(MemoryAccess *,
                                          MemoryLocation &) override;
  void invalidateInfo(MemoryAccess *) override;

  /// Whether we call resetClobberWalker() after each time we *actually* walk to
  /// answer a clobber query.
  void setAutoResetWalker(bool AutoReset) { AutoResetWalker = AutoReset; }

  /// Drop the walker's persistent data structures. At the moment, this means
  /// "drop the walker's cache of BasicBlocks ->
  /// earliest-MemoryAccess-we-can-optimize-to". This is necessary if we're
  /// going to have DT updates, if we remove MemoryAccesses, etc.
  void resetClobberWalker() { Walker.reset(); }
};

/// \brief Rename a single basic block into MemorySSA form.
/// Uses the standard SSA renaming algorithm.
/// \returns The new incoming value.
MemoryAccess *MemorySSA::renameBlock(BasicBlock *BB,
                                     MemoryAccess *IncomingVal) {
  auto It = PerBlockAccesses.find(BB);
  // Skip most processing if the list is empty.
  if (It != PerBlockAccesses.end()) {
    AccessList *Accesses = It->second.get();
    for (MemoryAccess &L : *Accesses) {
      switch (L.getValueID()) {
      case Value::MemoryUseVal:
        cast<MemoryUse>(&L)->setDefiningAccess(IncomingVal);
        break;
      case Value::MemoryDefVal:
        // We can't legally optimize defs, because we only allow single
        // memory phis/uses on operations, and if we optimize these, we can
        // end up with multiple reaching defs. Uses do not have this
        // problem, since they do not produce a value
        cast<MemoryDef>(&L)->setDefiningAccess(IncomingVal);
        IncomingVal = &L;
        break;
      case Value::MemoryPhiVal:
        IncomingVal = &L;
        break;
      }
    }
  }

  // Pass through values to our successors
  for (const BasicBlock *S : successors(BB)) {
    auto It = PerBlockAccesses.find(S);
    // Rename the phi nodes in our successor block
    if (It == PerBlockAccesses.end() || !isa<MemoryPhi>(It->second->front()))
      continue;
    AccessList *Accesses = It->second.get();
    auto *Phi = cast<MemoryPhi>(&Accesses->front());
    Phi->addIncoming(IncomingVal, BB);
  }

  return IncomingVal;
}

/// \brief This is the standard SSA renaming algorithm.
///
/// We walk the dominator tree in preorder, renaming accesses, and then filling
/// in phi nodes in our successors.
void MemorySSA::renamePass(DomTreeNode *Root, MemoryAccess *IncomingVal,
                           SmallPtrSet<BasicBlock *, 16> &Visited) {
  SmallVector<RenamePassData, 32> WorkStack;
  IncomingVal = renameBlock(Root->getBlock(), IncomingVal);
  WorkStack.push_back({Root, Root->begin(), IncomingVal});
  Visited.insert(Root->getBlock());

  while (!WorkStack.empty()) {
    DomTreeNode *Node = WorkStack.back().DTN;
    DomTreeNode::const_iterator ChildIt = WorkStack.back().ChildIt;
    IncomingVal = WorkStack.back().IncomingVal;

    if (ChildIt == Node->end()) {
      WorkStack.pop_back();
    } else {
      DomTreeNode *Child = *ChildIt;
      ++WorkStack.back().ChildIt;
      BasicBlock *BB = Child->getBlock();
      Visited.insert(BB);
      IncomingVal = renameBlock(BB, IncomingVal);
      WorkStack.push_back({Child, Child->begin(), IncomingVal});
    }
  }
}

/// \brief Compute dominator levels, used by the phi insertion algorithm above.
void MemorySSA::computeDomLevels(DenseMap<DomTreeNode *, unsigned> &DomLevels) {
  for (auto DFI = df_begin(DT->getRootNode()), DFE = df_end(DT->getRootNode());
       DFI != DFE; ++DFI)
    DomLevels[*DFI] = DFI.getPathLength() - 1;
}

/// \brief This handles unreachable block accesses by deleting phi nodes in
/// unreachable blocks, and marking all other unreachable MemoryAccess's as
/// being uses of the live on entry definition.
void MemorySSA::markUnreachableAsLiveOnEntry(BasicBlock *BB) {
  assert(!DT->isReachableFromEntry(BB) &&
         "Reachable block found while handling unreachable blocks");

  // Make sure phi nodes in our reachable successors end up with a
  // LiveOnEntryDef for our incoming edge, even though our block is forward
  // unreachable.  We could just disconnect these blocks from the CFG fully,
  // but we do not right now.
  for (const BasicBlock *S : successors(BB)) {
    if (!DT->isReachableFromEntry(S))
      continue;
    auto It = PerBlockAccesses.find(S);
    // Rename the phi nodes in our successor block
    if (It == PerBlockAccesses.end() || !isa<MemoryPhi>(It->second->front()))
      continue;
    AccessList *Accesses = It->second.get();
    auto *Phi = cast<MemoryPhi>(&Accesses->front());
    Phi->addIncoming(LiveOnEntryDef.get(), BB);
  }

  auto It = PerBlockAccesses.find(BB);
  if (It == PerBlockAccesses.end())
    return;

  auto &Accesses = It->second;
  for (auto AI = Accesses->begin(), AE = Accesses->end(); AI != AE;) {
    auto Next = std::next(AI);
    // If we have a phi, just remove it. We are going to replace all
    // users with live on entry.
    if (auto *UseOrDef = dyn_cast<MemoryUseOrDef>(AI))
      UseOrDef->setDefiningAccess(LiveOnEntryDef.get());
    else
      Accesses->erase(AI);
    AI = Next;
  }
}

MemorySSA::MemorySSA(Function &Func, AliasAnalysis *AA, DominatorTree *DT)
    : AA(AA), DT(DT), F(Func), LiveOnEntryDef(nullptr), Walker(nullptr),
      NextID(0) {
  buildMemorySSA();
}

MemorySSA::MemorySSA(MemorySSA &&MSSA)
    : AA(MSSA.AA), DT(MSSA.DT), F(MSSA.F),
      ValueToMemoryAccess(std::move(MSSA.ValueToMemoryAccess)),
      PerBlockAccesses(std::move(MSSA.PerBlockAccesses)),
      LiveOnEntryDef(std::move(MSSA.LiveOnEntryDef)),
      Walker(std::move(MSSA.Walker)), NextID(MSSA.NextID) {
  // Update the Walker MSSA pointer so it doesn't point to the moved-from MSSA
  // object any more.
  Walker->MSSA = this;
}

MemorySSA::~MemorySSA() {
  // Drop all our references
  for (const auto &Pair : PerBlockAccesses)
    for (MemoryAccess &MA : *Pair.second)
      MA.dropAllReferences();
}

MemorySSA::AccessList *MemorySSA::getOrCreateAccessList(const BasicBlock *BB) {
  auto Res = PerBlockAccesses.insert(std::make_pair(BB, nullptr));

  if (Res.second)
    Res.first->second = make_unique<AccessList>();
  return Res.first->second.get();
}

void MemorySSA::buildMemorySSA() {
  // We create an access to represent "live on entry", for things like
  // arguments or users of globals, where the memory they use is defined before
  // the beginning of the function. We do not actually insert it into the IR.
  // We do not define a live on exit for the immediate uses, and thus our
  // semantics do *not* imply that something with no immediate uses can simply
  // be removed.
  BasicBlock &StartingPoint = F.getEntryBlock();
  LiveOnEntryDef = make_unique<MemoryDef>(F.getContext(), nullptr, nullptr,
                                          &StartingPoint, NextID++);

  // We maintain lists of memory accesses per-block, trading memory for time. We
  // could just look up the memory access for every possible instruction in the
  // stream.
  SmallPtrSet<BasicBlock *, 32> DefiningBlocks;
  SmallPtrSet<BasicBlock *, 32> DefUseBlocks;
  // Go through each block, figure out where defs occur, and chain together all
  // the accesses.
  for (BasicBlock &B : F) {
    bool InsertIntoDef = false;
    AccessList *Accesses = nullptr;
    for (Instruction &I : B) {
      MemoryUseOrDef *MUD = createNewAccess(&I);
      if (!MUD)
        continue;
      InsertIntoDef |= isa<MemoryDef>(MUD);

      if (!Accesses)
        Accesses = getOrCreateAccessList(&B);
      Accesses->push_back(MUD);
    }
    if (InsertIntoDef)
      DefiningBlocks.insert(&B);
    if (Accesses)
      DefUseBlocks.insert(&B);
  }

  // Compute live-in.
  // Live in is normally defined as "all the blocks on the path from each def to
  // each of it's uses".
  // MemoryDef's are implicit uses of previous state, so they are also uses.
  // This means we don't really have def-only instructions.  The only
  // MemoryDef's that are not really uses are those that are of the LiveOnEntry
  // variable (because LiveOnEntry can reach anywhere, and every def is a
  // must-kill of LiveOnEntry).
  // In theory, you could precisely compute live-in by using alias-analysis to
  // disambiguate defs and uses to see which really pair up with which.
  // In practice, this would be really expensive and difficult. So we simply
  // assume all defs are also uses that need to be kept live.
  // Because of this, the end result of this live-in computation will be "the
  // entire set of basic blocks that reach any use".

  SmallPtrSet<BasicBlock *, 32> LiveInBlocks;
  SmallVector<BasicBlock *, 64> LiveInBlockWorklist(DefUseBlocks.begin(),
                                                    DefUseBlocks.end());
  // Now that we have a set of blocks where a value is live-in, recursively add
  // predecessors until we find the full region the value is live.
  while (!LiveInBlockWorklist.empty()) {
    BasicBlock *BB = LiveInBlockWorklist.pop_back_val();

    // The block really is live in here, insert it into the set.  If already in
    // the set, then it has already been processed.
    if (!LiveInBlocks.insert(BB).second)
      continue;

    // Since the value is live into BB, it is either defined in a predecessor or
    // live into it to.
    LiveInBlockWorklist.append(pred_begin(BB), pred_end(BB));
  }

  // Determine where our MemoryPhi's should go
  ForwardIDFCalculator IDFs(*DT);
  IDFs.setDefiningBlocks(DefiningBlocks);
  IDFs.setLiveInBlocks(LiveInBlocks);
  SmallVector<BasicBlock *, 32> IDFBlocks;
  IDFs.calculate(IDFBlocks);

  // Now place MemoryPhi nodes.
  for (auto &BB : IDFBlocks) {
    // Insert phi node
    AccessList *Accesses = getOrCreateAccessList(BB);
    MemoryPhi *Phi = new MemoryPhi(BB->getContext(), BB, NextID++);
    ValueToMemoryAccess.insert(std::make_pair(BB, Phi));
    // Phi's always are placed at the front of the block.
    Accesses->push_front(Phi);
  }

  // Now do regular SSA renaming on the MemoryDef/MemoryUse. Visited will get
  // filled in with all blocks.
  SmallPtrSet<BasicBlock *, 16> Visited;
  renamePass(DT->getRootNode(), LiveOnEntryDef.get(), Visited);

  CachingWalker *Walker = getWalkerImpl();

  // We're doing a batch of updates; don't drop useful caches between them.
  Walker->setAutoResetWalker(false);

  // Now optimize the MemoryUse's defining access to point to the nearest
  // dominating clobbering def.
  // This ensures that MemoryUse's that are killed by the same store are
  // immediate users of that store, one of the invariants we guarantee.
  for (auto DomNode : depth_first(DT)) {
    BasicBlock *BB = DomNode->getBlock();
    auto AI = PerBlockAccesses.find(BB);
    if (AI == PerBlockAccesses.end())
      continue;
    AccessList *Accesses = AI->second.get();
    for (auto &MA : *Accesses) {
      if (auto *MU = dyn_cast<MemoryUse>(&MA)) {
        Instruction *Inst = MU->getMemoryInst();
        MU->setDefiningAccess(Walker->getClobberingMemoryAccess(Inst));
      }
    }
  }

  Walker->setAutoResetWalker(true);
  Walker->resetClobberWalker();

  // Mark the uses in unreachable blocks as live on entry, so that they go
  // somewhere.
  for (auto &BB : F)
    if (!Visited.count(&BB))
      markUnreachableAsLiveOnEntry(&BB);
}

MemorySSAWalker *MemorySSA::getWalker() { return getWalkerImpl(); }

MemorySSA::CachingWalker *MemorySSA::getWalkerImpl() {
  if (Walker)
    return Walker.get();

  Walker = make_unique<CachingWalker>(this, AA, DT);
  return Walker.get();
}

MemoryPhi *MemorySSA::createMemoryPhi(BasicBlock *BB) {
  assert(!getMemoryAccess(BB) && "MemoryPhi already exists for this BB");
  AccessList *Accesses = getOrCreateAccessList(BB);
  MemoryPhi *Phi = new MemoryPhi(BB->getContext(), BB, NextID++);
  ValueToMemoryAccess.insert(std::make_pair(BB, Phi));
  // Phi's always are placed at the front of the block.
  Accesses->push_front(Phi);
  BlockNumberingValid.erase(BB);
  return Phi;
}

MemoryUseOrDef *MemorySSA::createDefinedAccess(Instruction *I,
                                               MemoryAccess *Definition) {
  assert(!isa<PHINode>(I) && "Cannot create a defined access for a PHI");
  MemoryUseOrDef *NewAccess = createNewAccess(I);
  assert(
      NewAccess != nullptr &&
      "Tried to create a memory access for a non-memory touching instruction");
  NewAccess->setDefiningAccess(Definition);
  return NewAccess;
}

MemoryAccess *MemorySSA::createMemoryAccessInBB(Instruction *I,
                                                MemoryAccess *Definition,
                                                const BasicBlock *BB,
                                                InsertionPlace Point) {
  MemoryUseOrDef *NewAccess = createDefinedAccess(I, Definition);
  auto *Accesses = getOrCreateAccessList(BB);
  if (Point == Beginning) {
    // It goes after any phi nodes
    auto AI = std::find_if(
        Accesses->begin(), Accesses->end(),
        [](const MemoryAccess &MA) { return !isa<MemoryPhi>(MA); });

    Accesses->insert(AI, NewAccess);
  } else {
    Accesses->push_back(NewAccess);
  }
  BlockNumberingValid.erase(BB);
  return NewAccess;
}
MemoryAccess *MemorySSA::createMemoryAccessBefore(Instruction *I,
                                                  MemoryAccess *Definition,
                                                  MemoryAccess *InsertPt) {
  assert(I->getParent() == InsertPt->getBlock() &&
         "New and old access must be in the same block");
  MemoryUseOrDef *NewAccess = createDefinedAccess(I, Definition);
  auto *Accesses = getOrCreateAccessList(InsertPt->getBlock());
  Accesses->insert(AccessList::iterator(InsertPt), NewAccess);
  BlockNumberingValid.erase(InsertPt->getBlock());
  return NewAccess;
}

MemoryAccess *MemorySSA::createMemoryAccessAfter(Instruction *I,
                                                 MemoryAccess *Definition,
                                                 MemoryAccess *InsertPt) {
  assert(I->getParent() == InsertPt->getBlock() &&
         "New and old access must be in the same block");
  MemoryUseOrDef *NewAccess = createDefinedAccess(I, Definition);
  auto *Accesses = getOrCreateAccessList(InsertPt->getBlock());
  Accesses->insertAfter(AccessList::iterator(InsertPt), NewAccess);
  BlockNumberingValid.erase(InsertPt->getBlock());
  return NewAccess;
}

/// \brief Helper function to create new memory accesses
MemoryUseOrDef *MemorySSA::createNewAccess(Instruction *I) {
  // The assume intrinsic has a control dependency which we model by claiming
  // that it writes arbitrarily. Ignore that fake memory dependency here.
  // FIXME: Replace this special casing with a more accurate modelling of
  // assume's control dependency.
  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I))
    if (II->getIntrinsicID() == Intrinsic::assume)
      return nullptr;

  // Find out what affect this instruction has on memory.
  ModRefInfo ModRef = AA->getModRefInfo(I);
  bool Def = bool(ModRef & MRI_Mod);
  bool Use = bool(ModRef & MRI_Ref);

  // It's possible for an instruction to not modify memory at all. During
  // construction, we ignore them.
  if (!Def && !Use)
    return nullptr;

  assert((Def || Use) &&
         "Trying to create a memory access with a non-memory instruction");

  MemoryUseOrDef *MUD;
  if (Def)
    MUD = new MemoryDef(I->getContext(), nullptr, I, I->getParent(), NextID++);
  else
    MUD = new MemoryUse(I->getContext(), nullptr, I, I->getParent());
  ValueToMemoryAccess.insert(std::make_pair(I, MUD));
  return MUD;
}

MemoryAccess *MemorySSA::findDominatingDef(BasicBlock *UseBlock,
                                           enum InsertionPlace Where) {
  // Handle the initial case
  if (Where == Beginning)
    // The only thing that could define us at the beginning is a phi node
    if (MemoryPhi *Phi = getMemoryAccess(UseBlock))
      return Phi;

  DomTreeNode *CurrNode = DT->getNode(UseBlock);
  // Need to be defined by our dominator
  if (Where == Beginning)
    CurrNode = CurrNode->getIDom();
  Where = End;
  while (CurrNode) {
    auto It = PerBlockAccesses.find(CurrNode->getBlock());
    if (It != PerBlockAccesses.end()) {
      auto &Accesses = It->second;
      for (MemoryAccess &RA : reverse(*Accesses)) {
        if (isa<MemoryDef>(RA) || isa<MemoryPhi>(RA))
          return &RA;
      }
    }
    CurrNode = CurrNode->getIDom();
  }
  return LiveOnEntryDef.get();
}

/// \brief Returns true if \p Replacer dominates \p Replacee .
bool MemorySSA::dominatesUse(const MemoryAccess *Replacer,
                             const MemoryAccess *Replacee) const {
  if (isa<MemoryUseOrDef>(Replacee))
    return DT->dominates(Replacer->getBlock(), Replacee->getBlock());
  const auto *MP = cast<MemoryPhi>(Replacee);
  // For a phi node, the use occurs in the predecessor block of the phi node.
  // Since we may occur multiple times in the phi node, we have to check each
  // operand to ensure Replacer dominates each operand where Replacee occurs.
  for (const Use &Arg : MP->operands()) {
    if (Arg.get() != Replacee &&
        !DT->dominates(Replacer->getBlock(), MP->getIncomingBlock(Arg)))
      return false;
  }
  return true;
}

/// \brief If all arguments of a MemoryPHI are defined by the same incoming
/// argument, return that argument.
static MemoryAccess *onlySingleValue(MemoryPhi *MP) {
  MemoryAccess *MA = nullptr;

  for (auto &Arg : MP->operands()) {
    if (!MA)
      MA = cast<MemoryAccess>(Arg);
    else if (MA != Arg)
      return nullptr;
  }
  return MA;
}

/// \brief Properly remove \p MA from all of MemorySSA's lookup tables.
///
/// Because of the way the intrusive list and use lists work, it is important to
/// do removal in the right order.
void MemorySSA::removeFromLookups(MemoryAccess *MA) {
  assert(MA->use_empty() &&
         "Trying to remove memory access that still has uses");
  BlockNumbering.erase(MA);
  if (MemoryUseOrDef *MUD = dyn_cast<MemoryUseOrDef>(MA))
    MUD->setDefiningAccess(nullptr);
  // Invalidate our walker's cache if necessary
  if (!isa<MemoryUse>(MA))
    Walker->invalidateInfo(MA);
  // The call below to erase will destroy MA, so we can't change the order we
  // are doing things here
  Value *MemoryInst;
  if (MemoryUseOrDef *MUD = dyn_cast<MemoryUseOrDef>(MA)) {
    MemoryInst = MUD->getMemoryInst();
  } else {
    MemoryInst = MA->getBlock();
  }
  ValueToMemoryAccess.erase(MemoryInst);

  auto AccessIt = PerBlockAccesses.find(MA->getBlock());
  std::unique_ptr<AccessList> &Accesses = AccessIt->second;
  Accesses->erase(MA);
  if (Accesses->empty())
    PerBlockAccesses.erase(AccessIt);
}

void MemorySSA::removeMemoryAccess(MemoryAccess *MA) {
  assert(!isLiveOnEntryDef(MA) && "Trying to remove the live on entry def");
  // We can only delete phi nodes if they have no uses, or we can replace all
  // uses with a single definition.
  MemoryAccess *NewDefTarget = nullptr;
  if (MemoryPhi *MP = dyn_cast<MemoryPhi>(MA)) {
    // Note that it is sufficient to know that all edges of the phi node have
    // the same argument.  If they do, by the definition of dominance frontiers
    // (which we used to place this phi), that argument must dominate this phi,
    // and thus, must dominate the phi's uses, and so we will not hit the assert
    // below.
    NewDefTarget = onlySingleValue(MP);
    assert((NewDefTarget || MP->use_empty()) &&
           "We can't delete this memory phi");
  } else {
    NewDefTarget = cast<MemoryUseOrDef>(MA)->getDefiningAccess();
  }

  // Re-point the uses at our defining access
  if (!MA->use_empty())
    MA->replaceAllUsesWith(NewDefTarget);

  // The call below to erase will destroy MA, so we can't change the order we
  // are doing things here
  removeFromLookups(MA);
}

void MemorySSA::print(raw_ostream &OS) const {
  MemorySSAAnnotatedWriter Writer(this);
  F.print(OS, &Writer);
}

void MemorySSA::dump() const {
  MemorySSAAnnotatedWriter Writer(this);
  F.print(dbgs(), &Writer);
}

void MemorySSA::verifyMemorySSA() const {
  verifyDefUses(F);
  verifyDomination(F);
  verifyOrdering(F);
}

/// \brief Verify that the order and existence of MemoryAccesses matches the
/// order and existence of memory affecting instructions.
void MemorySSA::verifyOrdering(Function &F) const {
  // Walk all the blocks, comparing what the lookups think and what the access
  // lists think, as well as the order in the blocks vs the order in the access
  // lists.
  SmallVector<MemoryAccess *, 32> ActualAccesses;
  for (BasicBlock &B : F) {
    const AccessList *AL = getBlockAccesses(&B);
    MemoryAccess *Phi = getMemoryAccess(&B);
    if (Phi)
      ActualAccesses.push_back(Phi);
    for (Instruction &I : B) {
      MemoryAccess *MA = getMemoryAccess(&I);
      assert((!MA || AL) && "We have memory affecting instructions "
                            "in this block but they are not in the "
                            "access list");
      if (MA)
        ActualAccesses.push_back(MA);
    }
    // Either we hit the assert, really have no accesses, or we have both
    // accesses and an access list
    if (!AL)
      continue;
    assert(AL->size() == ActualAccesses.size() &&
           "We don't have the same number of accesses in the block as on the "
           "access list");
    auto ALI = AL->begin();
    auto AAI = ActualAccesses.begin();
    while (ALI != AL->end() && AAI != ActualAccesses.end()) {
      assert(&*ALI == *AAI && "Not the same accesses in the same order");
      ++ALI;
      ++AAI;
    }
    ActualAccesses.clear();
  }
}

/// \brief Verify the domination properties of MemorySSA by checking that each
/// definition dominates all of its uses.
void MemorySSA::verifyDomination(Function &F) const {
  for (BasicBlock &B : F) {
    // Phi nodes are attached to basic blocks
    if (MemoryPhi *MP = getMemoryAccess(&B)) {
      for (User *U : MP->users()) {
        BasicBlock *UseBlock;
        // Phi operands are used on edges, we simulate the right domination by
        // acting as if the use occurred at the end of the predecessor block.
        if (MemoryPhi *P = dyn_cast<MemoryPhi>(U)) {
          for (const auto &Arg : P->operands()) {
            if (Arg == MP) {
              UseBlock = P->getIncomingBlock(Arg);
              break;
            }
          }
        } else {
          UseBlock = cast<MemoryAccess>(U)->getBlock();
        }
        (void)UseBlock;
        assert(DT->dominates(MP->getBlock(), UseBlock) &&
               "Memory PHI does not dominate it's uses");
      }
    }

    for (Instruction &I : B) {
      MemoryAccess *MD = dyn_cast_or_null<MemoryDef>(getMemoryAccess(&I));
      if (!MD)
        continue;

      for (User *U : MD->users()) {
        BasicBlock *UseBlock;
        (void)UseBlock;
        // Things are allowed to flow to phi nodes over their predecessor edge.
        if (auto *P = dyn_cast<MemoryPhi>(U)) {
          for (const auto &Arg : P->operands()) {
            if (Arg == MD) {
              UseBlock = P->getIncomingBlock(Arg);
              break;
            }
          }
        } else {
          UseBlock = cast<MemoryAccess>(U)->getBlock();
        }
        assert(DT->dominates(MD->getBlock(), UseBlock) &&
               "Memory Def does not dominate it's uses");
      }
    }
  }
}

/// \brief Verify the def-use lists in MemorySSA, by verifying that \p Use
/// appears in the use list of \p Def.
///
/// llvm_unreachable is used instead of asserts because this may be called in
/// a build without asserts. In that case, we don't want this to turn into a
/// nop.
void MemorySSA::verifyUseInDefs(MemoryAccess *Def, MemoryAccess *Use) const {
  // The live on entry use may cause us to get a NULL def here
  if (!Def) {
    if (!isLiveOnEntryDef(Use))
      llvm_unreachable("Null def but use not point to live on entry def");
  } else if (std::find(Def->user_begin(), Def->user_end(), Use) ==
             Def->user_end()) {
    llvm_unreachable("Did not find use in def's use list");
  }
}

/// \brief Verify the immediate use information, by walking all the memory
/// accesses and verifying that, for each use, it appears in the
/// appropriate def's use list
void MemorySSA::verifyDefUses(Function &F) const {
  for (BasicBlock &B : F) {
    // Phi nodes are attached to basic blocks
    if (MemoryPhi *Phi = getMemoryAccess(&B)) {
      assert(Phi->getNumOperands() == static_cast<unsigned>(std::distance(
                                          pred_begin(&B), pred_end(&B))) &&
             "Incomplete MemoryPhi Node");
      for (unsigned I = 0, E = Phi->getNumIncomingValues(); I != E; ++I)
        verifyUseInDefs(Phi->getIncomingValue(I), Phi);
    }

    for (Instruction &I : B) {
      if (MemoryAccess *MA = getMemoryAccess(&I)) {
        assert(isa<MemoryUseOrDef>(MA) &&
               "Found a phi node not attached to a bb");
        verifyUseInDefs(cast<MemoryUseOrDef>(MA)->getDefiningAccess(), MA);
      }
    }
  }
}

MemoryAccess *MemorySSA::getMemoryAccess(const Value *I) const {
  return ValueToMemoryAccess.lookup(I);
}

MemoryPhi *MemorySSA::getMemoryAccess(const BasicBlock *BB) const {
  return cast_or_null<MemoryPhi>(getMemoryAccess((const Value *)BB));
}

/// Perform a local numbering on blocks so that instruction ordering can be
/// determined in constant time.
/// TODO: We currently just number in order.  If we numbered by N, we could
/// allow at least N-1 sequences of insertBefore or insertAfter (and at least
/// log2(N) sequences of mixed before and after) without needing to invalidate
/// the numbering.
void MemorySSA::renumberBlock(const BasicBlock *B) const {
  // The pre-increment ensures the numbers really start at 1.
  unsigned long CurrentNumber = 0;
  const AccessList *AL = getBlockAccesses(B);
  assert(AL != nullptr && "Asking to renumber an empty block");
  for (const auto &I : *AL)
    BlockNumbering[&I] = ++CurrentNumber;
  BlockNumberingValid.insert(B);
}

/// \brief Determine, for two memory accesses in the same block,
/// whether \p Dominator dominates \p Dominatee.
/// \returns True if \p Dominator dominates \p Dominatee.
bool MemorySSA::locallyDominates(const MemoryAccess *Dominator,
                                 const MemoryAccess *Dominatee) const {

  const BasicBlock *DominatorBlock = Dominator->getBlock();

  assert((DominatorBlock == Dominatee->getBlock()) &&
         "Asking for local domination when accesses are in different blocks!");
  // A node dominates itself.
  if (Dominatee == Dominator)
    return true;

  // When Dominatee is defined on function entry, it is not dominated by another
  // memory access.
  if (isLiveOnEntryDef(Dominatee))
    return false;

  // When Dominator is defined on function entry, it dominates the other memory
  // access.
  if (isLiveOnEntryDef(Dominator))
    return true;

  if (!BlockNumberingValid.count(DominatorBlock))
    renumberBlock(DominatorBlock);

  unsigned long DominatorNum = BlockNumbering.lookup(Dominator);
  // All numbers start with 1
  assert(DominatorNum != 0 && "Block was not numbered properly");
  unsigned long DominateeNum = BlockNumbering.lookup(Dominatee);
  assert(DominateeNum != 0 && "Block was not numbered properly");
  return DominatorNum < DominateeNum;
}

bool MemorySSA::dominates(const MemoryAccess *Dominator,
                          const MemoryAccess *Dominatee) const {
  if (Dominator == Dominatee)
    return true;

  if (isLiveOnEntryDef(Dominatee))
    return false;

  if (Dominator->getBlock() != Dominatee->getBlock())
    return DT->dominates(Dominator->getBlock(), Dominatee->getBlock());
  return locallyDominates(Dominator, Dominatee);
}

const static char LiveOnEntryStr[] = "liveOnEntry";

void MemoryDef::print(raw_ostream &OS) const {
  MemoryAccess *UO = getDefiningAccess();

  OS << getID() << " = MemoryDef(";
  if (UO && UO->getID())
    OS << UO->getID();
  else
    OS << LiveOnEntryStr;
  OS << ')';
}

void MemoryPhi::print(raw_ostream &OS) const {
  bool First = true;
  OS << getID() << " = MemoryPhi(";
  for (const auto &Op : operands()) {
    BasicBlock *BB = getIncomingBlock(Op);
    MemoryAccess *MA = cast<MemoryAccess>(Op);
    if (!First)
      OS << ',';
    else
      First = false;

    OS << '{';
    if (BB->hasName())
      OS << BB->getName();
    else
      BB->printAsOperand(OS, false);
    OS << ',';
    if (unsigned ID = MA->getID())
      OS << ID;
    else
      OS << LiveOnEntryStr;
    OS << '}';
  }
  OS << ')';
}

MemoryAccess::~MemoryAccess() {}

void MemoryUse::print(raw_ostream &OS) const {
  MemoryAccess *UO = getDefiningAccess();
  OS << "MemoryUse(";
  if (UO && UO->getID())
    OS << UO->getID();
  else
    OS << LiveOnEntryStr;
  OS << ')';
}

void MemoryAccess::dump() const {
  print(dbgs());
  dbgs() << "\n";
}

char MemorySSAPrinterLegacyPass::ID = 0;

MemorySSAPrinterLegacyPass::MemorySSAPrinterLegacyPass() : FunctionPass(ID) {
  initializeMemorySSAPrinterLegacyPassPass(*PassRegistry::getPassRegistry());
}

void MemorySSAPrinterLegacyPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<MemorySSAWrapperPass>();
  AU.addPreserved<MemorySSAWrapperPass>();
}

bool MemorySSAPrinterLegacyPass::runOnFunction(Function &F) {
  auto &MSSA = getAnalysis<MemorySSAWrapperPass>().getMSSA();
  MSSA.print(dbgs());
  if (VerifyMemorySSA)
    MSSA.verifyMemorySSA();
  return false;
}

char MemorySSAAnalysis::PassID;

MemorySSA MemorySSAAnalysis::run(Function &F, AnalysisManager<Function> &AM) {
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &AA = AM.getResult<AAManager>(F);
  return MemorySSA(F, &AA, &DT);
}

PreservedAnalyses MemorySSAPrinterPass::run(Function &F,
                                            FunctionAnalysisManager &AM) {
  OS << "MemorySSA for function: " << F.getName() << "\n";
  AM.getResult<MemorySSAAnalysis>(F).print(OS);

  return PreservedAnalyses::all();
}

PreservedAnalyses MemorySSAVerifierPass::run(Function &F,
                                             FunctionAnalysisManager &AM) {
  AM.getResult<MemorySSAAnalysis>(F).verifyMemorySSA();

  return PreservedAnalyses::all();
}

char MemorySSAWrapperPass::ID = 0;

MemorySSAWrapperPass::MemorySSAWrapperPass() : FunctionPass(ID) {
  initializeMemorySSAWrapperPassPass(*PassRegistry::getPassRegistry());
}

void MemorySSAWrapperPass::releaseMemory() { MSSA.reset(); }

void MemorySSAWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequiredTransitive<DominatorTreeWrapperPass>();
  AU.addRequiredTransitive<AAResultsWrapperPass>();
}

bool MemorySSAWrapperPass::runOnFunction(Function &F) {
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
  MSSA.reset(new MemorySSA(F, &AA, &DT));
  return false;
}

void MemorySSAWrapperPass::verifyAnalysis() const { MSSA->verifyMemorySSA(); }

void MemorySSAWrapperPass::print(raw_ostream &OS, const Module *M) const {
  MSSA->print(OS);
}

MemorySSAWalker::MemorySSAWalker(MemorySSA *M) : MSSA(M) {}

MemorySSA::CachingWalker::CachingWalker(MemorySSA *M, AliasAnalysis *A,
                                        DominatorTree *D)
    : MemorySSAWalker(M), Walker(*M, *A, *D, Cache), AutoResetWalker(true) {}

MemorySSA::CachingWalker::~CachingWalker() {}

void MemorySSA::CachingWalker::invalidateInfo(MemoryAccess *MA) {
  // TODO: We can do much better cache invalidation with differently stored
  // caches.  For now, for MemoryUses, we simply remove them
  // from the cache, and kill the entire call/non-call cache for everything
  // else.  The problem is for phis or defs, currently we'd need to follow use
  // chains down and invalidate anything below us in the chain that currently
  // terminates at this access.

  // See if this is a MemoryUse, if so, just remove the cached info. MemoryUse
  // is by definition never a barrier, so nothing in the cache could point to
  // this use. In that case, we only need invalidate the info for the use
  // itself.

  if (MemoryUse *MU = dyn_cast<MemoryUse>(MA)) {
    UpwardsMemoryQuery Q(MU->getMemoryInst(), MU);
    Cache.remove(MU, Q.StartingLoc, Q.IsCall);
  } else {
    // If it is not a use, the best we can do right now is destroy the cache.
    Cache.clear();
  }

#ifdef EXPENSIVE_CHECKS
  verifyRemoved(MA);
#endif
}

/// \brief Walk the use-def chains starting at \p MA and find
/// the MemoryAccess that actually clobbers Loc.
///
/// \returns our clobbering memory access
MemoryAccess *MemorySSA::CachingWalker::getClobberingMemoryAccess(
    MemoryAccess *StartingAccess, UpwardsMemoryQuery &Q) {
  MemoryAccess *New = Walker.findClobber(StartingAccess, Q);
#ifdef EXPENSIVE_CHECKS
  MemoryAccess *NewNoCache =
      Walker.findClobber(StartingAccess, Q, /*UseWalkerCache=*/false);
  assert(NewNoCache == New && "Cache made us hand back a different result?");
#endif
  if (AutoResetWalker)
    resetClobberWalker();
  return New;
}

MemoryAccess *MemorySSA::CachingWalker::getClobberingMemoryAccess(
    MemoryAccess *StartingAccess, MemoryLocation &Loc) {
  if (isa<MemoryPhi>(StartingAccess))
    return StartingAccess;

  auto *StartingUseOrDef = cast<MemoryUseOrDef>(StartingAccess);
  if (MSSA->isLiveOnEntryDef(StartingUseOrDef))
    return StartingUseOrDef;

  Instruction *I = StartingUseOrDef->getMemoryInst();

  // Conservatively, fences are always clobbers, so don't perform the walk if we
  // hit a fence.
  if (!ImmutableCallSite(I) && I->isFenceLike())
    return StartingUseOrDef;

  UpwardsMemoryQuery Q;
  Q.OriginalAccess = StartingUseOrDef;
  Q.StartingLoc = Loc;
  Q.Inst = I;
  Q.IsCall = false;

  if (auto *CacheResult = Cache.lookup(StartingUseOrDef, Loc, Q.IsCall))
    return CacheResult;

  // Unlike the other function, do not walk to the def of a def, because we are
  // handed something we already believe is the clobbering access.
  MemoryAccess *DefiningAccess = isa<MemoryUse>(StartingUseOrDef)
                                     ? StartingUseOrDef->getDefiningAccess()
                                     : StartingUseOrDef;

  MemoryAccess *Clobber = getClobberingMemoryAccess(DefiningAccess, Q);
  DEBUG(dbgs() << "Starting Memory SSA clobber for " << *I << " is ");
  DEBUG(dbgs() << *StartingUseOrDef << "\n");
  DEBUG(dbgs() << "Final Memory SSA clobber for " << *I << " is ");
  DEBUG(dbgs() << *Clobber << "\n");
  return Clobber;
}

MemoryAccess *
MemorySSA::CachingWalker::getClobberingMemoryAccess(MemoryAccess *MA) {
  auto *StartingAccess = dyn_cast<MemoryUseOrDef>(MA);
  // If this is a MemoryPhi, we can't do anything.
  if (!StartingAccess)
    return MA;

  const Instruction *I = StartingAccess->getMemoryInst();
  UpwardsMemoryQuery Q(I, StartingAccess);
  // We can't sanely do anything with a fences, they conservatively
  // clobber all memory, and have no locations to get pointers from to
  // try to disambiguate.
  if (!Q.IsCall && I->isFenceLike())
    return StartingAccess;

  if (auto *CacheResult = Cache.lookup(StartingAccess, Q.StartingLoc, Q.IsCall))
    return CacheResult;

  // Start with the thing we already think clobbers this location
  MemoryAccess *DefiningAccess = StartingAccess->getDefiningAccess();

  // At this point, DefiningAccess may be the live on entry def.
  // If it is, we will not get a better result.
  if (MSSA->isLiveOnEntryDef(DefiningAccess))
    return DefiningAccess;

  MemoryAccess *Result = getClobberingMemoryAccess(DefiningAccess, Q);
  DEBUG(dbgs() << "Starting Memory SSA clobber for " << *I << " is ");
  DEBUG(dbgs() << *DefiningAccess << "\n");
  DEBUG(dbgs() << "Final Memory SSA clobber for " << *I << " is ");
  DEBUG(dbgs() << *Result << "\n");

  return Result;
}

// Verify that MA doesn't exist in any of the caches.
void MemorySSA::CachingWalker::verifyRemoved(MemoryAccess *MA) {
  assert(!Cache.contains(MA) && "Found removed MemoryAccess in cache.");
}

MemoryAccess *
DoNothingMemorySSAWalker::getClobberingMemoryAccess(MemoryAccess *MA) {
  if (auto *Use = dyn_cast<MemoryUseOrDef>(MA))
    return Use->getDefiningAccess();
  return MA;
}

MemoryAccess *DoNothingMemorySSAWalker::getClobberingMemoryAccess(
    MemoryAccess *StartingAccess, MemoryLocation &) {
  if (auto *Use = dyn_cast<MemoryUseOrDef>(StartingAccess))
    return Use->getDefiningAccess();
  return StartingAccess;
}
} // namespace llvm
