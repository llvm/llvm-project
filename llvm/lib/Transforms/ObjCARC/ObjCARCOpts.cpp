//===- ObjCARCOpts.cpp - ObjC ARC Optimization ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file defines ObjC ARC optimizations. ARC stands for Automatic
/// Reference Counting and is a system for managing reference counts for objects
/// in Objective C.
///
/// The optimizations performed include elimination of redundant, partially
/// redundant, and inconsequential reference count operations, elimination of
/// redundant weak pointer operations, and numerous minor simplifications.
///
/// WARNING: This file knows about certain library functions. It recognizes them
/// by name, and hardwires knowledge of their semantics.
///
/// WARNING: This file knows about how certain Objective-C library functions are
/// used. Naive LLVM IR transformations which would otherwise be
/// behavior-preserving may break these assumptions.
///
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "objc-arc-opts"
#include "ObjCARC.h"
#include "ARCRuntimeEntryPoints.h"
#include "DependencyAnalysis.h"
#include "ObjCARCAliasAnalysis.h"
#include "ProvenanceAnalysis.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::objcarc;

/// \defgroup MiscUtils Miscellaneous utilities that are not ARC specific.
/// @{

namespace {
  /// \brief An associative container with fast insertion-order (deterministic)
  /// iteration over its elements. Plus the special blot operation.
  template<class KeyT, class ValueT>
  class MapVector {
    /// Map keys to indices in Vector.
    typedef DenseMap<KeyT, size_t> MapTy;
    MapTy Map;

    typedef std::vector<std::pair<KeyT, ValueT> > VectorTy;
    /// Keys and values.
    VectorTy Vector;

  public:
    typedef typename VectorTy::iterator iterator;
    typedef typename VectorTy::const_iterator const_iterator;
    iterator begin() { return Vector.begin(); }
    iterator end() { return Vector.end(); }
    const_iterator begin() const { return Vector.begin(); }
    const_iterator end() const { return Vector.end(); }

#ifdef XDEBUG
    ~MapVector() {
      assert(Vector.size() >= Map.size()); // May differ due to blotting.
      for (typename MapTy::const_iterator I = Map.begin(), E = Map.end();
           I != E; ++I) {
        assert(I->second < Vector.size());
        assert(Vector[I->second].first == I->first);
      }
      for (typename VectorTy::const_iterator I = Vector.begin(),
           E = Vector.end(); I != E; ++I)
        assert(!I->first ||
               (Map.count(I->first) &&
                Map[I->first] == size_t(I - Vector.begin())));
    }
#endif

    ValueT &operator[](const KeyT &Arg) {
      std::pair<typename MapTy::iterator, bool> Pair =
        Map.insert(std::make_pair(Arg, size_t(0)));
      if (Pair.second) {
        size_t Num = Vector.size();
        Pair.first->second = Num;
        Vector.push_back(std::make_pair(Arg, ValueT()));
        return Vector[Num].second;
      }
      return Vector[Pair.first->second].second;
    }

    std::pair<iterator, bool>
    insert(const std::pair<KeyT, ValueT> &InsertPair) {
      std::pair<typename MapTy::iterator, bool> Pair =
        Map.insert(std::make_pair(InsertPair.first, size_t(0)));
      if (Pair.second) {
        size_t Num = Vector.size();
        Pair.first->second = Num;
        Vector.push_back(InsertPair);
        return std::make_pair(Vector.begin() + Num, true);
      }
      return std::make_pair(Vector.begin() + Pair.first->second, false);
    }

    iterator find(const KeyT &Key) {
      typename MapTy::iterator It = Map.find(Key);
      if (It == Map.end()) return Vector.end();
      return Vector.begin() + It->second;
    }

    const_iterator find(const KeyT &Key) const {
      typename MapTy::const_iterator It = Map.find(Key);
      if (It == Map.end()) return Vector.end();
      return Vector.begin() + It->second;
    }

    /// This is similar to erase, but instead of removing the element from the
    /// vector, it just zeros out the key in the vector. This leaves iterators
    /// intact, but clients must be prepared for zeroed-out keys when iterating.
    void blot(const KeyT &Key) {
      typename MapTy::iterator It = Map.find(Key);
      if (It == Map.end()) return;
      Vector[It->second].first = KeyT();
      Map.erase(It);
    }

    void clear() {
      Map.clear();
      Vector.clear();
    }
  };
}

/// @}
///
/// \defgroup ARCUtilities Utility declarations/definitions specific to ARC.
/// @{

/// \brief This is similar to StripPointerCastsAndObjCCalls but it stops as soon
/// as it finds a value with multiple uses.
static const Value *FindSingleUseIdentifiedObject(const Value *Arg) {
  if (Arg->hasOneUse()) {
    if (const BitCastInst *BC = dyn_cast<BitCastInst>(Arg))
      return FindSingleUseIdentifiedObject(BC->getOperand(0));
    if (const GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Arg))
      if (GEP->hasAllZeroIndices())
        return FindSingleUseIdentifiedObject(GEP->getPointerOperand());
    if (IsForwarding(GetBasicInstructionClass(Arg)))
      return FindSingleUseIdentifiedObject(
               cast<CallInst>(Arg)->getArgOperand(0));
    if (!IsObjCIdentifiedObject(Arg))
      return 0;
    return Arg;
  }

  // If we found an identifiable object but it has multiple uses, but they are
  // trivial uses, we can still consider this to be a single-use value.
  if (IsObjCIdentifiedObject(Arg)) {
    for (Value::const_use_iterator UI = Arg->use_begin(), UE = Arg->use_end();
         UI != UE; ++UI) {
      const User *U = *UI;
      if (!U->use_empty() || StripPointerCastsAndObjCCalls(U) != Arg)
         return 0;
    }

    return Arg;
  }

  return 0;
}

/// This is a wrapper around getUnderlyingObjCPtr along the lines of
/// GetUnderlyingObjects except that it returns early when it sees the first
/// alloca.
static inline bool AreAnyUnderlyingObjectsAnAlloca(const Value *V) {
  SmallPtrSet<const Value *, 4> Visited;
  SmallVector<const Value *, 4> Worklist;
  Worklist.push_back(V);
  do {
    const Value *P = Worklist.pop_back_val();
    P = GetUnderlyingObjCPtr(P);

    if (isa<AllocaInst>(P))
      return true;

    if (!Visited.insert(P))
      continue;

    if (const SelectInst *SI = dyn_cast<const SelectInst>(P)) {
      Worklist.push_back(SI->getTrueValue());
      Worklist.push_back(SI->getFalseValue());
      continue;
    }

    if (const PHINode *PN = dyn_cast<const PHINode>(P)) {
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
        Worklist.push_back(PN->getIncomingValue(i));
      continue;
    }
  } while (!Worklist.empty());

  return false;
}


/// @}
///
/// \defgroup ARCOpt ARC Optimization.
/// @{

// TODO: On code like this:
//
// objc_retain(%x)
// stuff_that_cannot_release()
// objc_autorelease(%x)
// stuff_that_cannot_release()
// objc_retain(%x)
// stuff_that_cannot_release()
// objc_autorelease(%x)
//
// The second retain and autorelease can be deleted.

// TODO: It should be possible to delete
// objc_autoreleasePoolPush and objc_autoreleasePoolPop
// pairs if nothing is actually autoreleased between them. Also, autorelease
// calls followed by objc_autoreleasePoolPop calls (perhaps in ObjC++ code
// after inlining) can be turned into plain release calls.

// TODO: Critical-edge splitting. If the optimial insertion point is
// a critical edge, the current algorithm has to fail, because it doesn't
// know how to split edges. It should be possible to make the optimizer
// think in terms of edges, rather than blocks, and then split critical
// edges on demand.

// TODO: OptimizeSequences could generalized to be Interprocedural.

// TODO: Recognize that a bunch of other objc runtime calls have
// non-escaping arguments and non-releasing arguments, and may be
// non-autoreleasing.

// TODO: Sink autorelease calls as far as possible. Unfortunately we
// usually can't sink them past other calls, which would be the main
// case where it would be useful.

// TODO: The pointer returned from objc_loadWeakRetained is retained.

// TODO: Delete release+retain pairs (rare).

STATISTIC(NumNoops,       "Number of no-op objc calls eliminated");
STATISTIC(NumPartialNoops, "Number of partially no-op objc calls eliminated");
STATISTIC(NumAutoreleases,"Number of autoreleases converted to releases");
STATISTIC(NumRets,        "Number of return value forwarding "
                          "retain+autoreleases eliminated");
STATISTIC(NumRRs,         "Number of retain+release paths eliminated");
STATISTIC(NumPeeps,       "Number of calls peephole-optimized");
#ifndef NDEBUG
STATISTIC(NumRetainsBeforeOpt,
          "Number of retains before optimization");
STATISTIC(NumReleasesBeforeOpt,
          "Number of releases before optimization");
STATISTIC(NumRetainsAfterOpt,
          "Number of retains after optimization");
STATISTIC(NumReleasesAfterOpt,
          "Number of releases after optimization");
#endif

namespace {
  /// \enum Sequence
  ///
  /// \brief A sequence of states that a pointer may go through in which an
  /// objc_retain and objc_release are actually needed.
  enum Sequence {
    S_None,
    S_Retain,         ///< objc_retain(x).
    S_CanRelease,     ///< foo(x) -- x could possibly see a ref count decrement.
    S_Use,            ///< any use of x.
    S_Stop,           ///< like S_Release, but code motion is stopped.
    S_Release,        ///< objc_release(x).
    S_MovableRelease  ///< objc_release(x), !clang.imprecise_release.
  };

  raw_ostream &operator<<(raw_ostream &OS, const Sequence S)
    LLVM_ATTRIBUTE_UNUSED;
  raw_ostream &operator<<(raw_ostream &OS, const Sequence S) {
    switch (S) {
    case S_None:
      return OS << "S_None";
    case S_Retain:
      return OS << "S_Retain";
    case S_CanRelease:
      return OS << "S_CanRelease";
    case S_Use:
      return OS << "S_Use";
    case S_Release:
      return OS << "S_Release";
    case S_MovableRelease:
      return OS << "S_MovableRelease";
    case S_Stop:
      return OS << "S_Stop";
    }
    llvm_unreachable("Unknown sequence type.");
  }
}

static Sequence MergeSeqs(Sequence A, Sequence B, bool TopDown) {
  // The easy cases.
  if (A == B)
    return A;
  if (A == S_None || B == S_None)
    return S_None;

  if (A > B) std::swap(A, B);
  if (TopDown) {
    // Choose the side which is further along in the sequence.
    if ((A == S_Retain || A == S_CanRelease) &&
        (B == S_CanRelease || B == S_Use))
      return B;
  } else {
    // Choose the side which is further along in the sequence.
    if ((A == S_Use || A == S_CanRelease) &&
        (B == S_Use || B == S_Release || B == S_Stop || B == S_MovableRelease))
      return A;
    // If both sides are releases, choose the more conservative one.
    if (A == S_Stop && (B == S_Release || B == S_MovableRelease))
      return A;
    if (A == S_Release && B == S_MovableRelease)
      return A;
  }

  return S_None;
}

namespace {
  /// \brief Unidirectional information about either a
  /// retain-decrement-use-release sequence or release-use-decrement-retain
  /// reverse sequence.
  struct RRInfo {
    /// After an objc_retain, the reference count of the referenced
    /// object is known to be positive. Similarly, before an objc_release, the
    /// reference count of the referenced object is known to be positive. If
    /// there are retain-release pairs in code regions where the retain count
    /// is known to be positive, they can be eliminated, regardless of any side
    /// effects between them.
    ///
    /// Also, a retain+release pair nested within another retain+release
    /// pair all on the known same pointer value can be eliminated, regardless
    /// of any intervening side effects.
    ///
    /// KnownSafe is true when either of these conditions is satisfied.
    bool KnownSafe;

    /// True of the objc_release calls are all marked with the "tail" keyword.
    bool IsTailCallRelease;

    /// If the Calls are objc_release calls and they all have a
    /// clang.imprecise_release tag, this is the metadata tag.
    MDNode *ReleaseMetadata;

    /// For a top-down sequence, the set of objc_retains or
    /// objc_retainBlocks. For bottom-up, the set of objc_releases.
    SmallPtrSet<Instruction *, 2> Calls;

    /// The set of optimal insert positions for moving calls in the opposite
    /// sequence.
    SmallPtrSet<Instruction *, 2> ReverseInsertPts;

    /// If this is true, we cannot perform code motion but can still remove
    /// retain/release pairs.
    bool CFGHazardAfflicted;

    RRInfo() :
      KnownSafe(false), IsTailCallRelease(false), ReleaseMetadata(0),
      CFGHazardAfflicted(false) {}

    void clear();

    /// Conservatively merge the two RRInfo. Returns true if a partial merge has
    /// occured, false otherwise.
    bool Merge(const RRInfo &Other);

  };
}

void RRInfo::clear() {
  KnownSafe = false;
  IsTailCallRelease = false;
  ReleaseMetadata = 0;
  Calls.clear();
  ReverseInsertPts.clear();
  CFGHazardAfflicted = false;
}

bool RRInfo::Merge(const RRInfo &Other) {
    // Conservatively merge the ReleaseMetadata information.
    if (ReleaseMetadata != Other.ReleaseMetadata)
      ReleaseMetadata = 0;

    // Conservatively merge the boolean state.
    KnownSafe &= Other.KnownSafe;
    IsTailCallRelease &= Other.IsTailCallRelease;
    CFGHazardAfflicted |= Other.CFGHazardAfflicted;

    // Merge the call sets.
    Calls.insert(Other.Calls.begin(), Other.Calls.end());

    // Merge the insert point sets. If there are any differences,
    // that makes this a partial merge.
    bool Partial = ReverseInsertPts.size() != Other.ReverseInsertPts.size();
    for (SmallPtrSet<Instruction *, 2>::const_iterator
         I = Other.ReverseInsertPts.begin(),
         E = Other.ReverseInsertPts.end(); I != E; ++I)
      Partial |= ReverseInsertPts.insert(*I);
    return Partial;
}

namespace {
  /// \brief This class summarizes several per-pointer runtime properties which
  /// are propogated through the flow graph.
  class PtrState {
    /// True if the reference count is known to be incremented.
    bool KnownPositiveRefCount;

    /// True if we've seen an opportunity for partial RR elimination, such as
    /// pushing calls into a CFG triangle or into one side of a CFG diamond.
    bool Partial;

    /// The current position in the sequence.
    unsigned char Seq : 8;

    /// Unidirectional information about the current sequence.
    RRInfo RRI;

  public:
    PtrState() : KnownPositiveRefCount(false), Partial(false),
                 Seq(S_None) {}


    bool IsKnownSafe() const {
      return RRI.KnownSafe;
    }

    void SetKnownSafe(const bool NewValue) {
      RRI.KnownSafe = NewValue;
    }

    bool IsTailCallRelease() const {
      return RRI.IsTailCallRelease;
    }

    void SetTailCallRelease(const bool NewValue) {
      RRI.IsTailCallRelease = NewValue;
    }

    bool IsTrackingImpreciseReleases() const {
      return RRI.ReleaseMetadata != 0;
    }

    const MDNode *GetReleaseMetadata() const {
      return RRI.ReleaseMetadata;
    }

    void SetReleaseMetadata(MDNode *NewValue) {
      RRI.ReleaseMetadata = NewValue;
    }

    bool IsCFGHazardAfflicted() const {
      return RRI.CFGHazardAfflicted;
    }

    void SetCFGHazardAfflicted(const bool NewValue) {
      RRI.CFGHazardAfflicted = NewValue;
    }

    void SetKnownPositiveRefCount() {
      DEBUG(dbgs() << "Setting Known Positive.\n");
      KnownPositiveRefCount = true;
    }

    void ClearKnownPositiveRefCount() {
      DEBUG(dbgs() << "Clearing Known Positive.\n");
      KnownPositiveRefCount = false;
    }

    bool HasKnownPositiveRefCount() const {
      return KnownPositiveRefCount;
    }

    void SetSeq(Sequence NewSeq) {
      DEBUG(dbgs() << "Old: " << Seq << "; New: " << NewSeq << "\n");
      Seq = NewSeq;
    }

    Sequence GetSeq() const {
      return static_cast<Sequence>(Seq);
    }

    void ClearSequenceProgress() {
      ResetSequenceProgress(S_None);
    }

    void ResetSequenceProgress(Sequence NewSeq) {
      DEBUG(dbgs() << "Resetting sequence progress.\n");
      SetSeq(NewSeq);
      Partial = false;
      RRI.clear();
    }

    void Merge(const PtrState &Other, bool TopDown);

    void InsertCall(Instruction *I) {
      RRI.Calls.insert(I);
    }

    void InsertReverseInsertPt(Instruction *I) {
      RRI.ReverseInsertPts.insert(I);
    }

    void ClearReverseInsertPts() {
      RRI.ReverseInsertPts.clear();
    }

    bool HasReverseInsertPts() const {
      return !RRI.ReverseInsertPts.empty();
    }

    const RRInfo &GetRRInfo() const {
      return RRI;
    }
  };
}

void
PtrState::Merge(const PtrState &Other, bool TopDown) {
  Seq = MergeSeqs(static_cast<Sequence>(Seq), static_cast<Sequence>(Other.Seq),
                  TopDown);
  KnownPositiveRefCount &= Other.KnownPositiveRefCount;

  // If we're not in a sequence (anymore), drop all associated state.
  if (Seq == S_None) {
    Partial = false;
    RRI.clear();
  } else if (Partial || Other.Partial) {
    // If we're doing a merge on a path that's previously seen a partial
    // merge, conservatively drop the sequence, to avoid doing partial
    // RR elimination. If the branch predicates for the two merge differ,
    // mixing them is unsafe.
    ClearSequenceProgress();
  } else {
    // Otherwise merge the other PtrState's RRInfo into our RRInfo. At this
    // point, we know that currently we are not partial. Stash whether or not
    // the merge operation caused us to undergo a partial merging of reverse
    // insertion points.
    Partial = RRI.Merge(Other.RRI);
  }
}

namespace {
  /// \brief Per-BasicBlock state.
  class BBState {
    /// The number of unique control paths from the entry which can reach this
    /// block.
    unsigned TopDownPathCount;

    /// The number of unique control paths to exits from this block.
    unsigned BottomUpPathCount;

    /// A type for PerPtrTopDown and PerPtrBottomUp.
    typedef MapVector<const Value *, PtrState> MapTy;

    /// The top-down traversal uses this to record information known about a
    /// pointer at the bottom of each block.
    MapTy PerPtrTopDown;

    /// The bottom-up traversal uses this to record information known about a
    /// pointer at the top of each block.
    MapTy PerPtrBottomUp;

    /// Effective predecessors of the current block ignoring ignorable edges and
    /// ignored backedges.
    SmallVector<BasicBlock *, 2> Preds;
    /// Effective successors of the current block ignoring ignorable edges and
    /// ignored backedges.
    SmallVector<BasicBlock *, 2> Succs;

  public:
    static const unsigned OverflowOccurredValue;

    BBState() : TopDownPathCount(0), BottomUpPathCount(0) { }

    typedef MapTy::iterator ptr_iterator;
    typedef MapTy::const_iterator ptr_const_iterator;

    ptr_iterator top_down_ptr_begin() { return PerPtrTopDown.begin(); }
    ptr_iterator top_down_ptr_end() { return PerPtrTopDown.end(); }
    ptr_const_iterator top_down_ptr_begin() const {
      return PerPtrTopDown.begin();
    }
    ptr_const_iterator top_down_ptr_end() const {
      return PerPtrTopDown.end();
    }

    ptr_iterator bottom_up_ptr_begin() { return PerPtrBottomUp.begin(); }
    ptr_iterator bottom_up_ptr_end() { return PerPtrBottomUp.end(); }
    ptr_const_iterator bottom_up_ptr_begin() const {
      return PerPtrBottomUp.begin();
    }
    ptr_const_iterator bottom_up_ptr_end() const {
      return PerPtrBottomUp.end();
    }

    /// Mark this block as being an entry block, which has one path from the
    /// entry by definition.
    void SetAsEntry() { TopDownPathCount = 1; }

    /// Mark this block as being an exit block, which has one path to an exit by
    /// definition.
    void SetAsExit()  { BottomUpPathCount = 1; }

    /// Attempt to find the PtrState object describing the top down state for
    /// pointer Arg. Return a new initialized PtrState describing the top down
    /// state for Arg if we do not find one.
    PtrState &getPtrTopDownState(const Value *Arg) {
      return PerPtrTopDown[Arg];
    }

    /// Attempt to find the PtrState object describing the bottom up state for
    /// pointer Arg. Return a new initialized PtrState describing the bottom up
    /// state for Arg if we do not find one.
    PtrState &getPtrBottomUpState(const Value *Arg) {
      return PerPtrBottomUp[Arg];
    }

    /// Attempt to find the PtrState object describing the bottom up state for
    /// pointer Arg.
    ptr_iterator findPtrBottomUpState(const Value *Arg) {
      return PerPtrBottomUp.find(Arg);
    }

    void clearBottomUpPointers() {
      PerPtrBottomUp.clear();
    }

    void clearTopDownPointers() {
      PerPtrTopDown.clear();
    }

    void InitFromPred(const BBState &Other);
    void InitFromSucc(const BBState &Other);
    void MergePred(const BBState &Other);
    void MergeSucc(const BBState &Other);

    /// Compute the number of possible unique paths from an entry to an exit
    /// which pass through this block. This is only valid after both the
    /// top-down and bottom-up traversals are complete.
    ///
    /// Returns true if overflow occured. Returns false if overflow did not
    /// occur.
    bool GetAllPathCountWithOverflow(unsigned &PathCount) const {
      if (TopDownPathCount == OverflowOccurredValue ||
          BottomUpPathCount == OverflowOccurredValue)
        return true;
      unsigned long long Product =
        (unsigned long long)TopDownPathCount*BottomUpPathCount;
      // Overflow occured if any of the upper bits of Product are set or if all
      // the lower bits of Product are all set.
      return (Product >> 32) ||
             ((PathCount = Product) == OverflowOccurredValue);
    }

    // Specialized CFG utilities.
    typedef SmallVectorImpl<BasicBlock *>::const_iterator edge_iterator;
    edge_iterator pred_begin() const { return Preds.begin(); }
    edge_iterator pred_end() const { return Preds.end(); }
    edge_iterator succ_begin() const { return Succs.begin(); }
    edge_iterator succ_end() const { return Succs.end(); }

    void addSucc(BasicBlock *Succ) { Succs.push_back(Succ); }
    void addPred(BasicBlock *Pred) { Preds.push_back(Pred); }

    bool isExit() const { return Succs.empty(); }
  };

  const unsigned BBState::OverflowOccurredValue = 0xffffffff;
}

void BBState::InitFromPred(const BBState &Other) {
  PerPtrTopDown = Other.PerPtrTopDown;
  TopDownPathCount = Other.TopDownPathCount;
}

void BBState::InitFromSucc(const BBState &Other) {
  PerPtrBottomUp = Other.PerPtrBottomUp;
  BottomUpPathCount = Other.BottomUpPathCount;
}

/// The top-down traversal uses this to merge information about predecessors to
/// form the initial state for a new block.
void BBState::MergePred(const BBState &Other) {
  if (TopDownPathCount == OverflowOccurredValue)
    return;

  // Other.TopDownPathCount can be 0, in which case it is either dead or a
  // loop backedge. Loop backedges are special.
  TopDownPathCount += Other.TopDownPathCount;

  // In order to be consistent, we clear the top down pointers when by adding
  // TopDownPathCount becomes OverflowOccurredValue even though "true" overflow
  // has not occured.
  if (TopDownPathCount == OverflowOccurredValue) {
    clearTopDownPointers();
    return;
  }

  // Check for overflow. If we have overflow, fall back to conservative
  // behavior.
  if (TopDownPathCount < Other.TopDownPathCount) {
    TopDownPathCount = OverflowOccurredValue;
    clearTopDownPointers();
    return;
  }

  // For each entry in the other set, if our set has an entry with the same key,
  // merge the entries. Otherwise, copy the entry and merge it with an empty
  // entry.
  for (ptr_const_iterator MI = Other.top_down_ptr_begin(),
       ME = Other.top_down_ptr_end(); MI != ME; ++MI) {
    std::pair<ptr_iterator, bool> Pair = PerPtrTopDown.insert(*MI);
    Pair.first->second.Merge(Pair.second ? PtrState() : MI->second,
                             /*TopDown=*/true);
  }

  // For each entry in our set, if the other set doesn't have an entry with the
  // same key, force it to merge with an empty entry.
  for (ptr_iterator MI = top_down_ptr_begin(),
       ME = top_down_ptr_end(); MI != ME; ++MI)
    if (Other.PerPtrTopDown.find(MI->first) == Other.PerPtrTopDown.end())
      MI->second.Merge(PtrState(), /*TopDown=*/true);
}

/// The bottom-up traversal uses this to merge information about successors to
/// form the initial state for a new block.
void BBState::MergeSucc(const BBState &Other) {
  if (BottomUpPathCount == OverflowOccurredValue)
    return;

  // Other.BottomUpPathCount can be 0, in which case it is either dead or a
  // loop backedge. Loop backedges are special.
  BottomUpPathCount += Other.BottomUpPathCount;

  // In order to be consistent, we clear the top down pointers when by adding
  // BottomUpPathCount becomes OverflowOccurredValue even though "true" overflow
  // has not occured.
  if (BottomUpPathCount == OverflowOccurredValue) {
    clearBottomUpPointers();
    return;
  }

  // Check for overflow. If we have overflow, fall back to conservative
  // behavior.
  if (BottomUpPathCount < Other.BottomUpPathCount) {
    BottomUpPathCount = OverflowOccurredValue;
    clearBottomUpPointers();
    return;
  }

  // For each entry in the other set, if our set has an entry with the
  // same key, merge the entries. Otherwise, copy the entry and merge
  // it with an empty entry.
  for (ptr_const_iterator MI = Other.bottom_up_ptr_begin(),
       ME = Other.bottom_up_ptr_end(); MI != ME; ++MI) {
    std::pair<ptr_iterator, bool> Pair = PerPtrBottomUp.insert(*MI);
    Pair.first->second.Merge(Pair.second ? PtrState() : MI->second,
                             /*TopDown=*/false);
  }

  // For each entry in our set, if the other set doesn't have an entry
  // with the same key, force it to merge with an empty entry.
  for (ptr_iterator MI = bottom_up_ptr_begin(),
       ME = bottom_up_ptr_end(); MI != ME; ++MI)
    if (Other.PerPtrBottomUp.find(MI->first) == Other.PerPtrBottomUp.end())
      MI->second.Merge(PtrState(), /*TopDown=*/false);
}

// Only enable ARC Annotations if we are building a debug version of
// libObjCARCOpts.
#ifndef NDEBUG
#define ARC_ANNOTATIONS
#endif

// Define some macros along the lines of DEBUG and some helper functions to make
// it cleaner to create annotations in the source code and to no-op when not
// building in debug mode.
#ifdef ARC_ANNOTATIONS

#include "llvm/Support/CommandLine.h"

/// Enable/disable ARC sequence annotations.
static cl::opt<bool>
EnableARCAnnotations("enable-objc-arc-annotations", cl::init(false),
                     cl::desc("Enable emission of arc data flow analysis "
                              "annotations"));
static cl::opt<bool>
DisableCheckForCFGHazards("disable-objc-arc-checkforcfghazards", cl::init(false),
                          cl::desc("Disable check for cfg hazards when "
                                   "annotating"));
static cl::opt<std::string>
ARCAnnotationTargetIdentifier("objc-arc-annotation-target-identifier",
                              cl::init(""),
                              cl::desc("filter out all data flow annotations "
                                       "but those that apply to the given "
                                       "target llvm identifier."));

/// This function appends a unique ARCAnnotationProvenanceSourceMDKind id to an
/// instruction so that we can track backwards when post processing via the llvm
/// arc annotation processor tool. If the function is an
static MDString *AppendMDNodeToSourcePtr(unsigned NodeId,
                                         Value *Ptr) {
  MDString *Hash = 0;

  // If pointer is a result of an instruction and it does not have a source
  // MDNode it, attach a new MDNode onto it. If pointer is a result of
  // an instruction and does have a source MDNode attached to it, return a
  // reference to said Node. Otherwise just return 0.
  if (Instruction *Inst = dyn_cast<Instruction>(Ptr)) {
    MDNode *Node;
    if (!(Node = Inst->getMetadata(NodeId))) {
      // We do not have any node. Generate and attatch the hash MDString to the
      // instruction.

      // We just use an MDString to ensure that this metadata gets written out
      // of line at the module level and to provide a very simple format
      // encoding the information herein. Both of these makes it simpler to
      // parse the annotations by a simple external program.
      std::string Str;
      raw_string_ostream os(Str);
      os << "(" << Inst->getParent()->getParent()->getName() << ",%"
         << Inst->getName() << ")";

      Hash = MDString::get(Inst->getContext(), os.str());
      Inst->setMetadata(NodeId, MDNode::get(Inst->getContext(),Hash));
    } else {
      // We have a node. Grab its hash and return it.
      assert(Node->getNumOperands() == 1 &&
        "An ARCAnnotationProvenanceSourceMDKind can only have 1 operand.");
      Hash = cast<MDString>(Node->getOperand(0));
    }
  } else if (Argument *Arg = dyn_cast<Argument>(Ptr)) {
    std::string str;
    raw_string_ostream os(str);
    os << "(" << Arg->getParent()->getName() << ",%" << Arg->getName()
       << ")";
    Hash = MDString::get(Arg->getContext(), os.str());
  }

  return Hash;
}

static std::string SequenceToString(Sequence A) {
  std::string str;
  raw_string_ostream os(str);
  os << A;
  return os.str();
}

/// Helper function to change a Sequence into a String object using our overload
/// for raw_ostream so we only have printing code in one location.
static MDString *SequenceToMDString(LLVMContext &Context,
                                    Sequence A) {
  return MDString::get(Context, SequenceToString(A));
}

/// A simple function to generate a MDNode which describes the change in state
/// for Value *Ptr caused by Instruction *Inst.
static void AppendMDNodeToInstForPtr(unsigned NodeId,
                                     Instruction *Inst,
                                     Value *Ptr,
                                     MDString *PtrSourceMDNodeID,
                                     Sequence OldSeq,
                                     Sequence NewSeq) {
  MDNode *Node = 0;
  Value *tmp[3] = {PtrSourceMDNodeID,
                   SequenceToMDString(Inst->getContext(),
                                      OldSeq),
                   SequenceToMDString(Inst->getContext(),
                                      NewSeq)};
  Node = MDNode::get(Inst->getContext(),
                     ArrayRef<Value*>(tmp, 3));

  Inst->setMetadata(NodeId, Node);
}

/// Add to the beginning of the basic block llvm.ptr.annotations which show the
/// state of a pointer at the entrance to a basic block.
static void GenerateARCBBEntranceAnnotation(const char *Name, BasicBlock *BB,
                                            Value *Ptr, Sequence Seq) {
  // If we have a target identifier, make sure that we match it before
  // continuing.
  if(!ARCAnnotationTargetIdentifier.empty() &&
     !Ptr->getName().equals(ARCAnnotationTargetIdentifier))
    return;

  Module *M = BB->getParent()->getParent();
  LLVMContext &C = M->getContext();
  Type *I8X = PointerType::getUnqual(Type::getInt8Ty(C));
  Type *I8XX = PointerType::getUnqual(I8X);
  Type *Params[] = {I8XX, I8XX};
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(C),
                                        ArrayRef<Type*>(Params, 2),
                                        /*isVarArg=*/false);
  Constant *Callee = M->getOrInsertFunction(Name, FTy);

  IRBuilder<> Builder(BB, BB->getFirstInsertionPt());

  Value *PtrName;
  StringRef Tmp = Ptr->getName();
  if (0 == (PtrName = M->getGlobalVariable(Tmp, true))) {
    Value *ActualPtrName = Builder.CreateGlobalStringPtr(Tmp,
                                                         Tmp + "_STR");
    PtrName = new GlobalVariable(*M, I8X, true, GlobalVariable::InternalLinkage,
                                 cast<Constant>(ActualPtrName), Tmp);
  }

  Value *S;
  std::string SeqStr = SequenceToString(Seq);
  if (0 == (S = M->getGlobalVariable(SeqStr, true))) {
    Value *ActualPtrName = Builder.CreateGlobalStringPtr(SeqStr,
                                                         SeqStr + "_STR");
    S = new GlobalVariable(*M, I8X, true, GlobalVariable::InternalLinkage,
                           cast<Constant>(ActualPtrName), SeqStr);
  }

  Builder.CreateCall2(Callee, PtrName, S);
}

/// Add to the end of the basic block llvm.ptr.annotations which show the state
/// of the pointer at the bottom of the basic block.
static void GenerateARCBBTerminatorAnnotation(const char *Name, BasicBlock *BB,
                                              Value *Ptr, Sequence Seq) {
  // If we have a target identifier, make sure that we match it before emitting
  // an annotation.
  if(!ARCAnnotationTargetIdentifier.empty() &&
     !Ptr->getName().equals(ARCAnnotationTargetIdentifier))
    return;

  Module *M = BB->getParent()->getParent();
  LLVMContext &C = M->getContext();
  Type *I8X = PointerType::getUnqual(Type::getInt8Ty(C));
  Type *I8XX = PointerType::getUnqual(I8X);
  Type *Params[] = {I8XX, I8XX};
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(C),
                                        ArrayRef<Type*>(Params, 2),
                                        /*isVarArg=*/false);
  Constant *Callee = M->getOrInsertFunction(Name, FTy);

  IRBuilder<> Builder(BB, llvm::prior(BB->end()));

  Value *PtrName;
  StringRef Tmp = Ptr->getName();
  if (0 == (PtrName = M->getGlobalVariable(Tmp, true))) {
    Value *ActualPtrName = Builder.CreateGlobalStringPtr(Tmp,
                                                         Tmp + "_STR");
    PtrName = new GlobalVariable(*M, I8X, true, GlobalVariable::InternalLinkage,
                                 cast<Constant>(ActualPtrName), Tmp);
  }

  Value *S;
  std::string SeqStr = SequenceToString(Seq);
  if (0 == (S = M->getGlobalVariable(SeqStr, true))) {
    Value *ActualPtrName = Builder.CreateGlobalStringPtr(SeqStr,
                                                         SeqStr + "_STR");
    S = new GlobalVariable(*M, I8X, true, GlobalVariable::InternalLinkage,
                           cast<Constant>(ActualPtrName), SeqStr);
  }
  Builder.CreateCall2(Callee, PtrName, S);
}

/// Adds a source annotation to pointer and a state change annotation to Inst
/// referencing the source annotation and the old/new state of pointer.
static void GenerateARCAnnotation(unsigned InstMDId,
                                  unsigned PtrMDId,
                                  Instruction *Inst,
                                  Value *Ptr,
                                  Sequence OldSeq,
                                  Sequence NewSeq) {
  if (EnableARCAnnotations) {
    // If we have a target identifier, make sure that we match it before
    // emitting an annotation.
    if(!ARCAnnotationTargetIdentifier.empty() &&
       !Ptr->getName().equals(ARCAnnotationTargetIdentifier))
      return;

    // First generate the source annotation on our pointer. This will return an
    // MDString* if Ptr actually comes from an instruction implying we can put
    // in a source annotation. If AppendMDNodeToSourcePtr returns 0 (i.e. NULL),
    // then we know that our pointer is from an Argument so we put a reference
    // to the argument number.
    //
    // The point of this is to make it easy for the
    // llvm-arc-annotation-processor tool to cross reference where the source
    // pointer is in the LLVM IR since the LLVM IR parser does not submit such
    // information via debug info for backends to use (since why would anyone
    // need such a thing from LLVM IR besides in non standard cases
    // [i.e. this]).
    MDString *SourcePtrMDNode =
      AppendMDNodeToSourcePtr(PtrMDId, Ptr);
    AppendMDNodeToInstForPtr(InstMDId, Inst, Ptr, SourcePtrMDNode, OldSeq,
                             NewSeq);
  }
}

// The actual interface for accessing the above functionality is defined via
// some simple macros which are defined below. We do this so that the user does
// not need to pass in what metadata id is needed resulting in cleaner code and
// additionally since it provides an easy way to conditionally no-op all
// annotation support in a non-debug build.

/// Use this macro to annotate a sequence state change when processing
/// instructions bottom up,
#define ANNOTATE_BOTTOMUP(inst, ptr, old, new)                          \
  GenerateARCAnnotation(ARCAnnotationBottomUpMDKind,                    \
                        ARCAnnotationProvenanceSourceMDKind, (inst),    \
                        const_cast<Value*>(ptr), (old), (new))
/// Use this macro to annotate a sequence state change when processing
/// instructions top down.
#define ANNOTATE_TOPDOWN(inst, ptr, old, new)                           \
  GenerateARCAnnotation(ARCAnnotationTopDownMDKind,                     \
                        ARCAnnotationProvenanceSourceMDKind, (inst),    \
                        const_cast<Value*>(ptr), (old), (new))

#define ANNOTATE_BB(_states, _bb, _name, _type, _direction)                   \
  do {                                                                        \
    if (EnableARCAnnotations) {                                               \
      for(BBState::ptr_const_iterator I = (_states)._direction##_ptr_begin(), \
          E = (_states)._direction##_ptr_end(); I != E; ++I) {                \
        Value *Ptr = const_cast<Value*>(I->first);                            \
        Sequence Seq = I->second.GetSeq();                                    \
        GenerateARCBB ## _type ## Annotation(_name, (_bb), Ptr, Seq);         \
      }                                                                       \
    }                                                                         \
  } while (0)

#define ANNOTATE_BOTTOMUP_BBSTART(_states, _basicblock)                       \
    ANNOTATE_BB(_states, _basicblock, "llvm.arc.annotation.bottomup.bbstart", \
                Entrance, bottom_up)
#define ANNOTATE_BOTTOMUP_BBEND(_states, _basicblock)                         \
    ANNOTATE_BB(_states, _basicblock, "llvm.arc.annotation.bottomup.bbend",   \
                Terminator, bottom_up)
#define ANNOTATE_TOPDOWN_BBSTART(_states, _basicblock)                        \
    ANNOTATE_BB(_states, _basicblock, "llvm.arc.annotation.topdown.bbstart",  \
                Entrance, top_down)
#define ANNOTATE_TOPDOWN_BBEND(_states, _basicblock)                          \
    ANNOTATE_BB(_states, _basicblock, "llvm.arc.annotation.topdown.bbend",    \
                Terminator, top_down)

#else // !ARC_ANNOTATION
// If annotations are off, noop.
#define ANNOTATE_BOTTOMUP(inst, ptr, old, new)
#define ANNOTATE_TOPDOWN(inst, ptr, old, new)
#define ANNOTATE_BOTTOMUP_BBSTART(states, basicblock)
#define ANNOTATE_BOTTOMUP_BBEND(states, basicblock)
#define ANNOTATE_TOPDOWN_BBSTART(states, basicblock)
#define ANNOTATE_TOPDOWN_BBEND(states, basicblock)
#endif // !ARC_ANNOTATION

namespace {
  /// \brief The main ARC optimization pass.
  class ObjCARCOpt : public FunctionPass {
    bool Changed;
    ProvenanceAnalysis PA;
    ARCRuntimeEntryPoints EP;

    // This is used to track if a pointer is stored into an alloca.
    DenseSet<const Value *> MultiOwnersSet;

    /// A flag indicating whether this optimization pass should run.
    bool Run;

    /// Flags which determine whether each of the interesting runtine functions
    /// is in fact used in the current function.
    unsigned UsedInThisFunction;

    /// The Metadata Kind for clang.imprecise_release metadata.
    unsigned ImpreciseReleaseMDKind;

    /// The Metadata Kind for clang.arc.copy_on_escape metadata.
    unsigned CopyOnEscapeMDKind;

    /// The Metadata Kind for clang.arc.no_objc_arc_exceptions metadata.
    unsigned NoObjCARCExceptionsMDKind;

#ifdef ARC_ANNOTATIONS
    /// The Metadata Kind for llvm.arc.annotation.bottomup metadata.
    unsigned ARCAnnotationBottomUpMDKind;
    /// The Metadata Kind for llvm.arc.annotation.topdown metadata.
    unsigned ARCAnnotationTopDownMDKind;
    /// The Metadata Kind for llvm.arc.annotation.provenancesource metadata.
    unsigned ARCAnnotationProvenanceSourceMDKind;
#endif // ARC_ANNOATIONS

    bool OptimizeRetainRVCall(Function &F, Instruction *RetainRV);
    void OptimizeAutoreleaseRVCall(Function &F, Instruction *AutoreleaseRV,
                                   InstructionClass &Class);
    void OptimizeIndividualCalls(Function &F);

    void CheckForCFGHazards(const BasicBlock *BB,
                            DenseMap<const BasicBlock *, BBState> &BBStates,
                            BBState &MyStates) const;
    bool VisitInstructionBottomUp(Instruction *Inst,
                                  BasicBlock *BB,
                                  MapVector<Value *, RRInfo> &Retains,
                                  BBState &MyStates);
    bool VisitBottomUp(BasicBlock *BB,
                       DenseMap<const BasicBlock *, BBState> &BBStates,
                       MapVector<Value *, RRInfo> &Retains);
    bool VisitInstructionTopDown(Instruction *Inst,
                                 DenseMap<Value *, RRInfo> &Releases,
                                 BBState &MyStates);
    bool VisitTopDown(BasicBlock *BB,
                      DenseMap<const BasicBlock *, BBState> &BBStates,
                      DenseMap<Value *, RRInfo> &Releases);
    bool Visit(Function &F,
               DenseMap<const BasicBlock *, BBState> &BBStates,
               MapVector<Value *, RRInfo> &Retains,
               DenseMap<Value *, RRInfo> &Releases);

    void MoveCalls(Value *Arg, RRInfo &RetainsToMove, RRInfo &ReleasesToMove,
                   MapVector<Value *, RRInfo> &Retains,
                   DenseMap<Value *, RRInfo> &Releases,
                   SmallVectorImpl<Instruction *> &DeadInsts,
                   Module *M);

    bool ConnectTDBUTraversals(DenseMap<const BasicBlock *, BBState> &BBStates,
                               MapVector<Value *, RRInfo> &Retains,
                               DenseMap<Value *, RRInfo> &Releases,
                               Module *M,
                               SmallVectorImpl<Instruction *> &NewRetains,
                               SmallVectorImpl<Instruction *> &NewReleases,
                               SmallVectorImpl<Instruction *> &DeadInsts,
                               RRInfo &RetainsToMove,
                               RRInfo &ReleasesToMove,
                               Value *Arg,
                               bool KnownSafe,
                               bool &AnyPairsCompletelyEliminated);

    bool PerformCodePlacement(DenseMap<const BasicBlock *, BBState> &BBStates,
                              MapVector<Value *, RRInfo> &Retains,
                              DenseMap<Value *, RRInfo> &Releases,
                              Module *M);

    void OptimizeWeakCalls(Function &F);

    bool OptimizeSequences(Function &F);

    void OptimizeReturns(Function &F);

#ifndef NDEBUG
    void GatherStatistics(Function &F, bool AfterOptimization = false);
#endif

    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    virtual bool doInitialization(Module &M);
    virtual bool runOnFunction(Function &F);
    virtual void releaseMemory();

  public:
    static char ID;
    ObjCARCOpt() : FunctionPass(ID) {
      initializeObjCARCOptPass(*PassRegistry::getPassRegistry());
    }
  };
}

char ObjCARCOpt::ID = 0;
INITIALIZE_PASS_BEGIN(ObjCARCOpt,
                      "objc-arc", "ObjC ARC optimization", false, false)
INITIALIZE_PASS_DEPENDENCY(ObjCARCAliasAnalysis)
INITIALIZE_PASS_END(ObjCARCOpt,
                    "objc-arc", "ObjC ARC optimization", false, false)

Pass *llvm::createObjCARCOptPass() {
  return new ObjCARCOpt();
}

void ObjCARCOpt::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<ObjCARCAliasAnalysis>();
  AU.addRequired<AliasAnalysis>();
  // ARC optimization doesn't currently split critical edges.
  AU.setPreservesCFG();
}

/// Turn objc_retainAutoreleasedReturnValue into objc_retain if the operand is
/// not a return value.  Or, if it can be paired with an
/// objc_autoreleaseReturnValue, delete the pair and return true.
bool
ObjCARCOpt::OptimizeRetainRVCall(Function &F, Instruction *RetainRV) {
  // Check for the argument being from an immediately preceding call or invoke.
  const Value *Arg = GetObjCArg(RetainRV);
  ImmutableCallSite CS(Arg);
  if (const Instruction *Call = CS.getInstruction()) {
    if (Call->getParent() == RetainRV->getParent()) {
      BasicBlock::const_iterator I = Call;
      ++I;
      while (IsNoopInstruction(I)) ++I;
      if (&*I == RetainRV)
        return false;
    } else if (const InvokeInst *II = dyn_cast<InvokeInst>(Call)) {
      BasicBlock *RetainRVParent = RetainRV->getParent();
      if (II->getNormalDest() == RetainRVParent) {
        BasicBlock::const_iterator I = RetainRVParent->begin();
        while (IsNoopInstruction(I)) ++I;
        if (&*I == RetainRV)
          return false;
      }
    }
  }

  // Check for being preceded by an objc_autoreleaseReturnValue on the same
  // pointer. In this case, we can delete the pair.
  BasicBlock::iterator I = RetainRV, Begin = RetainRV->getParent()->begin();
  if (I != Begin) {
    do --I; while (I != Begin && IsNoopInstruction(I));
    if (GetBasicInstructionClass(I) == IC_AutoreleaseRV &&
        GetObjCArg(I) == Arg) {
      Changed = true;
      ++NumPeeps;

      DEBUG(dbgs() << "Erasing autoreleaseRV,retainRV pair: " << *I << "\n"
                   << "Erasing " << *RetainRV << "\n");

      EraseInstruction(I);
      EraseInstruction(RetainRV);
      return true;
    }
  }

  // Turn it to a plain objc_retain.
  Changed = true;
  ++NumPeeps;

  DEBUG(dbgs() << "Transforming objc_retainAutoreleasedReturnValue => "
                  "objc_retain since the operand is not a return value.\n"
                  "Old = " << *RetainRV << "\n");

  Constant *NewDecl = EP.get(ARCRuntimeEntryPoints::EPT_Retain);
  cast<CallInst>(RetainRV)->setCalledFunction(NewDecl);

  DEBUG(dbgs() << "New = " << *RetainRV << "\n");

  return false;
}

/// Turn objc_autoreleaseReturnValue into objc_autorelease if the result is not
/// used as a return value.
void
ObjCARCOpt::OptimizeAutoreleaseRVCall(Function &F, Instruction *AutoreleaseRV,
                                      InstructionClass &Class) {
  // Check for a return of the pointer value.
  const Value *Ptr = GetObjCArg(AutoreleaseRV);
  SmallVector<const Value *, 2> Users;
  Users.push_back(Ptr);
  do {
    Ptr = Users.pop_back_val();
    for (Value::const_use_iterator UI = Ptr->use_begin(), UE = Ptr->use_end();
         UI != UE; ++UI) {
      const User *I = *UI;
      if (isa<ReturnInst>(I) || GetBasicInstructionClass(I) == IC_RetainRV)
        return;
      if (isa<BitCastInst>(I))
        Users.push_back(I);
    }
  } while (!Users.empty());

  Changed = true;
  ++NumPeeps;

  DEBUG(dbgs() << "Transforming objc_autoreleaseReturnValue => "
                  "objc_autorelease since its operand is not used as a return "
                  "value.\n"
                  "Old = " << *AutoreleaseRV << "\n");

  CallInst *AutoreleaseRVCI = cast<CallInst>(AutoreleaseRV);
  Constant *NewDecl = EP.get(ARCRuntimeEntryPoints::EPT_Autorelease);
  AutoreleaseRVCI->setCalledFunction(NewDecl);
  AutoreleaseRVCI->setTailCall(false); // Never tail call objc_autorelease.
  Class = IC_Autorelease;

  DEBUG(dbgs() << "New: " << *AutoreleaseRV << "\n");

}

/// Visit each call, one at a time, and make simplifications without doing any
/// additional analysis.
void ObjCARCOpt::OptimizeIndividualCalls(Function &F) {
  DEBUG(dbgs() << "\n== ObjCARCOpt::OptimizeIndividualCalls ==\n");
  // Reset all the flags in preparation for recomputing them.
  UsedInThisFunction = 0;

  // Visit all objc_* calls in F.
  for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ) {
    Instruction *Inst = &*I++;

    InstructionClass Class = GetBasicInstructionClass(Inst);

    DEBUG(dbgs() << "Visiting: Class: " << Class << "; " << *Inst << "\n");

    switch (Class) {
    default: break;

    // Delete no-op casts. These function calls have special semantics, but
    // the semantics are entirely implemented via lowering in the front-end,
    // so by the time they reach the optimizer, they are just no-op calls
    // which return their argument.
    //
    // There are gray areas here, as the ability to cast reference-counted
    // pointers to raw void* and back allows code to break ARC assumptions,
    // however these are currently considered to be unimportant.
    case IC_NoopCast:
      Changed = true;
      ++NumNoops;
      DEBUG(dbgs() << "Erasing no-op cast: " << *Inst << "\n");
      EraseInstruction(Inst);
      continue;

    // If the pointer-to-weak-pointer is null, it's undefined behavior.
    case IC_StoreWeak:
    case IC_LoadWeak:
    case IC_LoadWeakRetained:
    case IC_InitWeak:
    case IC_DestroyWeak: {
      CallInst *CI = cast<CallInst>(Inst);
      if (IsNullOrUndef(CI->getArgOperand(0))) {
        Changed = true;
        Type *Ty = CI->getArgOperand(0)->getType();
        new StoreInst(UndefValue::get(cast<PointerType>(Ty)->getElementType()),
                      Constant::getNullValue(Ty),
                      CI);
        llvm::Value *NewValue = UndefValue::get(CI->getType());
        DEBUG(dbgs() << "A null pointer-to-weak-pointer is undefined behavior."
                       "\nOld = " << *CI << "\nNew = " << *NewValue << "\n");
        CI->replaceAllUsesWith(NewValue);
        CI->eraseFromParent();
        continue;
      }
      break;
    }
    case IC_CopyWeak:
    case IC_MoveWeak: {
      CallInst *CI = cast<CallInst>(Inst);
      if (IsNullOrUndef(CI->getArgOperand(0)) ||
          IsNullOrUndef(CI->getArgOperand(1))) {
        Changed = true;
        Type *Ty = CI->getArgOperand(0)->getType();
        new StoreInst(UndefValue::get(cast<PointerType>(Ty)->getElementType()),
                      Constant::getNullValue(Ty),
                      CI);

        llvm::Value *NewValue = UndefValue::get(CI->getType());
        DEBUG(dbgs() << "A null pointer-to-weak-pointer is undefined behavior."
                        "\nOld = " << *CI << "\nNew = " << *NewValue << "\n");

        CI->replaceAllUsesWith(NewValue);
        CI->eraseFromParent();
        continue;
      }
      break;
    }
    case IC_RetainRV:
      if (OptimizeRetainRVCall(F, Inst))
        continue;
      break;
    case IC_AutoreleaseRV:
      OptimizeAutoreleaseRVCall(F, Inst, Class);
      break;
    }

    // objc_autorelease(x) -> objc_release(x) if x is otherwise unused.
    if (IsAutorelease(Class) && Inst->use_empty()) {
      CallInst *Call = cast<CallInst>(Inst);
      const Value *Arg = Call->getArgOperand(0);
      Arg = FindSingleUseIdentifiedObject(Arg);
      if (Arg) {
        Changed = true;
        ++NumAutoreleases;

        // Create the declaration lazily.
        LLVMContext &C = Inst->getContext();

        Constant *Decl = EP.get(ARCRuntimeEntryPoints::EPT_Release);
        CallInst *NewCall = CallInst::Create(Decl, Call->getArgOperand(0), "",
                                             Call);
        NewCall->setMetadata(ImpreciseReleaseMDKind, MDNode::get(C, None));

        DEBUG(dbgs() << "Replacing autorelease{,RV}(x) with objc_release(x) "
              "since x is otherwise unused.\nOld: " << *Call << "\nNew: "
              << *NewCall << "\n");

        EraseInstruction(Call);
        Inst = NewCall;
        Class = IC_Release;
      }
    }

    // For functions which can never be passed stack arguments, add
    // a tail keyword.
    if (IsAlwaysTail(Class)) {
      Changed = true;
      DEBUG(dbgs() << "Adding tail keyword to function since it can never be "
                      "passed stack args: " << *Inst << "\n");
      cast<CallInst>(Inst)->setTailCall();
    }

    // Ensure that functions that can never have a "tail" keyword due to the
    // semantics of ARC truly do not do so.
    if (IsNeverTail(Class)) {
      Changed = true;
      DEBUG(dbgs() << "Removing tail keyword from function: " << *Inst <<
            "\n");
      cast<CallInst>(Inst)->setTailCall(false);
    }

    // Set nounwind as needed.
    if (IsNoThrow(Class)) {
      Changed = true;
      DEBUG(dbgs() << "Found no throw class. Setting nounwind on: " << *Inst
                   << "\n");
      cast<CallInst>(Inst)->setDoesNotThrow();
    }

    if (!IsNoopOnNull(Class)) {
      UsedInThisFunction |= 1 << Class;
      continue;
    }

    const Value *Arg = GetObjCArg(Inst);

    // ARC calls with null are no-ops. Delete them.
    if (IsNullOrUndef(Arg)) {
      Changed = true;
      ++NumNoops;
      DEBUG(dbgs() << "ARC calls with  null are no-ops. Erasing: " << *Inst
            << "\n");
      EraseInstruction(Inst);
      continue;
    }

    // Keep track of which of retain, release, autorelease, and retain_block
    // are actually present in this function.
    UsedInThisFunction |= 1 << Class;

    // If Arg is a PHI, and one or more incoming values to the
    // PHI are null, and the call is control-equivalent to the PHI, and there
    // are no relevant side effects between the PHI and the call, the call
    // could be pushed up to just those paths with non-null incoming values.
    // For now, don't bother splitting critical edges for this.
    SmallVector<std::pair<Instruction *, const Value *>, 4> Worklist;
    Worklist.push_back(std::make_pair(Inst, Arg));
    do {
      std::pair<Instruction *, const Value *> Pair = Worklist.pop_back_val();
      Inst = Pair.first;
      Arg = Pair.second;

      const PHINode *PN = dyn_cast<PHINode>(Arg);
      if (!PN) continue;

      // Determine if the PHI has any null operands, or any incoming
      // critical edges.
      bool HasNull = false;
      bool HasCriticalEdges = false;
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
        Value *Incoming =
          StripPointerCastsAndObjCCalls(PN->getIncomingValue(i));
        if (IsNullOrUndef(Incoming))
          HasNull = true;
        else if (cast<TerminatorInst>(PN->getIncomingBlock(i)->back())
                   .getNumSuccessors() != 1) {
          HasCriticalEdges = true;
          break;
        }
      }
      // If we have null operands and no critical edges, optimize.
      if (!HasCriticalEdges && HasNull) {
        SmallPtrSet<Instruction *, 4> DependingInstructions;
        SmallPtrSet<const BasicBlock *, 4> Visited;

        // Check that there is nothing that cares about the reference
        // count between the call and the phi.
        switch (Class) {
        case IC_Retain:
        case IC_RetainBlock:
          // These can always be moved up.
          break;
        case IC_Release:
          // These can't be moved across things that care about the retain
          // count.
          FindDependencies(NeedsPositiveRetainCount, Arg,
                           Inst->getParent(), Inst,
                           DependingInstructions, Visited, PA);
          break;
        case IC_Autorelease:
          // These can't be moved across autorelease pool scope boundaries.
          FindDependencies(AutoreleasePoolBoundary, Arg,
                           Inst->getParent(), Inst,
                           DependingInstructions, Visited, PA);
          break;
        case IC_RetainRV:
        case IC_AutoreleaseRV:
          // Don't move these; the RV optimization depends on the autoreleaseRV
          // being tail called, and the retainRV being immediately after a call
          // (which might still happen if we get lucky with codegen layout, but
          // it's not worth taking the chance).
          continue;
        default:
          llvm_unreachable("Invalid dependence flavor");
        }

        if (DependingInstructions.size() == 1 &&
            *DependingInstructions.begin() == PN) {
          Changed = true;
          ++NumPartialNoops;
          // Clone the call into each predecessor that has a non-null value.
          CallInst *CInst = cast<CallInst>(Inst);
          Type *ParamTy = CInst->getArgOperand(0)->getType();
          for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
            Value *Incoming =
              StripPointerCastsAndObjCCalls(PN->getIncomingValue(i));
            if (!IsNullOrUndef(Incoming)) {
              CallInst *Clone = cast<CallInst>(CInst->clone());
              Value *Op = PN->getIncomingValue(i);
              Instruction *InsertPos = &PN->getIncomingBlock(i)->back();
              if (Op->getType() != ParamTy)
                Op = new BitCastInst(Op, ParamTy, "", InsertPos);
              Clone->setArgOperand(0, Op);
              Clone->insertBefore(InsertPos);

              DEBUG(dbgs() << "Cloning "
                           << *CInst << "\n"
                           "And inserting clone at " << *InsertPos << "\n");
              Worklist.push_back(std::make_pair(Clone, Incoming));
            }
          }
          // Erase the original call.
          DEBUG(dbgs() << "Erasing: " << *CInst << "\n");
          EraseInstruction(CInst);
          continue;
        }
      }
    } while (!Worklist.empty());
  }
}

/// If we have a top down pointer in the S_Use state, make sure that there are
/// no CFG hazards by checking the states of various bottom up pointers.
static void CheckForUseCFGHazard(const Sequence SuccSSeq,
                                 const bool SuccSRRIKnownSafe,
                                 PtrState &S,
                                 bool &SomeSuccHasSame,
                                 bool &AllSuccsHaveSame,
                                 bool &NotAllSeqEqualButKnownSafe,
                                 bool &ShouldContinue) {
  switch (SuccSSeq) {
  case S_CanRelease: {
    if (!S.IsKnownSafe() && !SuccSRRIKnownSafe) {
      S.ClearSequenceProgress();
      break;
    }
    S.SetCFGHazardAfflicted(true);
    ShouldContinue = true;
    break;
  }
  case S_Use:
    SomeSuccHasSame = true;
    break;
  case S_Stop:
  case S_Release:
  case S_MovableRelease:
    if (!S.IsKnownSafe() && !SuccSRRIKnownSafe)
      AllSuccsHaveSame = false;
    else
      NotAllSeqEqualButKnownSafe = true;
    break;
  case S_Retain:
    llvm_unreachable("bottom-up pointer in retain state!");
  case S_None:
    llvm_unreachable("This should have been handled earlier.");
  }
}

/// If we have a Top Down pointer in the S_CanRelease state, make sure that
/// there are no CFG hazards by checking the states of various bottom up
/// pointers.
static void CheckForCanReleaseCFGHazard(const Sequence SuccSSeq,
                                        const bool SuccSRRIKnownSafe,
                                        PtrState &S,
                                        bool &SomeSuccHasSame,
                                        bool &AllSuccsHaveSame,
                                        bool &NotAllSeqEqualButKnownSafe) {
  switch (SuccSSeq) {
  case S_CanRelease:
    SomeSuccHasSame = true;
    break;
  case S_Stop:
  case S_Release:
  case S_MovableRelease:
  case S_Use:
    if (!S.IsKnownSafe() && !SuccSRRIKnownSafe)
      AllSuccsHaveSame = false;
    else
      NotAllSeqEqualButKnownSafe = true;
    break;
  case S_Retain:
    llvm_unreachable("bottom-up pointer in retain state!");
  case S_None:
    llvm_unreachable("This should have been handled earlier.");
  }
}

/// Check for critical edges, loop boundaries, irreducible control flow, or
/// other CFG structures where moving code across the edge would result in it
/// being executed more.
void
ObjCARCOpt::CheckForCFGHazards(const BasicBlock *BB,
                               DenseMap<const BasicBlock *, BBState> &BBStates,
                               BBState &MyStates) const {
  // If any top-down local-use or possible-dec has a succ which is earlier in
  // the sequence, forget it.
  for (BBState::ptr_iterator I = MyStates.top_down_ptr_begin(),
         E = MyStates.top_down_ptr_end(); I != E; ++I) {
    PtrState &S = I->second;
    const Sequence Seq = I->second.GetSeq();

    // We only care about S_Retain, S_CanRelease, and S_Use.
    if (Seq == S_None)
      continue;

    // Make sure that if extra top down states are added in the future that this
    // code is updated to handle it.
    assert((Seq == S_Retain || Seq == S_CanRelease || Seq == S_Use) &&
           "Unknown top down sequence state.");

    const Value *Arg = I->first;
    const TerminatorInst *TI = cast<TerminatorInst>(&BB->back());
    bool SomeSuccHasSame = false;
    bool AllSuccsHaveSame = true;
    bool NotAllSeqEqualButKnownSafe = false;

    succ_const_iterator SI(TI), SE(TI, false);

    for (; SI != SE; ++SI) {
      // If VisitBottomUp has pointer information for this successor, take
      // what we know about it.
      const DenseMap<const BasicBlock *, BBState>::iterator BBI =
        BBStates.find(*SI);
      assert(BBI != BBStates.end());
      const PtrState &SuccS = BBI->second.getPtrBottomUpState(Arg);
      const Sequence SuccSSeq = SuccS.GetSeq();

      // If bottom up, the pointer is in an S_None state, clear the sequence
      // progress since the sequence in the bottom up state finished
      // suggesting a mismatch in between retains/releases. This is true for
      // all three cases that we are handling here: S_Retain, S_Use, and
      // S_CanRelease.
      if (SuccSSeq == S_None) {
        S.ClearSequenceProgress();
        continue;
      }

      // If we have S_Use or S_CanRelease, perform our check for cfg hazard
      // checks.
      const bool SuccSRRIKnownSafe = SuccS.IsKnownSafe();

      // *NOTE* We do not use Seq from above here since we are allowing for
      // S.GetSeq() to change while we are visiting basic blocks.
      switch(S.GetSeq()) {
      case S_Use: {
        bool ShouldContinue = false;
        CheckForUseCFGHazard(SuccSSeq, SuccSRRIKnownSafe, S, SomeSuccHasSame,
                             AllSuccsHaveSame, NotAllSeqEqualButKnownSafe,
                             ShouldContinue);
        if (ShouldContinue)
          continue;
        break;
      }
      case S_CanRelease: {
        CheckForCanReleaseCFGHazard(SuccSSeq, SuccSRRIKnownSafe, S,
                                    SomeSuccHasSame, AllSuccsHaveSame,
                                    NotAllSeqEqualButKnownSafe);
        break;
      }
      case S_Retain:
      case S_None:
      case S_Stop:
      case S_Release:
      case S_MovableRelease:
        break;
      }
    }

    // If the state at the other end of any of the successor edges
    // matches the current state, require all edges to match. This
    // guards against loops in the middle of a sequence.
    if (SomeSuccHasSame && !AllSuccsHaveSame) {
      S.ClearSequenceProgress();
    } else if (NotAllSeqEqualButKnownSafe) {
      // If we would have cleared the state foregoing the fact that we are known
      // safe, stop code motion. This is because whether or not it is safe to
      // remove RR pairs via KnownSafe is an orthogonal concept to whether we
      // are allowed to perform code motion.
      S.SetCFGHazardAfflicted(true);
    }
  }
}

bool
ObjCARCOpt::VisitInstructionBottomUp(Instruction *Inst,
                                     BasicBlock *BB,
                                     MapVector<Value *, RRInfo> &Retains,
                                     BBState &MyStates) {
  bool NestingDetected = false;
  InstructionClass Class = GetInstructionClass(Inst);
  const Value *Arg = 0;

  DEBUG(dbgs() << "Class: " << Class << "\n");

  switch (Class) {
  case IC_Release: {
    Arg = GetObjCArg(Inst);

    PtrState &S = MyStates.getPtrBottomUpState(Arg);

    // If we see two releases in a row on the same pointer. If so, make
    // a note, and we'll cicle back to revisit it after we've
    // hopefully eliminated the second release, which may allow us to
    // eliminate the first release too.
    // Theoretically we could implement removal of nested retain+release
    // pairs by making PtrState hold a stack of states, but this is
    // simple and avoids adding overhead for the non-nested case.
    if (S.GetSeq() == S_Release || S.GetSeq() == S_MovableRelease) {
      DEBUG(dbgs() << "Found nested releases (i.e. a release pair)\n");
      NestingDetected = true;
    }

    MDNode *ReleaseMetadata = Inst->getMetadata(ImpreciseReleaseMDKind);
    Sequence NewSeq = ReleaseMetadata ? S_MovableRelease : S_Release;
    ANNOTATE_BOTTOMUP(Inst, Arg, S.GetSeq(), NewSeq);
    S.ResetSequenceProgress(NewSeq);
    S.SetReleaseMetadata(ReleaseMetadata);
    S.SetKnownSafe(S.HasKnownPositiveRefCount());
    S.SetTailCallRelease(cast<CallInst>(Inst)->isTailCall());
    S.InsertCall(Inst);
    S.SetKnownPositiveRefCount();
    break;
  }
  case IC_RetainBlock:
    // In OptimizeIndividualCalls, we have strength reduced all optimizable
    // objc_retainBlocks to objc_retains. Thus at this point any
    // objc_retainBlocks that we see are not optimizable.
    break;
  case IC_Retain:
  case IC_RetainRV: {
    Arg = GetObjCArg(Inst);

    PtrState &S = MyStates.getPtrBottomUpState(Arg);
    S.SetKnownPositiveRefCount();

    Sequence OldSeq = S.GetSeq();
    switch (OldSeq) {
    case S_Stop:
    case S_Release:
    case S_MovableRelease:
    case S_Use:
      // If OldSeq is not S_Use or OldSeq is S_Use and we are tracking an
      // imprecise release, clear our reverse insertion points.
      if (OldSeq != S_Use || S.IsTrackingImpreciseReleases())
        S.ClearReverseInsertPts();
      // FALL THROUGH
    case S_CanRelease:
      // Don't do retain+release tracking for IC_RetainRV, because it's
      // better to let it remain as the first instruction after a call.
      if (Class != IC_RetainRV)
        Retains[Inst] = S.GetRRInfo();
      S.ClearSequenceProgress();
      break;
    case S_None:
      break;
    case S_Retain:
      llvm_unreachable("bottom-up pointer in retain state!");
    }
    ANNOTATE_BOTTOMUP(Inst, Arg, OldSeq, S.GetSeq());
    // A retain moving bottom up can be a use.
    break;
  }
  case IC_AutoreleasepoolPop:
    // Conservatively, clear MyStates for all known pointers.
    MyStates.clearBottomUpPointers();
    return NestingDetected;
  case IC_AutoreleasepoolPush:
  case IC_None:
    // These are irrelevant.
    return NestingDetected;
  case IC_User:
    // If we have a store into an alloca of a pointer we are tracking, the
    // pointer has multiple owners implying that we must be more conservative.
    //
    // This comes up in the context of a pointer being ``KnownSafe''. In the
    // presense of a block being initialized, the frontend will emit the
    // objc_retain on the original pointer and the release on the pointer loaded
    // from the alloca. The optimizer will through the provenance analysis
    // realize that the two are related, but since we only require KnownSafe in
    // one direction, will match the inner retain on the original pointer with
    // the guard release on the original pointer. This is fixed by ensuring that
    // in the presense of allocas we only unconditionally remove pointers if
    // both our retain and our release are KnownSafe.
    if (StoreInst *SI = dyn_cast<StoreInst>(Inst)) {
      if (AreAnyUnderlyingObjectsAnAlloca(SI->getPointerOperand())) {
        BBState::ptr_iterator I = MyStates.findPtrBottomUpState(
          StripPointerCastsAndObjCCalls(SI->getValueOperand()));
        if (I != MyStates.bottom_up_ptr_end())
          MultiOwnersSet.insert(I->first);
      }
    }
    break;
  default:
    break;
  }

  // Consider any other possible effects of this instruction on each
  // pointer being tracked.
  for (BBState::ptr_iterator MI = MyStates.bottom_up_ptr_begin(),
       ME = MyStates.bottom_up_ptr_end(); MI != ME; ++MI) {
    const Value *Ptr = MI->first;
    if (Ptr == Arg)
      continue; // Handled above.
    PtrState &S = MI->second;
    Sequence Seq = S.GetSeq();

    // Check for possible releases.
    if (CanAlterRefCount(Inst, Ptr, PA, Class)) {
      DEBUG(dbgs() << "CanAlterRefCount: Seq: " << Seq << "; " << *Ptr
            << "\n");
      S.ClearKnownPositiveRefCount();
      switch (Seq) {
      case S_Use:
        S.SetSeq(S_CanRelease);
        ANNOTATE_BOTTOMUP(Inst, Ptr, Seq, S.GetSeq());
        continue;
      case S_CanRelease:
      case S_Release:
      case S_MovableRelease:
      case S_Stop:
      case S_None:
        break;
      case S_Retain:
        llvm_unreachable("bottom-up pointer in retain state!");
      }
    }

    // Check for possible direct uses.
    switch (Seq) {
    case S_Release:
    case S_MovableRelease:
      if (CanUse(Inst, Ptr, PA, Class)) {
        DEBUG(dbgs() << "CanUse: Seq: " << Seq << "; " << *Ptr
              << "\n");
        assert(!S.HasReverseInsertPts());
        // If this is an invoke instruction, we're scanning it as part of
        // one of its successor blocks, since we can't insert code after it
        // in its own block, and we don't want to split critical edges.
        if (isa<InvokeInst>(Inst))
          S.InsertReverseInsertPt(BB->getFirstInsertionPt());
        else
          S.InsertReverseInsertPt(llvm::next(BasicBlock::iterator(Inst)));
        S.SetSeq(S_Use);
        ANNOTATE_BOTTOMUP(Inst, Ptr, Seq, S_Use);
      } else if (Seq == S_Release && IsUser(Class)) {
        DEBUG(dbgs() << "PreciseReleaseUse: Seq: " << Seq << "; " << *Ptr
              << "\n");
        // Non-movable releases depend on any possible objc pointer use.
        S.SetSeq(S_Stop);
        ANNOTATE_BOTTOMUP(Inst, Ptr, S_Release, S_Stop);
        assert(!S.HasReverseInsertPts());
        // As above; handle invoke specially.
        if (isa<InvokeInst>(Inst))
          S.InsertReverseInsertPt(BB->getFirstInsertionPt());
        else
          S.InsertReverseInsertPt(llvm::next(BasicBlock::iterator(Inst)));
      }
      break;
    case S_Stop:
      if (CanUse(Inst, Ptr, PA, Class)) {
        DEBUG(dbgs() << "PreciseStopUse: Seq: " << Seq << "; " << *Ptr
              << "\n");
        S.SetSeq(S_Use);
        ANNOTATE_BOTTOMUP(Inst, Ptr, Seq, S_Use);
      }
      break;
    case S_CanRelease:
    case S_Use:
    case S_None:
      break;
    case S_Retain:
      llvm_unreachable("bottom-up pointer in retain state!");
    }
  }

  return NestingDetected;
}

bool
ObjCARCOpt::VisitBottomUp(BasicBlock *BB,
                          DenseMap<const BasicBlock *, BBState> &BBStates,
                          MapVector<Value *, RRInfo> &Retains) {

  DEBUG(dbgs() << "\n== ObjCARCOpt::VisitBottomUp ==\n");

  bool NestingDetected = false;
  BBState &MyStates = BBStates[BB];

  // Merge the states from each successor to compute the initial state
  // for the current block.
  BBState::edge_iterator SI(MyStates.succ_begin()),
                         SE(MyStates.succ_end());
  if (SI != SE) {
    const BasicBlock *Succ = *SI;
    DenseMap<const BasicBlock *, BBState>::iterator I = BBStates.find(Succ);
    assert(I != BBStates.end());
    MyStates.InitFromSucc(I->second);
    ++SI;
    for (; SI != SE; ++SI) {
      Succ = *SI;
      I = BBStates.find(Succ);
      assert(I != BBStates.end());
      MyStates.MergeSucc(I->second);
    }
  }

  // If ARC Annotations are enabled, output the current state of pointers at the
  // bottom of the basic block.
  ANNOTATE_BOTTOMUP_BBEND(MyStates, BB);

  // Visit all the instructions, bottom-up.
  for (BasicBlock::iterator I = BB->end(), E = BB->begin(); I != E; --I) {
    Instruction *Inst = llvm::prior(I);

    // Invoke instructions are visited as part of their successors (below).
    if (isa<InvokeInst>(Inst))
      continue;

    DEBUG(dbgs() << "Visiting " << *Inst << "\n");

    NestingDetected |= VisitInstructionBottomUp(Inst, BB, Retains, MyStates);
  }

  // If there's a predecessor with an invoke, visit the invoke as if it were
  // part of this block, since we can't insert code after an invoke in its own
  // block, and we don't want to split critical edges.
  for (BBState::edge_iterator PI(MyStates.pred_begin()),
       PE(MyStates.pred_end()); PI != PE; ++PI) {
    BasicBlock *Pred = *PI;
    if (InvokeInst *II = dyn_cast<InvokeInst>(&Pred->back()))
      NestingDetected |= VisitInstructionBottomUp(II, BB, Retains, MyStates);
  }

  // If ARC Annotations are enabled, output the current state of pointers at the
  // top of the basic block.
  ANNOTATE_BOTTOMUP_BBSTART(MyStates, BB);

  return NestingDetected;
}

bool
ObjCARCOpt::VisitInstructionTopDown(Instruction *Inst,
                                    DenseMap<Value *, RRInfo> &Releases,
                                    BBState &MyStates) {
  bool NestingDetected = false;
  InstructionClass Class = GetInstructionClass(Inst);
  const Value *Arg = 0;

  switch (Class) {
  case IC_RetainBlock:
    // In OptimizeIndividualCalls, we have strength reduced all optimizable
    // objc_retainBlocks to objc_retains. Thus at this point any
    // objc_retainBlocks that we see are not optimizable.
    break;
  case IC_Retain:
  case IC_RetainRV: {
    Arg = GetObjCArg(Inst);

    PtrState &S = MyStates.getPtrTopDownState(Arg);

    // Don't do retain+release tracking for IC_RetainRV, because it's
    // better to let it remain as the first instruction after a call.
    if (Class != IC_RetainRV) {
      // If we see two retains in a row on the same pointer. If so, make
      // a note, and we'll cicle back to revisit it after we've
      // hopefully eliminated the second retain, which may allow us to
      // eliminate the first retain too.
      // Theoretically we could implement removal of nested retain+release
      // pairs by making PtrState hold a stack of states, but this is
      // simple and avoids adding overhead for the non-nested case.
      if (S.GetSeq() == S_Retain)
        NestingDetected = true;

      ANNOTATE_TOPDOWN(Inst, Arg, S.GetSeq(), S_Retain);
      S.ResetSequenceProgress(S_Retain);
      S.SetKnownSafe(S.HasKnownPositiveRefCount());
      S.InsertCall(Inst);
    }

    S.SetKnownPositiveRefCount();

    // A retain can be a potential use; procede to the generic checking
    // code below.
    break;
  }
  case IC_Release: {
    Arg = GetObjCArg(Inst);

    PtrState &S = MyStates.getPtrTopDownState(Arg);
    S.ClearKnownPositiveRefCount();

    Sequence OldSeq = S.GetSeq();

    MDNode *ReleaseMetadata = Inst->getMetadata(ImpreciseReleaseMDKind);

    switch (OldSeq) {
    case S_Retain:
    case S_CanRelease:
      if (OldSeq == S_Retain || ReleaseMetadata != 0)
        S.ClearReverseInsertPts();
      // FALL THROUGH
    case S_Use:
      S.SetReleaseMetadata(ReleaseMetadata);
      S.SetTailCallRelease(cast<CallInst>(Inst)->isTailCall());
      Releases[Inst] = S.GetRRInfo();
      ANNOTATE_TOPDOWN(Inst, Arg, S.GetSeq(), S_None);
      S.ClearSequenceProgress();
      break;
    case S_None:
      break;
    case S_Stop:
    case S_Release:
    case S_MovableRelease:
      llvm_unreachable("top-down pointer in release state!");
    }
    break;
  }
  case IC_AutoreleasepoolPop:
    // Conservatively, clear MyStates for all known pointers.
    MyStates.clearTopDownPointers();
    return NestingDetected;
  case IC_AutoreleasepoolPush:
  case IC_None:
    // These are irrelevant.
    return NestingDetected;
  default:
    break;
  }

  // Consider any other possible effects of this instruction on each
  // pointer being tracked.
  for (BBState::ptr_iterator MI = MyStates.top_down_ptr_begin(),
       ME = MyStates.top_down_ptr_end(); MI != ME; ++MI) {
    const Value *Ptr = MI->first;
    if (Ptr == Arg)
      continue; // Handled above.
    PtrState &S = MI->second;
    Sequence Seq = S.GetSeq();

    // Check for possible releases.
    if (CanAlterRefCount(Inst, Ptr, PA, Class)) {
      DEBUG(dbgs() << "CanAlterRefCount: Seq: " << Seq << "; " << *Ptr
            << "\n");
      S.ClearKnownPositiveRefCount();
      switch (Seq) {
      case S_Retain:
        S.SetSeq(S_CanRelease);
        ANNOTATE_TOPDOWN(Inst, Ptr, Seq, S_CanRelease);
        assert(!S.HasReverseInsertPts());
        S.InsertReverseInsertPt(Inst);

        // One call can't cause a transition from S_Retain to S_CanRelease
        // and S_CanRelease to S_Use. If we've made the first transition,
        // we're done.
        continue;
      case S_Use:
      case S_CanRelease:
      case S_None:
        break;
      case S_Stop:
      case S_Release:
      case S_MovableRelease:
        llvm_unreachable("top-down pointer in release state!");
      }
    }

    // Check for possible direct uses.
    switch (Seq) {
    case S_CanRelease:
      if (CanUse(Inst, Ptr, PA, Class)) {
        DEBUG(dbgs() << "CanUse: Seq: " << Seq << "; " << *Ptr
              << "\n");
        S.SetSeq(S_Use);
        ANNOTATE_TOPDOWN(Inst, Ptr, Seq, S_Use);
      }
      break;
    case S_Retain:
    case S_Use:
    case S_None:
      break;
    case S_Stop:
    case S_Release:
    case S_MovableRelease:
      llvm_unreachable("top-down pointer in release state!");
    }
  }

  return NestingDetected;
}

bool
ObjCARCOpt::VisitTopDown(BasicBlock *BB,
                         DenseMap<const BasicBlock *, BBState> &BBStates,
                         DenseMap<Value *, RRInfo> &Releases) {
  DEBUG(dbgs() << "\n== ObjCARCOpt::VisitTopDown ==\n");
  bool NestingDetected = false;
  BBState &MyStates = BBStates[BB];

  // Merge the states from each predecessor to compute the initial state
  // for the current block.
  BBState::edge_iterator PI(MyStates.pred_begin()),
                         PE(MyStates.pred_end());
  if (PI != PE) {
    const BasicBlock *Pred = *PI;
    DenseMap<const BasicBlock *, BBState>::iterator I = BBStates.find(Pred);
    assert(I != BBStates.end());
    MyStates.InitFromPred(I->second);
    ++PI;
    for (; PI != PE; ++PI) {
      Pred = *PI;
      I = BBStates.find(Pred);
      assert(I != BBStates.end());
      MyStates.MergePred(I->second);
    }
  }

  // If ARC Annotations are enabled, output the current state of pointers at the
  // top of the basic block.
  ANNOTATE_TOPDOWN_BBSTART(MyStates, BB);

  // Visit all the instructions, top-down.
  for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
    Instruction *Inst = I;

    DEBUG(dbgs() << "Visiting " << *Inst << "\n");

    NestingDetected |= VisitInstructionTopDown(Inst, Releases, MyStates);
  }

  // If ARC Annotations are enabled, output the current state of pointers at the
  // bottom of the basic block.
  ANNOTATE_TOPDOWN_BBEND(MyStates, BB);

#ifdef ARC_ANNOTATIONS
  if (!(EnableARCAnnotations && DisableCheckForCFGHazards))
#endif
  CheckForCFGHazards(BB, BBStates, MyStates);
  return NestingDetected;
}

static void
ComputePostOrders(Function &F,
                  SmallVectorImpl<BasicBlock *> &PostOrder,
                  SmallVectorImpl<BasicBlock *> &ReverseCFGPostOrder,
                  unsigned NoObjCARCExceptionsMDKind,
                  DenseMap<const BasicBlock *, BBState> &BBStates) {
  /// The visited set, for doing DFS walks.
  SmallPtrSet<BasicBlock *, 16> Visited;

  // Do DFS, computing the PostOrder.
  SmallPtrSet<BasicBlock *, 16> OnStack;
  SmallVector<std::pair<BasicBlock *, succ_iterator>, 16> SuccStack;

  // Functions always have exactly one entry block, and we don't have
  // any other block that we treat like an entry block.
  BasicBlock *EntryBB = &F.getEntryBlock();
  BBState &MyStates = BBStates[EntryBB];
  MyStates.SetAsEntry();
  TerminatorInst *EntryTI = cast<TerminatorInst>(&EntryBB->back());
  SuccStack.push_back(std::make_pair(EntryBB, succ_iterator(EntryTI)));
  Visited.insert(EntryBB);
  OnStack.insert(EntryBB);
  do {
  dfs_next_succ:
    BasicBlock *CurrBB = SuccStack.back().first;
    TerminatorInst *TI = cast<TerminatorInst>(&CurrBB->back());
    succ_iterator SE(TI, false);

    while (SuccStack.back().second != SE) {
      BasicBlock *SuccBB = *SuccStack.back().second++;
      if (Visited.insert(SuccBB)) {
        TerminatorInst *TI = cast<TerminatorInst>(&SuccBB->back());
        SuccStack.push_back(std::make_pair(SuccBB, succ_iterator(TI)));
        BBStates[CurrBB].addSucc(SuccBB);
        BBState &SuccStates = BBStates[SuccBB];
        SuccStates.addPred(CurrBB);
        OnStack.insert(SuccBB);
        goto dfs_next_succ;
      }

      if (!OnStack.count(SuccBB)) {
        BBStates[CurrBB].addSucc(SuccBB);
        BBStates[SuccBB].addPred(CurrBB);
      }
    }
    OnStack.erase(CurrBB);
    PostOrder.push_back(CurrBB);
    SuccStack.pop_back();
  } while (!SuccStack.empty());

  Visited.clear();

  // Do reverse-CFG DFS, computing the reverse-CFG PostOrder.
  // Functions may have many exits, and there also blocks which we treat
  // as exits due to ignored edges.
  SmallVector<std::pair<BasicBlock *, BBState::edge_iterator>, 16> PredStack;
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
    BasicBlock *ExitBB = I;
    BBState &MyStates = BBStates[ExitBB];
    if (!MyStates.isExit())
      continue;

    MyStates.SetAsExit();

    PredStack.push_back(std::make_pair(ExitBB, MyStates.pred_begin()));
    Visited.insert(ExitBB);
    while (!PredStack.empty()) {
    reverse_dfs_next_succ:
      BBState::edge_iterator PE = BBStates[PredStack.back().first].pred_end();
      while (PredStack.back().second != PE) {
        BasicBlock *BB = *PredStack.back().second++;
        if (Visited.insert(BB)) {
          PredStack.push_back(std::make_pair(BB, BBStates[BB].pred_begin()));
          goto reverse_dfs_next_succ;
        }
      }
      ReverseCFGPostOrder.push_back(PredStack.pop_back_val().first);
    }
  }
}

// Visit the function both top-down and bottom-up.
bool
ObjCARCOpt::Visit(Function &F,
                  DenseMap<const BasicBlock *, BBState> &BBStates,
                  MapVector<Value *, RRInfo> &Retains,
                  DenseMap<Value *, RRInfo> &Releases) {

  // Use reverse-postorder traversals, because we magically know that loops
  // will be well behaved, i.e. they won't repeatedly call retain on a single
  // pointer without doing a release. We can't use the ReversePostOrderTraversal
  // class here because we want the reverse-CFG postorder to consider each
  // function exit point, and we want to ignore selected cycle edges.
  SmallVector<BasicBlock *, 16> PostOrder;
  SmallVector<BasicBlock *, 16> ReverseCFGPostOrder;
  ComputePostOrders(F, PostOrder, ReverseCFGPostOrder,
                    NoObjCARCExceptionsMDKind,
                    BBStates);

  // Use reverse-postorder on the reverse CFG for bottom-up.
  bool BottomUpNestingDetected = false;
  for (SmallVectorImpl<BasicBlock *>::const_reverse_iterator I =
       ReverseCFGPostOrder.rbegin(), E = ReverseCFGPostOrder.rend();
       I != E; ++I)
    BottomUpNestingDetected |= VisitBottomUp(*I, BBStates, Retains);

  // Use reverse-postorder for top-down.
  bool TopDownNestingDetected = false;
  for (SmallVectorImpl<BasicBlock *>::const_reverse_iterator I =
       PostOrder.rbegin(), E = PostOrder.rend();
       I != E; ++I)
    TopDownNestingDetected |= VisitTopDown(*I, BBStates, Releases);

  return TopDownNestingDetected && BottomUpNestingDetected;
}

/// Move the calls in RetainsToMove and ReleasesToMove.
void ObjCARCOpt::MoveCalls(Value *Arg,
                           RRInfo &RetainsToMove,
                           RRInfo &ReleasesToMove,
                           MapVector<Value *, RRInfo> &Retains,
                           DenseMap<Value *, RRInfo> &Releases,
                           SmallVectorImpl<Instruction *> &DeadInsts,
                           Module *M) {
  Type *ArgTy = Arg->getType();
  Type *ParamTy = PointerType::getUnqual(Type::getInt8Ty(ArgTy->getContext()));

  DEBUG(dbgs() << "== ObjCARCOpt::MoveCalls ==\n");

  // Insert the new retain and release calls.
  for (SmallPtrSet<Instruction *, 2>::const_iterator
       PI = ReleasesToMove.ReverseInsertPts.begin(),
       PE = ReleasesToMove.ReverseInsertPts.end(); PI != PE; ++PI) {
    Instruction *InsertPt = *PI;
    Value *MyArg = ArgTy == ParamTy ? Arg :
                   new BitCastInst(Arg, ParamTy, "", InsertPt);
    Constant *Decl = EP.get(ARCRuntimeEntryPoints::EPT_Retain);
    CallInst *Call = CallInst::Create(Decl, MyArg, "", InsertPt);
    Call->setDoesNotThrow();
    Call->setTailCall();

    DEBUG(dbgs() << "Inserting new Retain: " << *Call << "\n"
                    "At insertion point: " << *InsertPt << "\n");
  }
  for (SmallPtrSet<Instruction *, 2>::const_iterator
       PI = RetainsToMove.ReverseInsertPts.begin(),
       PE = RetainsToMove.ReverseInsertPts.end(); PI != PE; ++PI) {
    Instruction *InsertPt = *PI;
    Value *MyArg = ArgTy == ParamTy ? Arg :
                   new BitCastInst(Arg, ParamTy, "", InsertPt);
    Constant *Decl = EP.get(ARCRuntimeEntryPoints::EPT_Release);
    CallInst *Call = CallInst::Create(Decl, MyArg, "", InsertPt);
    // Attach a clang.imprecise_release metadata tag, if appropriate.
    if (MDNode *M = ReleasesToMove.ReleaseMetadata)
      Call->setMetadata(ImpreciseReleaseMDKind, M);
    Call->setDoesNotThrow();
    if (ReleasesToMove.IsTailCallRelease)
      Call->setTailCall();

    DEBUG(dbgs() << "Inserting new Release: " << *Call << "\n"
                    "At insertion point: " << *InsertPt << "\n");
  }

  // Delete the original retain and release calls.
  for (SmallPtrSet<Instruction *, 2>::const_iterator
       AI = RetainsToMove.Calls.begin(),
       AE = RetainsToMove.Calls.end(); AI != AE; ++AI) {
    Instruction *OrigRetain = *AI;
    Retains.blot(OrigRetain);
    DeadInsts.push_back(OrigRetain);
    DEBUG(dbgs() << "Deleting retain: " << *OrigRetain << "\n");
  }
  for (SmallPtrSet<Instruction *, 2>::const_iterator
       AI = ReleasesToMove.Calls.begin(),
       AE = ReleasesToMove.Calls.end(); AI != AE; ++AI) {
    Instruction *OrigRelease = *AI;
    Releases.erase(OrigRelease);
    DeadInsts.push_back(OrigRelease);
    DEBUG(dbgs() << "Deleting release: " << *OrigRelease << "\n");
  }

}

bool
ObjCARCOpt::ConnectTDBUTraversals(DenseMap<const BasicBlock *, BBState>
                                    &BBStates,
                                  MapVector<Value *, RRInfo> &Retains,
                                  DenseMap<Value *, RRInfo> &Releases,
                                  Module *M,
                                  SmallVectorImpl<Instruction *> &NewRetains,
                                  SmallVectorImpl<Instruction *> &NewReleases,
                                  SmallVectorImpl<Instruction *> &DeadInsts,
                                  RRInfo &RetainsToMove,
                                  RRInfo &ReleasesToMove,
                                  Value *Arg,
                                  bool KnownSafe,
                                  bool &AnyPairsCompletelyEliminated) {
  // If a pair happens in a region where it is known that the reference count
  // is already incremented, we can similarly ignore possible decrements unless
  // we are dealing with a retainable object with multiple provenance sources.
  bool KnownSafeTD = true, KnownSafeBU = true;
  bool MultipleOwners = false;
  bool CFGHazardAfflicted = false;

  // Connect the dots between the top-down-collected RetainsToMove and
  // bottom-up-collected ReleasesToMove to form sets of related calls.
  // This is an iterative process so that we connect multiple releases
  // to multiple retains if needed.
  unsigned OldDelta = 0;
  unsigned NewDelta = 0;
  unsigned OldCount = 0;
  unsigned NewCount = 0;
  bool FirstRelease = true;
  for (;;) {
    for (SmallVectorImpl<Instruction *>::const_iterator
           NI = NewRetains.begin(), NE = NewRetains.end(); NI != NE; ++NI) {
      Instruction *NewRetain = *NI;
      MapVector<Value *, RRInfo>::const_iterator It = Retains.find(NewRetain);
      assert(It != Retains.end());
      const RRInfo &NewRetainRRI = It->second;
      KnownSafeTD &= NewRetainRRI.KnownSafe;
      MultipleOwners =
        MultipleOwners || MultiOwnersSet.count(GetObjCArg(NewRetain));
      for (SmallPtrSet<Instruction *, 2>::const_iterator
             LI = NewRetainRRI.Calls.begin(),
             LE = NewRetainRRI.Calls.end(); LI != LE; ++LI) {
        Instruction *NewRetainRelease = *LI;
        DenseMap<Value *, RRInfo>::const_iterator Jt =
          Releases.find(NewRetainRelease);
        if (Jt == Releases.end())
          return false;
        const RRInfo &NewRetainReleaseRRI = Jt->second;

        // If the release does not have a reference to the retain as well,
        // something happened which is unaccounted for. Do not do anything.
        //
        // This can happen if we catch an additive overflow during path count
        // merging.
        if (!NewRetainReleaseRRI.Calls.count(NewRetain))
          return false;

        if (ReleasesToMove.Calls.insert(NewRetainRelease)) {

          // If we overflow when we compute the path count, don't remove/move
          // anything.
          const BBState &NRRBBState = BBStates[NewRetainRelease->getParent()];
          unsigned PathCount = BBState::OverflowOccurredValue;
          if (NRRBBState.GetAllPathCountWithOverflow(PathCount))
            return false;
          assert(PathCount != BBState::OverflowOccurredValue &&
                 "PathCount at this point can not be "
                 "OverflowOccurredValue.");
          OldDelta -= PathCount;

          // Merge the ReleaseMetadata and IsTailCallRelease values.
          if (FirstRelease) {
            ReleasesToMove.ReleaseMetadata =
              NewRetainReleaseRRI.ReleaseMetadata;
            ReleasesToMove.IsTailCallRelease =
              NewRetainReleaseRRI.IsTailCallRelease;
            FirstRelease = false;
          } else {
            if (ReleasesToMove.ReleaseMetadata !=
                NewRetainReleaseRRI.ReleaseMetadata)
              ReleasesToMove.ReleaseMetadata = 0;
            if (ReleasesToMove.IsTailCallRelease !=
                NewRetainReleaseRRI.IsTailCallRelease)
              ReleasesToMove.IsTailCallRelease = false;
          }

          // Collect the optimal insertion points.
          if (!KnownSafe)
            for (SmallPtrSet<Instruction *, 2>::const_iterator
                   RI = NewRetainReleaseRRI.ReverseInsertPts.begin(),
                   RE = NewRetainReleaseRRI.ReverseInsertPts.end();
                 RI != RE; ++RI) {
              Instruction *RIP = *RI;
              if (ReleasesToMove.ReverseInsertPts.insert(RIP)) {
                // If we overflow when we compute the path count, don't
                // remove/move anything.
                const BBState &RIPBBState = BBStates[RIP->getParent()];
                PathCount = BBState::OverflowOccurredValue;
                if (RIPBBState.GetAllPathCountWithOverflow(PathCount))
                  return false;
                assert(PathCount != BBState::OverflowOccurredValue &&
                       "PathCount at this point can not be "
                       "OverflowOccurredValue.");
                NewDelta -= PathCount;
              }
            }
          NewReleases.push_back(NewRetainRelease);
        }
      }
    }
    NewRetains.clear();
    if (NewReleases.empty()) break;

    // Back the other way.
    for (SmallVectorImpl<Instruction *>::const_iterator
           NI = NewReleases.begin(), NE = NewReleases.end(); NI != NE; ++NI) {
      Instruction *NewRelease = *NI;
      DenseMap<Value *, RRInfo>::const_iterator It =
        Releases.find(NewRelease);
      assert(It != Releases.end());
      const RRInfo &NewReleaseRRI = It->second;
      KnownSafeBU &= NewReleaseRRI.KnownSafe;
      CFGHazardAfflicted |= NewReleaseRRI.CFGHazardAfflicted;
      for (SmallPtrSet<Instruction *, 2>::const_iterator
             LI = NewReleaseRRI.Calls.begin(),
             LE = NewReleaseRRI.Calls.end(); LI != LE; ++LI) {
        Instruction *NewReleaseRetain = *LI;
        MapVector<Value *, RRInfo>::const_iterator Jt =
          Retains.find(NewReleaseRetain);
        if (Jt == Retains.end())
          return false;
        const RRInfo &NewReleaseRetainRRI = Jt->second;

        // If the retain does not have a reference to the release as well,
        // something happened which is unaccounted for. Do not do anything.
        //
        // This can happen if we catch an additive overflow during path count
        // merging.
        if (!NewReleaseRetainRRI.Calls.count(NewRelease))
          return false;

        if (RetainsToMove.Calls.insert(NewReleaseRetain)) {
          // If we overflow when we compute the path count, don't remove/move
          // anything.
          const BBState &NRRBBState = BBStates[NewReleaseRetain->getParent()];
          unsigned PathCount = BBState::OverflowOccurredValue;
          if (NRRBBState.GetAllPathCountWithOverflow(PathCount))
            return false;
          assert(PathCount != BBState::OverflowOccurredValue &&
                 "PathCount at this point can not be "
                 "OverflowOccurredValue.");
          OldDelta += PathCount;
          OldCount += PathCount;

          // Collect the optimal insertion points.
          if (!KnownSafe)
            for (SmallPtrSet<Instruction *, 2>::const_iterator
                   RI = NewReleaseRetainRRI.ReverseInsertPts.begin(),
                   RE = NewReleaseRetainRRI.ReverseInsertPts.end();
                 RI != RE; ++RI) {
              Instruction *RIP = *RI;
              if (RetainsToMove.ReverseInsertPts.insert(RIP)) {
                // If we overflow when we compute the path count, don't
                // remove/move anything.
                const BBState &RIPBBState = BBStates[RIP->getParent()];

                PathCount = BBState::OverflowOccurredValue;
                if (RIPBBState.GetAllPathCountWithOverflow(PathCount))
                  return false;
                assert(PathCount != BBState::OverflowOccurredValue &&
                       "PathCount at this point can not be "
                       "OverflowOccurredValue.");
                NewDelta += PathCount;
                NewCount += PathCount;
              }
            }
          NewRetains.push_back(NewReleaseRetain);
        }
      }
    }
    NewReleases.clear();
    if (NewRetains.empty()) break;
  }

  // If the pointer is known incremented in 1 direction and we do not have
  // MultipleOwners, we can safely remove the retain/releases. Otherwise we need
  // to be known safe in both directions.
  bool UnconditionallySafe = (KnownSafeTD && KnownSafeBU) ||
    ((KnownSafeTD || KnownSafeBU) && !MultipleOwners);
  if (UnconditionallySafe) {
    RetainsToMove.ReverseInsertPts.clear();
    ReleasesToMove.ReverseInsertPts.clear();
    NewCount = 0;
  } else {
    // Determine whether the new insertion points we computed preserve the
    // balance of retain and release calls through the program.
    // TODO: If the fully aggressive solution isn't valid, try to find a
    // less aggressive solution which is.
    if (NewDelta != 0)
      return false;

    // At this point, we are not going to remove any RR pairs, but we still are
    // able to move RR pairs. If one of our pointers is afflicted with
    // CFGHazards, we cannot perform such code motion so exit early.
    const bool WillPerformCodeMotion = RetainsToMove.ReverseInsertPts.size() ||
      ReleasesToMove.ReverseInsertPts.size();
    if (CFGHazardAfflicted && WillPerformCodeMotion)
      return false;
  }

  // Determine whether the original call points are balanced in the retain and
  // release calls through the program. If not, conservatively don't touch
  // them.
  // TODO: It's theoretically possible to do code motion in this case, as
  // long as the existing imbalances are maintained.
  if (OldDelta != 0)
    return false;

#ifdef ARC_ANNOTATIONS
  // Do not move calls if ARC annotations are requested.
  if (EnableARCAnnotations)
    return false;
#endif // ARC_ANNOTATIONS

  Changed = true;
  assert(OldCount != 0 && "Unreachable code?");
  NumRRs += OldCount - NewCount;
  // Set to true if we completely removed any RR pairs.
  AnyPairsCompletelyEliminated = NewCount == 0;

  // We can move calls!
  return true;
}

/// Identify pairings between the retains and releases, and delete and/or move
/// them.
bool
ObjCARCOpt::PerformCodePlacement(DenseMap<const BasicBlock *, BBState>
                                   &BBStates,
                                 MapVector<Value *, RRInfo> &Retains,
                                 DenseMap<Value *, RRInfo> &Releases,
                                 Module *M) {
  DEBUG(dbgs() << "\n== ObjCARCOpt::PerformCodePlacement ==\n");

  bool AnyPairsCompletelyEliminated = false;
  RRInfo RetainsToMove;
  RRInfo ReleasesToMove;
  SmallVector<Instruction *, 4> NewRetains;
  SmallVector<Instruction *, 4> NewReleases;
  SmallVector<Instruction *, 8> DeadInsts;

  // Visit each retain.
  for (MapVector<Value *, RRInfo>::const_iterator I = Retains.begin(),
       E = Retains.end(); I != E; ++I) {
    Value *V = I->first;
    if (!V) continue; // blotted

    Instruction *Retain = cast<Instruction>(V);

    DEBUG(dbgs() << "Visiting: " << *Retain << "\n");

    Value *Arg = GetObjCArg(Retain);

    // If the object being released is in static or stack storage, we know it's
    // not being managed by ObjC reference counting, so we can delete pairs
    // regardless of what possible decrements or uses lie between them.
    bool KnownSafe = isa<Constant>(Arg) || isa<AllocaInst>(Arg);

    // A constant pointer can't be pointing to an object on the heap. It may
    // be reference-counted, but it won't be deleted.
    if (const LoadInst *LI = dyn_cast<LoadInst>(Arg))
      if (const GlobalVariable *GV =
            dyn_cast<GlobalVariable>(
              StripPointerCastsAndObjCCalls(LI->getPointerOperand())))
        if (GV->isConstant())
          KnownSafe = true;

    // Connect the dots between the top-down-collected RetainsToMove and
    // bottom-up-collected ReleasesToMove to form sets of related calls.
    NewRetains.push_back(Retain);
    bool PerformMoveCalls =
      ConnectTDBUTraversals(BBStates, Retains, Releases, M, NewRetains,
                            NewReleases, DeadInsts, RetainsToMove,
                            ReleasesToMove, Arg, KnownSafe,
                            AnyPairsCompletelyEliminated);

    if (PerformMoveCalls) {
      // Ok, everything checks out and we're all set. Let's move/delete some
      // code!
      MoveCalls(Arg, RetainsToMove, ReleasesToMove,
                Retains, Releases, DeadInsts, M);
    }

    // Clean up state for next retain.
    NewReleases.clear();
    NewRetains.clear();
    RetainsToMove.clear();
    ReleasesToMove.clear();
  }

  // Now that we're done moving everything, we can delete the newly dead
  // instructions, as we no longer need them as insert points.
  while (!DeadInsts.empty())
    EraseInstruction(DeadInsts.pop_back_val());

  return AnyPairsCompletelyEliminated;
}

/// Weak pointer optimizations.
void ObjCARCOpt::OptimizeWeakCalls(Function &F) {
  DEBUG(dbgs() << "\n== ObjCARCOpt::OptimizeWeakCalls ==\n");

  // First, do memdep-style RLE and S2L optimizations. We can't use memdep
  // itself because it uses AliasAnalysis and we need to do provenance
  // queries instead.
  for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ) {
    Instruction *Inst = &*I++;

    DEBUG(dbgs() << "Visiting: " << *Inst << "\n");

    InstructionClass Class = GetBasicInstructionClass(Inst);
    if (Class != IC_LoadWeak && Class != IC_LoadWeakRetained)
      continue;

    // Delete objc_loadWeak calls with no users.
    if (Class == IC_LoadWeak && Inst->use_empty()) {
      Inst->eraseFromParent();
      continue;
    }

    // TODO: For now, just look for an earlier available version of this value
    // within the same block. Theoretically, we could do memdep-style non-local
    // analysis too, but that would want caching. A better approach would be to
    // use the technique that EarlyCSE uses.
    inst_iterator Current = llvm::prior(I);
    BasicBlock *CurrentBB = Current.getBasicBlockIterator();
    for (BasicBlock::iterator B = CurrentBB->begin(),
                              J = Current.getInstructionIterator();
         J != B; --J) {
      Instruction *EarlierInst = &*llvm::prior(J);
      InstructionClass EarlierClass = GetInstructionClass(EarlierInst);
      switch (EarlierClass) {
      case IC_LoadWeak:
      case IC_LoadWeakRetained: {
        // If this is loading from the same pointer, replace this load's value
        // with that one.
        CallInst *Call = cast<CallInst>(Inst);
        CallInst *EarlierCall = cast<CallInst>(EarlierInst);
        Value *Arg = Call->getArgOperand(0);
        Value *EarlierArg = EarlierCall->getArgOperand(0);
        switch (PA.getAA()->alias(Arg, EarlierArg)) {
        case AliasAnalysis::MustAlias:
          Changed = true;
          // If the load has a builtin retain, insert a plain retain for it.
          if (Class == IC_LoadWeakRetained) {
            Constant *Decl = EP.get(ARCRuntimeEntryPoints::EPT_Retain);
            CallInst *CI = CallInst::Create(Decl, EarlierCall, "", Call);
            CI->setTailCall();
          }
          // Zap the fully redundant load.
          Call->replaceAllUsesWith(EarlierCall);
          Call->eraseFromParent();
          goto clobbered;
        case AliasAnalysis::MayAlias:
        case AliasAnalysis::PartialAlias:
          goto clobbered;
        case AliasAnalysis::NoAlias:
          break;
        }
        break;
      }
      case IC_StoreWeak:
      case IC_InitWeak: {
        // If this is storing to the same pointer and has the same size etc.
        // replace this load's value with the stored value.
        CallInst *Call = cast<CallInst>(Inst);
        CallInst *EarlierCall = cast<CallInst>(EarlierInst);
        Value *Arg = Call->getArgOperand(0);
        Value *EarlierArg = EarlierCall->getArgOperand(0);
        switch (PA.getAA()->alias(Arg, EarlierArg)) {
        case AliasAnalysis::MustAlias:
          Changed = true;
          // If the load has a builtin retain, insert a plain retain for it.
          if (Class == IC_LoadWeakRetained) {
            Constant *Decl = EP.get(ARCRuntimeEntryPoints::EPT_Retain);
            CallInst *CI = CallInst::Create(Decl, EarlierCall, "", Call);
            CI->setTailCall();
          }
          // Zap the fully redundant load.
          Call->replaceAllUsesWith(EarlierCall->getArgOperand(1));
          Call->eraseFromParent();
          goto clobbered;
        case AliasAnalysis::MayAlias:
        case AliasAnalysis::PartialAlias:
          goto clobbered;
        case AliasAnalysis::NoAlias:
          break;
        }
        break;
      }
      case IC_MoveWeak:
      case IC_CopyWeak:
        // TOOD: Grab the copied value.
        goto clobbered;
      case IC_AutoreleasepoolPush:
      case IC_None:
      case IC_IntrinsicUser:
      case IC_User:
        // Weak pointers are only modified through the weak entry points
        // (and arbitrary calls, which could call the weak entry points).
        break;
      default:
        // Anything else could modify the weak pointer.
        goto clobbered;
      }
    }
  clobbered:;
  }

  // Then, for each destroyWeak with an alloca operand, check to see if
  // the alloca and all its users can be zapped.
  for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ) {
    Instruction *Inst = &*I++;
    InstructionClass Class = GetBasicInstructionClass(Inst);
    if (Class != IC_DestroyWeak)
      continue;

    CallInst *Call = cast<CallInst>(Inst);
    Value *Arg = Call->getArgOperand(0);
    if (AllocaInst *Alloca = dyn_cast<AllocaInst>(Arg)) {
      for (Value::use_iterator UI = Alloca->use_begin(),
           UE = Alloca->use_end(); UI != UE; ++UI) {
        const Instruction *UserInst = cast<Instruction>(*UI);
        switch (GetBasicInstructionClass(UserInst)) {
        case IC_InitWeak:
        case IC_StoreWeak:
        case IC_DestroyWeak:
          continue;
        default:
          goto done;
        }
      }
      Changed = true;
      for (Value::use_iterator UI = Alloca->use_begin(),
           UE = Alloca->use_end(); UI != UE; ) {
        CallInst *UserInst = cast<CallInst>(*UI++);
        switch (GetBasicInstructionClass(UserInst)) {
        case IC_InitWeak:
        case IC_StoreWeak:
          // These functions return their second argument.
          UserInst->replaceAllUsesWith(UserInst->getArgOperand(1));
          break;
        case IC_DestroyWeak:
          // No return value.
          break;
        default:
          llvm_unreachable("alloca really is used!");
        }
        UserInst->eraseFromParent();
      }
      Alloca->eraseFromParent();
    done:;
    }
  }
}

/// Identify program paths which execute sequences of retains and releases which
/// can be eliminated.
bool ObjCARCOpt::OptimizeSequences(Function &F) {
  // Releases, Retains - These are used to store the results of the main flow
  // analysis. These use Value* as the key instead of Instruction* so that the
  // map stays valid when we get around to rewriting code and calls get
  // replaced by arguments.
  DenseMap<Value *, RRInfo> Releases;
  MapVector<Value *, RRInfo> Retains;

  // This is used during the traversal of the function to track the
  // states for each identified object at each block.
  DenseMap<const BasicBlock *, BBState> BBStates;

  // Analyze the CFG of the function, and all instructions.
  bool NestingDetected = Visit(F, BBStates, Retains, Releases);

  // Transform.
  bool AnyPairsCompletelyEliminated = PerformCodePlacement(BBStates, Retains,
                                                           Releases,
                                                           F.getParent());

  // Cleanup.
  MultiOwnersSet.clear();

  return AnyPairsCompletelyEliminated && NestingDetected;
}

/// Check if there is a dependent call earlier that does not have anything in
/// between the Retain and the call that can affect the reference count of their
/// shared pointer argument. Note that Retain need not be in BB.
static bool
HasSafePathToPredecessorCall(const Value *Arg, Instruction *Retain,
                             SmallPtrSet<Instruction *, 4> &DepInsts,
                             SmallPtrSet<const BasicBlock *, 4> &Visited,
                             ProvenanceAnalysis &PA) {
  FindDependencies(CanChangeRetainCount, Arg, Retain->getParent(), Retain,
                   DepInsts, Visited, PA);
  if (DepInsts.size() != 1)
    return false;

  CallInst *Call =
    dyn_cast_or_null<CallInst>(*DepInsts.begin());

  // Check that the pointer is the return value of the call.
  if (!Call || Arg != Call)
    return false;

  // Check that the call is a regular call.
  InstructionClass Class = GetBasicInstructionClass(Call);
  if (Class != IC_CallOrUser && Class != IC_Call)
    return false;

  return true;
}

/// Find a dependent retain that precedes the given autorelease for which there
/// is nothing in between the two instructions that can affect the ref count of
/// Arg.
static CallInst *
FindPredecessorRetainWithSafePath(const Value *Arg, BasicBlock *BB,
                                  Instruction *Autorelease,
                                  SmallPtrSet<Instruction *, 4> &DepInsts,
                                  SmallPtrSet<const BasicBlock *, 4> &Visited,
                                  ProvenanceAnalysis &PA) {
  FindDependencies(CanChangeRetainCount, Arg,
                   BB, Autorelease, DepInsts, Visited, PA);
  if (DepInsts.size() != 1)
    return 0;

  CallInst *Retain =
    dyn_cast_or_null<CallInst>(*DepInsts.begin());

  // Check that we found a retain with the same argument.
  if (!Retain ||
      !IsRetain(GetBasicInstructionClass(Retain)) ||
      GetObjCArg(Retain) != Arg) {
    return 0;
  }

  return Retain;
}

/// Look for an ``autorelease'' instruction dependent on Arg such that there are
/// no instructions dependent on Arg that need a positive ref count in between
/// the autorelease and the ret.
static CallInst *
FindPredecessorAutoreleaseWithSafePath(const Value *Arg, BasicBlock *BB,
                                       ReturnInst *Ret,
                                       SmallPtrSet<Instruction *, 4> &DepInsts,
                                       SmallPtrSet<const BasicBlock *, 4> &V,
                                       ProvenanceAnalysis &PA) {
  FindDependencies(NeedsPositiveRetainCount, Arg,
                   BB, Ret, DepInsts, V, PA);
  if (DepInsts.size() != 1)
    return 0;

  CallInst *Autorelease =
    dyn_cast_or_null<CallInst>(*DepInsts.begin());
  if (!Autorelease)
    return 0;
  InstructionClass AutoreleaseClass = GetBasicInstructionClass(Autorelease);
  if (!IsAutorelease(AutoreleaseClass))
    return 0;
  if (GetObjCArg(Autorelease) != Arg)
    return 0;

  return Autorelease;
}

/// Look for this pattern:
/// \code
///    %call = call i8* @something(...)
///    %2 = call i8* @objc_retain(i8* %call)
///    %3 = call i8* @objc_autorelease(i8* %2)
///    ret i8* %3
/// \endcode
/// And delete the retain and autorelease.
void ObjCARCOpt::OptimizeReturns(Function &F) {
  if (!F.getReturnType()->isPointerTy())
    return;

  DEBUG(dbgs() << "\n== ObjCARCOpt::OptimizeReturns ==\n");

  SmallPtrSet<Instruction *, 4> DependingInstructions;
  SmallPtrSet<const BasicBlock *, 4> Visited;
  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI) {
    BasicBlock *BB = FI;
    ReturnInst *Ret = dyn_cast<ReturnInst>(&BB->back());

    DEBUG(dbgs() << "Visiting: " << *Ret << "\n");

    if (!Ret)
      continue;

    const Value *Arg = StripPointerCastsAndObjCCalls(Ret->getOperand(0));

    // Look for an ``autorelease'' instruction that is a predecessor of Ret and
    // dependent on Arg such that there are no instructions dependent on Arg
    // that need a positive ref count in between the autorelease and Ret.
    CallInst *Autorelease =
      FindPredecessorAutoreleaseWithSafePath(Arg, BB, Ret,
                                             DependingInstructions, Visited,
                                             PA);
    DependingInstructions.clear();
    Visited.clear();

    if (!Autorelease)
      continue;

    CallInst *Retain =
      FindPredecessorRetainWithSafePath(Arg, BB, Autorelease,
                                        DependingInstructions, Visited, PA);
    DependingInstructions.clear();
    Visited.clear();

    if (!Retain)
      continue;

    // Check that there is nothing that can affect the reference count
    // between the retain and the call.  Note that Retain need not be in BB.
    bool HasSafePathToCall = HasSafePathToPredecessorCall(Arg, Retain,
                                                          DependingInstructions,
                                                          Visited, PA);
    DependingInstructions.clear();
    Visited.clear();

    if (!HasSafePathToCall)
      continue;

    // If so, we can zap the retain and autorelease.
    Changed = true;
    ++NumRets;
    DEBUG(dbgs() << "Erasing: " << *Retain << "\nErasing: "
          << *Autorelease << "\n");
    EraseInstruction(Retain);
    EraseInstruction(Autorelease);
  }
}

#ifndef NDEBUG
void
ObjCARCOpt::GatherStatistics(Function &F, bool AfterOptimization) {
  llvm::Statistic &NumRetains =
    AfterOptimization? NumRetainsAfterOpt : NumRetainsBeforeOpt;
  llvm::Statistic &NumReleases =
    AfterOptimization? NumReleasesAfterOpt : NumReleasesBeforeOpt;

  for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ) {
    Instruction *Inst = &*I++;
    switch (GetBasicInstructionClass(Inst)) {
    default:
      break;
    case IC_Retain:
      ++NumRetains;
      break;
    case IC_Release:
      ++NumReleases;
      break;
    }
  }
}
#endif

bool ObjCARCOpt::doInitialization(Module &M) {
  if (!EnableARCOpts)
    return false;

  // If nothing in the Module uses ARC, don't do anything.
  Run = ModuleHasARC(M);
  if (!Run)
    return false;

  // Identify the imprecise release metadata kind.
  ImpreciseReleaseMDKind =
    M.getContext().getMDKindID("clang.imprecise_release");
  CopyOnEscapeMDKind =
    M.getContext().getMDKindID("clang.arc.copy_on_escape");
  NoObjCARCExceptionsMDKind =
    M.getContext().getMDKindID("clang.arc.no_objc_arc_exceptions");
#ifdef ARC_ANNOTATIONS
  ARCAnnotationBottomUpMDKind =
    M.getContext().getMDKindID("llvm.arc.annotation.bottomup");
  ARCAnnotationTopDownMDKind =
    M.getContext().getMDKindID("llvm.arc.annotation.topdown");
  ARCAnnotationProvenanceSourceMDKind =
    M.getContext().getMDKindID("llvm.arc.annotation.provenancesource");
#endif // ARC_ANNOTATIONS

  // Intuitively, objc_retain and others are nocapture, however in practice
  // they are not, because they return their argument value. And objc_release
  // calls finalizers which can have arbitrary side effects.

  // Initialize our runtime entry point cache.
  EP.Initialize(&M);

  return false;
}

bool ObjCARCOpt::runOnFunction(Function &F) {
  if (!EnableARCOpts)
    return false;

  // If nothing in the Module uses ARC, don't do anything.
  if (!Run)
    return false;

  Changed = false;

  DEBUG(dbgs() << "<<< ObjCARCOpt: Visiting Function: " << F.getName() << " >>>"
        "\n");

  PA.setAA(&getAnalysis<AliasAnalysis>());

#ifndef NDEBUG
  if (AreStatisticsEnabled()) {
    GatherStatistics(F, false);
  }
#endif

  // This pass performs several distinct transformations. As a compile-time aid
  // when compiling code that isn't ObjC, skip these if the relevant ObjC
  // library functions aren't declared.

  // Preliminary optimizations. This also computes UsedInThisFunction.
  OptimizeIndividualCalls(F);

  // Optimizations for weak pointers.
  if (UsedInThisFunction & ((1 << IC_LoadWeak) |
                            (1 << IC_LoadWeakRetained) |
                            (1 << IC_StoreWeak) |
                            (1 << IC_InitWeak) |
                            (1 << IC_CopyWeak) |
                            (1 << IC_MoveWeak) |
                            (1 << IC_DestroyWeak)))
    OptimizeWeakCalls(F);

  // Optimizations for retain+release pairs.
  if (UsedInThisFunction & ((1 << IC_Retain) |
                            (1 << IC_RetainRV) |
                            (1 << IC_RetainBlock)))
    if (UsedInThisFunction & (1 << IC_Release))
      // Run OptimizeSequences until it either stops making changes or
      // no retain+release pair nesting is detected.
      while (OptimizeSequences(F)) {}

  // Optimizations if objc_autorelease is used.
  if (UsedInThisFunction & ((1 << IC_Autorelease) |
                            (1 << IC_AutoreleaseRV)))
    OptimizeReturns(F);

  // Gather statistics after optimization.
#ifndef NDEBUG
  if (AreStatisticsEnabled()) {
    GatherStatistics(F, true);
  }
#endif

  DEBUG(dbgs() << "\n");

  return Changed;
}

void ObjCARCOpt::releaseMemory() {
  PA.clear();
}

/// @}
///
