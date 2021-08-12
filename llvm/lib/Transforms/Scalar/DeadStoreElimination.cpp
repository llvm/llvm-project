//===- DeadStoreElimination.cpp - MemorySSA Backed Dead Store Elimination -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The code below implements dead store elimination using MemorySSA. It uses
// the following general approach: given a MemoryDef, walk upwards to find
// clobbering MemoryDefs that may be killed by the starting def. Then check
// that there are no uses that may read the location of the original MemoryDef
// in between both MemoryDefs. A bit more concretely:
//
// For all MemoryDefs StartDef:
// 1. Get the next dominating clobbering MemoryDef (EarlierAccess) by walking
//    upwards.
// 2. Check that there are no reads between EarlierAccess and the StartDef by
//    checking all uses starting at EarlierAccess and walking until we see
//    StartDef.
// 3. For each found CurrentDef, check that:
//   1. There are no barrier instructions between CurrentDef and StartDef (like
//       throws or stores with ordering constraints).
//   2. StartDef is executed whenever CurrentDef is executed.
//   3. StartDef completely overwrites CurrentDef.
// 4. Erase CurrentDef from the function and MemorySSA.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/DeadStoreElimination.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Analysis/MustExecute.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugCounter.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/AssumeBundleBuilder.h"
#include "llvm/Transforms/Utils/BuildLibCalls.h"
#include "llvm/Transforms/Utils/Local.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <utility>

using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "dse"

STATISTIC(NumRemainingStores, "Number of stores remaining after DSE");
STATISTIC(NumRedundantStores, "Number of redundant stores deleted");
STATISTIC(NumFastStores, "Number of stores deleted");
STATISTIC(NumFastOther, "Number of other instrs removed");
STATISTIC(NumCompletePartials, "Number of stores dead by later partials");
STATISTIC(NumModifiedStores, "Number of stores modified");
STATISTIC(NumCFGChecks, "Number of stores modified");
STATISTIC(NumCFGTries, "Number of stores modified");
STATISTIC(NumCFGSuccess, "Number of stores modified");
STATISTIC(NumGetDomMemoryDefPassed,
          "Number of times a valid candidate is returned from getDomMemoryDef");
STATISTIC(NumDomMemDefChecks,
          "Number iterations check for reads in getDomMemoryDef");

DEBUG_COUNTER(MemorySSACounter, "dse-memoryssa",
              "Controls which MemoryDefs are eliminated.");

static cl::opt<bool>
EnablePartialOverwriteTracking("enable-dse-partial-overwrite-tracking",
  cl::init(true), cl::Hidden,
  cl::desc("Enable partial-overwrite tracking in DSE"));

static cl::opt<bool>
EnablePartialStoreMerging("enable-dse-partial-store-merging",
  cl::init(true), cl::Hidden,
  cl::desc("Enable partial store merging in DSE"));

static cl::opt<unsigned>
    MemorySSAScanLimit("dse-memoryssa-scanlimit", cl::init(150), cl::Hidden,
                       cl::desc("The number of memory instructions to scan for "
                                "dead store elimination (default = 100)"));
static cl::opt<unsigned> MemorySSAUpwardsStepLimit(
    "dse-memoryssa-walklimit", cl::init(90), cl::Hidden,
    cl::desc("The maximum number of steps while walking upwards to find "
             "MemoryDefs that may be killed (default = 90)"));

static cl::opt<unsigned> MemorySSAPartialStoreLimit(
    "dse-memoryssa-partial-store-limit", cl::init(5), cl::Hidden,
    cl::desc("The maximum number candidates that only partially overwrite the "
             "killing MemoryDef to consider"
             " (default = 5)"));

static cl::opt<unsigned> MemorySSADefsPerBlockLimit(
    "dse-memoryssa-defs-per-block-limit", cl::init(5000), cl::Hidden,
    cl::desc("The number of MemoryDefs we consider as candidates to eliminated "
             "other stores per basic block (default = 5000)"));

static cl::opt<unsigned> MemorySSASameBBStepCost(
    "dse-memoryssa-samebb-cost", cl::init(1), cl::Hidden,
    cl::desc(
        "The cost of a step in the same basic block as the killing MemoryDef"
        "(default = 1)"));

static cl::opt<unsigned>
    MemorySSAOtherBBStepCost("dse-memoryssa-otherbb-cost", cl::init(5),
                             cl::Hidden,
                             cl::desc("The cost of a step in a different basic "
                                      "block than the killing MemoryDef"
                                      "(default = 5)"));

static cl::opt<unsigned> MemorySSAPathCheckLimit(
    "dse-memoryssa-path-check-limit", cl::init(50), cl::Hidden,
    cl::desc("The maximum number of blocks to check when trying to prove that "
             "all paths to an exit go through a killing block (default = 50)"));

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//
using OverlapIntervalsTy = std::map<int64_t, int64_t>;
using InstOverlapIntervalsTy = DenseMap<Instruction *, OverlapIntervalsTy>;

/// Does this instruction write some memory?  This only returns true for things
/// that we can analyze with other helpers below.
static bool hasAnalyzableMemoryWrite(Instruction *I,
                                     const TargetLibraryInfo &TLI) {
  if (isa<StoreInst>(I))
    return true;
  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
    switch (II->getIntrinsicID()) {
    default:
      return false;
    case Intrinsic::memset:
    case Intrinsic::memmove:
    case Intrinsic::memcpy:
    case Intrinsic::memcpy_inline:
    case Intrinsic::memcpy_element_unordered_atomic:
    case Intrinsic::memmove_element_unordered_atomic:
    case Intrinsic::memset_element_unordered_atomic:
    case Intrinsic::init_trampoline:
    case Intrinsic::lifetime_end:
    case Intrinsic::masked_store:
      return true;
    }
  }
  if (auto *CB = dyn_cast<CallBase>(I)) {
    LibFunc LF;
    if (TLI.getLibFunc(*CB, LF) && TLI.has(LF)) {
      switch (LF) {
      case LibFunc_strcpy:
      case LibFunc_strncpy:
      case LibFunc_strcat:
      case LibFunc_strncat:
        return true;
      default:
        return false;
      }
    }
  }
  return false;
}

/// Return a Location stored to by the specified instruction. If isRemovable
/// returns true, this function and getLocForRead completely describe the memory
/// operations for this instruction.
static MemoryLocation getLocForWrite(Instruction *Inst,
                                     const TargetLibraryInfo &TLI) {
  if (StoreInst *SI = dyn_cast<StoreInst>(Inst))
    return MemoryLocation::get(SI);

  // memcpy/memmove/memset.
  if (auto *MI = dyn_cast<AnyMemIntrinsic>(Inst))
    return MemoryLocation::getForDest(MI);

  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(Inst)) {
    switch (II->getIntrinsicID()) {
    default:
      return MemoryLocation(); // Unhandled intrinsic.
    case Intrinsic::init_trampoline:
      return MemoryLocation::getAfter(II->getArgOperand(0));
    case Intrinsic::masked_store:
      return MemoryLocation::getForArgument(II, 1, TLI);
    case Intrinsic::lifetime_end: {
      uint64_t Len = cast<ConstantInt>(II->getArgOperand(0))->getZExtValue();
      return MemoryLocation(II->getArgOperand(1), Len);
    }
    }
  }
  if (auto *CB = dyn_cast<CallBase>(Inst))
    // All the supported TLI functions so far happen to have dest as their
    // first argument.
    return MemoryLocation::getAfter(CB->getArgOperand(0));
  return MemoryLocation();
}

/// If the value of this instruction and the memory it writes to is unused, may
/// we delete this instruction?
static bool isRemovable(Instruction *I) {
  // Don't remove volatile/atomic stores.
  if (StoreInst *SI = dyn_cast<StoreInst>(I))
    return SI->isUnordered();

  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
    switch (II->getIntrinsicID()) {
    default: llvm_unreachable("doesn't pass 'hasAnalyzableMemoryWrite' predicate");
    case Intrinsic::lifetime_end:
      // Never remove dead lifetime_end's, e.g. because it is followed by a
      // free.
      return false;
    case Intrinsic::init_trampoline:
      // Always safe to remove init_trampoline.
      return true;
    case Intrinsic::memset:
    case Intrinsic::memmove:
    case Intrinsic::memcpy:
    case Intrinsic::memcpy_inline:
      // Don't remove volatile memory intrinsics.
      return !cast<MemIntrinsic>(II)->isVolatile();
    case Intrinsic::memcpy_element_unordered_atomic:
    case Intrinsic::memmove_element_unordered_atomic:
    case Intrinsic::memset_element_unordered_atomic:
    case Intrinsic::masked_store:
      return true;
    }
  }

  // note: only get here for calls with analyzable writes - i.e. libcalls
  if (auto *CB = dyn_cast<CallBase>(I))
    return CB->use_empty();

  return false;
}

/// Returns true if the end of this instruction can be safely shortened in
/// length.
static bool isShortenableAtTheEnd(Instruction *I) {
  // Don't shorten stores for now
  if (isa<StoreInst>(I))
    return false;

  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
    switch (II->getIntrinsicID()) {
      default: return false;
      case Intrinsic::memset:
      case Intrinsic::memcpy:
      case Intrinsic::memcpy_element_unordered_atomic:
      case Intrinsic::memset_element_unordered_atomic:
        // Do shorten memory intrinsics.
        // FIXME: Add memmove if it's also safe to transform.
        return true;
    }
  }

  // Don't shorten libcalls calls for now.

  return false;
}

/// Returns true if the beginning of this instruction can be safely shortened
/// in length.
static bool isShortenableAtTheBeginning(Instruction *I) {
  // FIXME: Handle only memset for now. Supporting memcpy/memmove should be
  // easily done by offsetting the source address.
  return isa<AnyMemSetInst>(I);
}

static uint64_t getPointerSize(const Value *V, const DataLayout &DL,
                               const TargetLibraryInfo &TLI,
                               const Function *F) {
  uint64_t Size;
  ObjectSizeOpts Opts;
  Opts.NullIsUnknownSize = NullPointerIsDefined(F);

  if (getObjectSize(V, Size, DL, &TLI, Opts))
    return Size;
  return MemoryLocation::UnknownSize;
}

namespace {

enum OverwriteResult {
  OW_Begin,
  OW_Complete,
  OW_End,
  OW_PartialEarlierWithFullLater,
  OW_MaybePartial,
  OW_Unknown
};

} // end anonymous namespace

/// Check if two instruction are masked stores that completely
/// overwrite one another. More specifically, \p Later has to
/// overwrite \p Earlier.
static OverwriteResult isMaskedStoreOverwrite(const Instruction *Later,
                                              const Instruction *Earlier,
                                              BatchAAResults &AA) {
  const auto *IIL = dyn_cast<IntrinsicInst>(Later);
  const auto *IIE = dyn_cast<IntrinsicInst>(Earlier);
  if (IIL == nullptr || IIE == nullptr)
    return OW_Unknown;
  if (IIL->getIntrinsicID() != Intrinsic::masked_store ||
      IIE->getIntrinsicID() != Intrinsic::masked_store)
    return OW_Unknown;
  // Pointers.
  Value *LP = IIL->getArgOperand(1)->stripPointerCasts();
  Value *EP = IIE->getArgOperand(1)->stripPointerCasts();
  if (LP != EP && !AA.isMustAlias(LP, EP))
    return OW_Unknown;
  // Masks.
  // TODO: check that Later's mask is a superset of the Earlier's mask.
  if (IIL->getArgOperand(3) != IIE->getArgOperand(3))
    return OW_Unknown;
  return OW_Complete;
}

/// Return 'OW_Complete' if a store to the 'Later' location completely
/// overwrites a store to the 'Earlier' location, 'OW_End' if the end of the
/// 'Earlier' location is completely overwritten by 'Later', 'OW_Begin' if the
/// beginning of the 'Earlier' location is overwritten by 'Later'.
/// 'OW_PartialEarlierWithFullLater' means that an earlier (big) store was
/// overwritten by a latter (smaller) store which doesn't write outside the big
/// store's memory locations. Returns 'OW_Unknown' if nothing can be determined.
/// NOTE: This function must only be called if both \p Later and \p Earlier
/// write to the same underlying object with valid \p EarlierOff and \p
/// LaterOff.
static OverwriteResult isPartialOverwrite(const MemoryLocation &Later,
                                          const MemoryLocation &Earlier,
                                          int64_t EarlierOff, int64_t LaterOff,
                                          Instruction *DepWrite,
                                          InstOverlapIntervalsTy &IOL) {
  const uint64_t LaterSize = Later.Size.getValue();
  const uint64_t EarlierSize = Earlier.Size.getValue();
  // We may now overlap, although the overlap is not complete. There might also
  // be other incomplete overlaps, and together, they might cover the complete
  // earlier write.
  // Note: The correctness of this logic depends on the fact that this function
  // is not even called providing DepWrite when there are any intervening reads.
  if (EnablePartialOverwriteTracking &&
      LaterOff < int64_t(EarlierOff + EarlierSize) &&
      int64_t(LaterOff + LaterSize) >= EarlierOff) {

    // Insert our part of the overlap into the map.
    auto &IM = IOL[DepWrite];
    LLVM_DEBUG(dbgs() << "DSE: Partial overwrite: Earlier [" << EarlierOff
                      << ", " << int64_t(EarlierOff + EarlierSize)
                      << ") Later [" << LaterOff << ", "
                      << int64_t(LaterOff + LaterSize) << ")\n");

    // Make sure that we only insert non-overlapping intervals and combine
    // adjacent intervals. The intervals are stored in the map with the ending
    // offset as the key (in the half-open sense) and the starting offset as
    // the value.
    int64_t LaterIntStart = LaterOff, LaterIntEnd = LaterOff + LaterSize;

    // Find any intervals ending at, or after, LaterIntStart which start
    // before LaterIntEnd.
    auto ILI = IM.lower_bound(LaterIntStart);
    if (ILI != IM.end() && ILI->second <= LaterIntEnd) {
      // This existing interval is overlapped with the current store somewhere
      // in [LaterIntStart, LaterIntEnd]. Merge them by erasing the existing
      // intervals and adjusting our start and end.
      LaterIntStart = std::min(LaterIntStart, ILI->second);
      LaterIntEnd = std::max(LaterIntEnd, ILI->first);
      ILI = IM.erase(ILI);

      // Continue erasing and adjusting our end in case other previous
      // intervals are also overlapped with the current store.
      //
      // |--- ealier 1 ---|  |--- ealier 2 ---|
      //     |------- later---------|
      //
      while (ILI != IM.end() && ILI->second <= LaterIntEnd) {
        assert(ILI->second > LaterIntStart && "Unexpected interval");
        LaterIntEnd = std::max(LaterIntEnd, ILI->first);
        ILI = IM.erase(ILI);
      }
    }

    IM[LaterIntEnd] = LaterIntStart;

    ILI = IM.begin();
    if (ILI->second <= EarlierOff &&
        ILI->first >= int64_t(EarlierOff + EarlierSize)) {
      LLVM_DEBUG(dbgs() << "DSE: Full overwrite from partials: Earlier ["
                        << EarlierOff << ", "
                        << int64_t(EarlierOff + EarlierSize)
                        << ") Composite Later [" << ILI->second << ", "
                        << ILI->first << ")\n");
      ++NumCompletePartials;
      return OW_Complete;
    }
  }

  // Check for an earlier store which writes to all the memory locations that
  // the later store writes to.
  if (EnablePartialStoreMerging && LaterOff >= EarlierOff &&
      int64_t(EarlierOff + EarlierSize) > LaterOff &&
      uint64_t(LaterOff - EarlierOff) + LaterSize <= EarlierSize) {
    LLVM_DEBUG(dbgs() << "DSE: Partial overwrite an earlier load ["
                      << EarlierOff << ", "
                      << int64_t(EarlierOff + EarlierSize)
                      << ") by a later store [" << LaterOff << ", "
                      << int64_t(LaterOff + LaterSize) << ")\n");
    // TODO: Maybe come up with a better name?
    return OW_PartialEarlierWithFullLater;
  }

  // Another interesting case is if the later store overwrites the end of the
  // earlier store.
  //
  //      |--earlier--|
  //                |--   later   --|
  //
  // In this case we may want to trim the size of earlier to avoid generating
  // writes to addresses which will definitely be overwritten later
  if (!EnablePartialOverwriteTracking &&
      (LaterOff > EarlierOff && LaterOff < int64_t(EarlierOff + EarlierSize) &&
       int64_t(LaterOff + LaterSize) >= int64_t(EarlierOff + EarlierSize)))
    return OW_End;

  // Finally, we also need to check if the later store overwrites the beginning
  // of the earlier store.
  //
  //                |--earlier--|
  //      |--   later   --|
  //
  // In this case we may want to move the destination address and trim the size
  // of earlier to avoid generating writes to addresses which will definitely
  // be overwritten later.
  if (!EnablePartialOverwriteTracking &&
      (LaterOff <= EarlierOff && int64_t(LaterOff + LaterSize) > EarlierOff)) {
    assert(int64_t(LaterOff + LaterSize) < int64_t(EarlierOff + EarlierSize) &&
           "Expect to be handled as OW_Complete");
    return OW_Begin;
  }
  // Otherwise, they don't completely overlap.
  return OW_Unknown;
}

/// Returns true if the memory which is accessed by the second instruction is not
/// modified between the first and the second instruction.
/// Precondition: Second instruction must be dominated by the first
/// instruction.
static bool
memoryIsNotModifiedBetween(Instruction *FirstI, Instruction *SecondI,
                           BatchAAResults &AA, const DataLayout &DL,
                           DominatorTree *DT) {
  // Do a backwards scan through the CFG from SecondI to FirstI. Look for
  // instructions which can modify the memory location accessed by SecondI.
  //
  // While doing the walk keep track of the address to check. It might be
  // different in different basic blocks due to PHI translation.
  using BlockAddressPair = std::pair<BasicBlock *, PHITransAddr>;
  SmallVector<BlockAddressPair, 16> WorkList;
  // Keep track of the address we visited each block with. Bail out if we
  // visit a block with different addresses.
  DenseMap<BasicBlock *, Value *> Visited;

  BasicBlock::iterator FirstBBI(FirstI);
  ++FirstBBI;
  BasicBlock::iterator SecondBBI(SecondI);
  BasicBlock *FirstBB = FirstI->getParent();
  BasicBlock *SecondBB = SecondI->getParent();
  MemoryLocation MemLoc;
  if (auto *MemSet = dyn_cast<MemSetInst>(SecondI))
    MemLoc = MemoryLocation::getForDest(MemSet);
  else
    MemLoc = MemoryLocation::get(SecondI);

  auto *MemLocPtr = const_cast<Value *>(MemLoc.Ptr);

  // Start checking the SecondBB.
  WorkList.push_back(
      std::make_pair(SecondBB, PHITransAddr(MemLocPtr, DL, nullptr)));
  bool isFirstBlock = true;

  // Check all blocks going backward until we reach the FirstBB.
  while (!WorkList.empty()) {
    BlockAddressPair Current = WorkList.pop_back_val();
    BasicBlock *B = Current.first;
    PHITransAddr &Addr = Current.second;
    Value *Ptr = Addr.getAddr();

    // Ignore instructions before FirstI if this is the FirstBB.
    BasicBlock::iterator BI = (B == FirstBB ? FirstBBI : B->begin());

    BasicBlock::iterator EI;
    if (isFirstBlock) {
      // Ignore instructions after SecondI if this is the first visit of SecondBB.
      assert(B == SecondBB && "first block is not the store block");
      EI = SecondBBI;
      isFirstBlock = false;
    } else {
      // It's not SecondBB or (in case of a loop) the second visit of SecondBB.
      // In this case we also have to look at instructions after SecondI.
      EI = B->end();
    }
    for (; BI != EI; ++BI) {
      Instruction *I = &*BI;
      if (I->mayWriteToMemory() && I != SecondI)
        if (isModSet(AA.getModRefInfo(I, MemLoc.getWithNewPtr(Ptr))))
          return false;
    }
    if (B != FirstBB) {
      assert(B != &FirstBB->getParent()->getEntryBlock() &&
          "Should not hit the entry block because SI must be dominated by LI");
      for (BasicBlock *Pred : predecessors(B)) {
        PHITransAddr PredAddr = Addr;
        if (PredAddr.NeedsPHITranslationFromBlock(B)) {
          if (!PredAddr.IsPotentiallyPHITranslatable())
            return false;
          if (PredAddr.PHITranslateValue(B, Pred, DT, false))
            return false;
        }
        Value *TranslatedPtr = PredAddr.getAddr();
        auto Inserted = Visited.insert(std::make_pair(Pred, TranslatedPtr));
        if (!Inserted.second) {
          // We already visited this block before. If it was with a different
          // address - bail out!
          if (TranslatedPtr != Inserted.first->second)
            return false;
          // ... otherwise just skip it.
          continue;
        }
        WorkList.push_back(std::make_pair(Pred, PredAddr));
      }
    }
  }
  return true;
}

static bool tryToShorten(Instruction *EarlierWrite, int64_t &EarlierStart,
                         uint64_t &EarlierSize, int64_t LaterStart,
                         uint64_t LaterSize, bool IsOverwriteEnd) {
  auto *EarlierIntrinsic = cast<AnyMemIntrinsic>(EarlierWrite);
  Align PrefAlign = EarlierIntrinsic->getDestAlign().valueOrOne();

  // We assume that memet/memcpy operates in chunks of the "largest" native
  // type size and aligned on the same value. That means optimal start and size
  // of memset/memcpy should be modulo of preferred alignment of that type. That
  // is it there is no any sense in trying to reduce store size any further
  // since any "extra" stores comes for free anyway.
  // On the other hand, maximum alignment we can achieve is limited by alignment
  // of initial store.

  // TODO: Limit maximum alignment by preferred (or abi?) alignment of the
  // "largest" native type.
  // Note: What is the proper way to get that value?
  // Should TargetTransformInfo::getRegisterBitWidth be used or anything else?
  // PrefAlign = std::min(DL.getPrefTypeAlign(LargestType), PrefAlign);

  int64_t ToRemoveStart = 0;
  uint64_t ToRemoveSize = 0;
  // Compute start and size of the region to remove. Make sure 'PrefAlign' is
  // maintained on the remaining store.
  if (IsOverwriteEnd) {
    // Calculate required adjustment for 'LaterStart'in order to keep remaining
    // store size aligned on 'PerfAlign'.
    uint64_t Off =
        offsetToAlignment(uint64_t(LaterStart - EarlierStart), PrefAlign);
    ToRemoveStart = LaterStart + Off;
    if (EarlierSize <= uint64_t(ToRemoveStart - EarlierStart))
      return false;
    ToRemoveSize = EarlierSize - uint64_t(ToRemoveStart - EarlierStart);
  } else {
    ToRemoveStart = EarlierStart;
    assert(LaterSize >= uint64_t(EarlierStart - LaterStart) &&
           "Not overlapping accesses?");
    ToRemoveSize = LaterSize - uint64_t(EarlierStart - LaterStart);
    // Calculate required adjustment for 'ToRemoveSize'in order to keep
    // start of the remaining store aligned on 'PerfAlign'.
    uint64_t Off = offsetToAlignment(ToRemoveSize, PrefAlign);
    if (Off != 0) {
      if (ToRemoveSize <= (PrefAlign.value() - Off))
        return false;
      ToRemoveSize -= PrefAlign.value() - Off;
    }
    assert(isAligned(PrefAlign, ToRemoveSize) &&
           "Should preserve selected alignment");
  }

  assert(ToRemoveSize > 0 && "Shouldn't reach here if nothing to remove");
  assert(EarlierSize > ToRemoveSize && "Can't remove more than original size");

  uint64_t NewSize = EarlierSize - ToRemoveSize;
  if (auto *AMI = dyn_cast<AtomicMemIntrinsic>(EarlierWrite)) {
    // When shortening an atomic memory intrinsic, the newly shortened
    // length must remain an integer multiple of the element size.
    const uint32_t ElementSize = AMI->getElementSizeInBytes();
    if (0 != NewSize % ElementSize)
      return false;
  }

  LLVM_DEBUG(dbgs() << "DSE: Remove Dead Store:\n  OW "
                    << (IsOverwriteEnd ? "END" : "BEGIN") << ": "
                    << *EarlierWrite << "\n  KILLER [" << ToRemoveStart << ", "
                    << int64_t(ToRemoveStart + ToRemoveSize) << ")\n");

  Value *EarlierWriteLength = EarlierIntrinsic->getLength();
  Value *TrimmedLength =
      ConstantInt::get(EarlierWriteLength->getType(), NewSize);
  EarlierIntrinsic->setLength(TrimmedLength);
  EarlierIntrinsic->setDestAlignment(PrefAlign);

  if (!IsOverwriteEnd) {
    Value *OrigDest = EarlierIntrinsic->getRawDest();
    Type *Int8PtrTy =
        Type::getInt8PtrTy(EarlierIntrinsic->getContext(),
                           OrigDest->getType()->getPointerAddressSpace());
    Value *Dest = OrigDest;
    if (OrigDest->getType() != Int8PtrTy)
      Dest = CastInst::CreatePointerCast(OrigDest, Int8PtrTy, "", EarlierWrite);
    Value *Indices[1] = {
        ConstantInt::get(EarlierWriteLength->getType(), ToRemoveSize)};
    Instruction *NewDestGEP = GetElementPtrInst::CreateInBounds(
        Type::getInt8Ty(EarlierIntrinsic->getContext()),
        Dest, Indices, "", EarlierWrite);
    NewDestGEP->setDebugLoc(EarlierIntrinsic->getDebugLoc());
    if (NewDestGEP->getType() != OrigDest->getType())
      NewDestGEP = CastInst::CreatePointerCast(NewDestGEP, OrigDest->getType(),
                                               "", EarlierWrite);
    EarlierIntrinsic->setDest(NewDestGEP);
  }

  // Finally update start and size of earlier access.
  if (!IsOverwriteEnd)
    EarlierStart += ToRemoveSize;
  EarlierSize = NewSize;

  return true;
}

static bool tryToShortenEnd(Instruction *EarlierWrite,
                            OverlapIntervalsTy &IntervalMap,
                            int64_t &EarlierStart, uint64_t &EarlierSize) {
  if (IntervalMap.empty() || !isShortenableAtTheEnd(EarlierWrite))
    return false;

  OverlapIntervalsTy::iterator OII = --IntervalMap.end();
  int64_t LaterStart = OII->second;
  uint64_t LaterSize = OII->first - LaterStart;

  assert(OII->first - LaterStart >= 0 && "Size expected to be positive");

  if (LaterStart > EarlierStart &&
      // Note: "LaterStart - EarlierStart" is known to be positive due to
      // preceding check.
      (uint64_t)(LaterStart - EarlierStart) < EarlierSize &&
      // Note: "EarlierSize - (uint64_t)(LaterStart - EarlierStart)" is known to
      // be non negative due to preceding checks.
      LaterSize >= EarlierSize - (uint64_t)(LaterStart - EarlierStart)) {
    if (tryToShorten(EarlierWrite, EarlierStart, EarlierSize, LaterStart,
                     LaterSize, true)) {
      IntervalMap.erase(OII);
      return true;
    }
  }
  return false;
}

static bool tryToShortenBegin(Instruction *EarlierWrite,
                              OverlapIntervalsTy &IntervalMap,
                              int64_t &EarlierStart, uint64_t &EarlierSize) {
  if (IntervalMap.empty() || !isShortenableAtTheBeginning(EarlierWrite))
    return false;

  OverlapIntervalsTy::iterator OII = IntervalMap.begin();
  int64_t LaterStart = OII->second;
  uint64_t LaterSize = OII->first - LaterStart;

  assert(OII->first - LaterStart >= 0 && "Size expected to be positive");

  if (LaterStart <= EarlierStart &&
      // Note: "EarlierStart - LaterStart" is known to be non negative due to
      // preceding check.
      LaterSize > (uint64_t)(EarlierStart - LaterStart)) {
    // Note: "LaterSize - (uint64_t)(EarlierStart - LaterStart)" is known to be
    // positive due to preceding checks.
    assert(LaterSize - (uint64_t)(EarlierStart - LaterStart) < EarlierSize &&
           "Should have been handled as OW_Complete");
    if (tryToShorten(EarlierWrite, EarlierStart, EarlierSize, LaterStart,
                     LaterSize, false)) {
      IntervalMap.erase(OII);
      return true;
    }
  }
  return false;
}

static bool removePartiallyOverlappedStores(const DataLayout &DL,
                                            InstOverlapIntervalsTy &IOL,
                                            const TargetLibraryInfo &TLI) {
  bool Changed = false;
  for (auto OI : IOL) {
    Instruction *EarlierWrite = OI.first;
    MemoryLocation Loc = getLocForWrite(EarlierWrite, TLI);
    assert(isRemovable(EarlierWrite) && "Expect only removable instruction");

    const Value *Ptr = Loc.Ptr->stripPointerCasts();
    int64_t EarlierStart = 0;
    uint64_t EarlierSize = Loc.Size.getValue();
    GetPointerBaseWithConstantOffset(Ptr, EarlierStart, DL);
    OverlapIntervalsTy &IntervalMap = OI.second;
    Changed |=
        tryToShortenEnd(EarlierWrite, IntervalMap, EarlierStart, EarlierSize);
    if (IntervalMap.empty())
      continue;
    Changed |=
        tryToShortenBegin(EarlierWrite, IntervalMap, EarlierStart, EarlierSize);
  }
  return Changed;
}

static Constant *tryToMergePartialOverlappingStores(
    StoreInst *Earlier, StoreInst *Later, int64_t InstWriteOffset,
    int64_t DepWriteOffset, const DataLayout &DL, BatchAAResults &AA,
    DominatorTree *DT) {

  if (Earlier && isa<ConstantInt>(Earlier->getValueOperand()) &&
      DL.typeSizeEqualsStoreSize(Earlier->getValueOperand()->getType()) &&
      Later && isa<ConstantInt>(Later->getValueOperand()) &&
      DL.typeSizeEqualsStoreSize(Later->getValueOperand()->getType()) &&
      memoryIsNotModifiedBetween(Earlier, Later, AA, DL, DT)) {
    // If the store we find is:
    //   a) partially overwritten by the store to 'Loc'
    //   b) the later store is fully contained in the earlier one and
    //   c) they both have a constant value
    //   d) none of the two stores need padding
    // Merge the two stores, replacing the earlier store's value with a
    // merge of both values.
    // TODO: Deal with other constant types (vectors, etc), and probably
    // some mem intrinsics (if needed)

    APInt EarlierValue =
        cast<ConstantInt>(Earlier->getValueOperand())->getValue();
    APInt LaterValue = cast<ConstantInt>(Later->getValueOperand())->getValue();
    unsigned LaterBits = LaterValue.getBitWidth();
    assert(EarlierValue.getBitWidth() > LaterValue.getBitWidth());
    LaterValue = LaterValue.zext(EarlierValue.getBitWidth());

    // Offset of the smaller store inside the larger store
    unsigned BitOffsetDiff = (InstWriteOffset - DepWriteOffset) * 8;
    unsigned LShiftAmount = DL.isBigEndian() ? EarlierValue.getBitWidth() -
                                                   BitOffsetDiff - LaterBits
                                             : BitOffsetDiff;
    APInt Mask = APInt::getBitsSet(EarlierValue.getBitWidth(), LShiftAmount,
                                   LShiftAmount + LaterBits);
    // Clear the bits we'll be replacing, then OR with the smaller
    // store, shifted appropriately.
    APInt Merged = (EarlierValue & ~Mask) | (LaterValue << LShiftAmount);
    LLVM_DEBUG(dbgs() << "DSE: Merge Stores:\n  Earlier: " << *Earlier
                      << "\n  Later: " << *Later
                      << "\n  Merged Value: " << Merged << '\n');
    return ConstantInt::get(Earlier->getValueOperand()->getType(), Merged);
  }
  return nullptr;
}

namespace {
// Returns true if \p I is an intrisnic that does not read or write memory.
bool isNoopIntrinsic(Instruction *I) {
  if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
    switch (II->getIntrinsicID()) {
    case Intrinsic::lifetime_start:
    case Intrinsic::lifetime_end:
    case Intrinsic::invariant_end:
    case Intrinsic::launder_invariant_group:
    case Intrinsic::assume:
      return true;
    case Intrinsic::dbg_addr:
    case Intrinsic::dbg_declare:
    case Intrinsic::dbg_label:
    case Intrinsic::dbg_value:
      llvm_unreachable("Intrinsic should not be modeled in MemorySSA");
    default:
      return false;
    }
  }
  return false;
}

// Check if we can ignore \p D for DSE.
bool canSkipDef(MemoryDef *D, bool DefVisibleToCaller,
                const TargetLibraryInfo &TLI) {
  Instruction *DI = D->getMemoryInst();
  // Calls that only access inaccessible memory cannot read or write any memory
  // locations we consider for elimination.
  if (auto *CB = dyn_cast<CallBase>(DI))
    if (CB->onlyAccessesInaccessibleMemory()) {
      if (isAllocLikeFn(DI, &TLI))
        return false;
      return true;
    }
  // We can eliminate stores to locations not visible to the caller across
  // throwing instructions.
  if (DI->mayThrow() && !DefVisibleToCaller)
    return true;

  // We can remove the dead stores, irrespective of the fence and its ordering
  // (release/acquire/seq_cst). Fences only constraints the ordering of
  // already visible stores, it does not make a store visible to other
  // threads. So, skipping over a fence does not change a store from being
  // dead.
  if (isa<FenceInst>(DI))
    return true;

  // Skip intrinsics that do not really read or modify memory.
  if (isNoopIntrinsic(DI))
    return true;

  return false;
}

struct DSEState {
  Function &F;
  AliasAnalysis &AA;

  /// The single BatchAA instance that is used to cache AA queries. It will
  /// not be invalidated over the whole run. This is safe, because:
  /// 1. Only memory writes are removed, so the alias cache for memory
  ///    locations remains valid.
  /// 2. No new instructions are added (only instructions removed), so cached
  ///    information for a deleted value cannot be accessed by a re-used new
  ///    value pointer.
  BatchAAResults BatchAA;

  MemorySSA &MSSA;
  DominatorTree &DT;
  PostDominatorTree &PDT;
  const TargetLibraryInfo &TLI;
  const DataLayout &DL;
  const LoopInfo &LI;

  // Whether the function contains any irreducible control flow, useful for
  // being accurately able to detect loops.
  bool ContainsIrreducibleLoops;

  // All MemoryDefs that potentially could kill other MemDefs.
  SmallVector<MemoryDef *, 64> MemDefs;
  // Any that should be skipped as they are already deleted
  SmallPtrSet<MemoryAccess *, 4> SkipStores;
  // Keep track of all of the objects that are invisible to the caller before
  // the function returns.
  // SmallPtrSet<const Value *, 16> InvisibleToCallerBeforeRet;
  DenseMap<const Value *, bool> InvisibleToCallerBeforeRet;
  // Keep track of all of the objects that are invisible to the caller after
  // the function returns.
  DenseMap<const Value *, bool> InvisibleToCallerAfterRet;
  // Keep track of blocks with throwing instructions not modeled in MemorySSA.
  SmallPtrSet<BasicBlock *, 16> ThrowingBlocks;
  // Post-order numbers for each basic block. Used to figure out if memory
  // accesses are executed before another access.
  DenseMap<BasicBlock *, unsigned> PostOrderNumbers;

  /// Keep track of instructions (partly) overlapping with killing MemoryDefs per
  /// basic block.
  DenseMap<BasicBlock *, InstOverlapIntervalsTy> IOLs;

  DSEState(Function &F, AliasAnalysis &AA, MemorySSA &MSSA, DominatorTree &DT,
           PostDominatorTree &PDT, const TargetLibraryInfo &TLI,
           const LoopInfo &LI)
      : F(F), AA(AA), BatchAA(AA), MSSA(MSSA), DT(DT), PDT(PDT), TLI(TLI),
        DL(F.getParent()->getDataLayout()), LI(LI) {}

  static DSEState get(Function &F, AliasAnalysis &AA, MemorySSA &MSSA,
                      DominatorTree &DT, PostDominatorTree &PDT,
                      const TargetLibraryInfo &TLI, const LoopInfo &LI) {
    DSEState State(F, AA, MSSA, DT, PDT, TLI, LI);
    // Collect blocks with throwing instructions not modeled in MemorySSA and
    // alloc-like objects.
    unsigned PO = 0;
    for (BasicBlock *BB : post_order(&F)) {
      State.PostOrderNumbers[BB] = PO++;
      for (Instruction &I : *BB) {
        MemoryAccess *MA = MSSA.getMemoryAccess(&I);
        if (I.mayThrow() && !MA)
          State.ThrowingBlocks.insert(I.getParent());

        auto *MD = dyn_cast_or_null<MemoryDef>(MA);
        if (MD && State.MemDefs.size() < MemorySSADefsPerBlockLimit &&
            (State.getLocForWriteEx(&I) || State.isMemTerminatorInst(&I)))
          State.MemDefs.push_back(MD);
      }
    }

    // Treat byval or inalloca arguments the same as Allocas, stores to them are
    // dead at the end of the function.
    for (Argument &AI : F.args())
      if (AI.hasPassPointeeByValueCopyAttr()) {
        // For byval, the caller doesn't know the address of the allocation.
        if (AI.hasByValAttr())
          State.InvisibleToCallerBeforeRet.insert({&AI, true});
        State.InvisibleToCallerAfterRet.insert({&AI, true});
      }

    // Collect whether there is any irreducible control flow in the function.
    State.ContainsIrreducibleLoops = mayContainIrreducibleControl(F, &LI);

    return State;
  }

  /// Return 'OW_Complete' if a store to the 'Later' location (by \p LaterI
  /// instruction) completely overwrites a store to the 'Earlier' location.
  /// (by \p EarlierI instruction).
  /// Return OW_MaybePartial if \p Later does not completely overwrite
  /// \p Earlier, but they both write to the same underlying object. In that
  /// case, use isPartialOverwrite to check if \p Later partially overwrites
  /// \p Earlier. Returns 'OW_Unknown' if nothing can be determined.
  OverwriteResult
  isOverwrite(const Instruction *LaterI, const Instruction *EarlierI,
              const MemoryLocation &Later, const MemoryLocation &Earlier,
              int64_t &EarlierOff, int64_t &LaterOff) {
    // AliasAnalysis does not always account for loops. Limit overwrite checks
    // to dependencies for which we can guarantee they are independant of any
    // loops they are in.
    if (!isGuaranteedLoopIndependent(EarlierI, LaterI, Earlier))
      return OW_Unknown;

    // FIXME: Vet that this works for size upper-bounds. Seems unlikely that we'll
    // get imprecise values here, though (except for unknown sizes).
    if (!Later.Size.isPrecise() || !Earlier.Size.isPrecise()) {
      // In case no constant size is known, try to an IR values for the number
      // of bytes written and check if they match.
      const auto *LaterMemI = dyn_cast<MemIntrinsic>(LaterI);
      const auto *EarlierMemI = dyn_cast<MemIntrinsic>(EarlierI);
      if (LaterMemI && EarlierMemI) {
        const Value *LaterV = LaterMemI->getLength();
        const Value *EarlierV = EarlierMemI->getLength();
        if (LaterV == EarlierV && BatchAA.isMustAlias(Earlier, Later))
          return OW_Complete;
      }

      // Masked stores have imprecise locations, but we can reason about them
      // to some extent.
      return isMaskedStoreOverwrite(LaterI, EarlierI, BatchAA);
    }

    const uint64_t LaterSize = Later.Size.getValue();
    const uint64_t EarlierSize = Earlier.Size.getValue();

    // Query the alias information
    AliasResult AAR = BatchAA.alias(Later, Earlier);

    // If the start pointers are the same, we just have to compare sizes to see if
    // the later store was larger than the earlier store.
    if (AAR == AliasResult::MustAlias) {
      // Make sure that the Later size is >= the Earlier size.
      if (LaterSize >= EarlierSize)
        return OW_Complete;
    }

    // If we hit a partial alias we may have a full overwrite
    if (AAR == AliasResult::PartialAlias && AAR.hasOffset()) {
      int32_t Off = AAR.getOffset();
      if (Off >= 0 && (uint64_t)Off + EarlierSize <= LaterSize)
        return OW_Complete;
    }

    // Check to see if the later store is to the entire object (either a global,
    // an alloca, or a byval/inalloca argument).  If so, then it clearly
    // overwrites any other store to the same object.
    const Value *P1 = Earlier.Ptr->stripPointerCasts();
    const Value *P2 = Later.Ptr->stripPointerCasts();
    const Value *UO1 = getUnderlyingObject(P1), *UO2 = getUnderlyingObject(P2);

    // If we can't resolve the same pointers to the same object, then we can't
    // analyze them at all.
    if (UO1 != UO2)
      return OW_Unknown;

    // If the "Later" store is to a recognizable object, get its size.
    uint64_t ObjectSize = getPointerSize(UO2, DL, TLI, &F);
    if (ObjectSize != MemoryLocation::UnknownSize)
      if (ObjectSize == LaterSize && ObjectSize >= EarlierSize)
        return OW_Complete;

    // Okay, we have stores to two completely different pointers.  Try to
    // decompose the pointer into a "base + constant_offset" form.  If the base
    // pointers are equal, then we can reason about the two stores.
    EarlierOff = 0;
    LaterOff = 0;
    const Value *BP1 = GetPointerBaseWithConstantOffset(P1, EarlierOff, DL);
    const Value *BP2 = GetPointerBaseWithConstantOffset(P2, LaterOff, DL);

    // If the base pointers still differ, we have two completely different stores.
    if (BP1 != BP2)
      return OW_Unknown;

    // The later access completely overlaps the earlier store if and only if
    // both start and end of the earlier one is "inside" the later one:
    //    |<->|--earlier--|<->|
    //    |-------later-------|
    // Accesses may overlap if and only if start of one of them is "inside"
    // another one:
    //    |<->|--earlier--|<----->|
    //    |-------later-------|
    //           OR
    //    |----- earlier -----|
    //    |<->|---later---|<----->|
    //
    // We have to be careful here as *Off is signed while *.Size is unsigned.

    // Check if the earlier access starts "not before" the later one.
    if (EarlierOff >= LaterOff) {
      // If the earlier access ends "not after" the later access then the earlier
      // one is completely overwritten by the later one.
      if (uint64_t(EarlierOff - LaterOff) + EarlierSize <= LaterSize)
        return OW_Complete;
      // If start of the earlier access is "before" end of the later access then
      // accesses overlap.
      else if ((uint64_t)(EarlierOff - LaterOff) < LaterSize)
        return OW_MaybePartial;
    }
    // If start of the later access is "before" end of the earlier access then
    // accesses overlap.
    else if ((uint64_t)(LaterOff - EarlierOff) < EarlierSize) {
      return OW_MaybePartial;
    }

    // Can reach here only if accesses are known not to overlap. There is no
    // dedicated code to indicate no overlap so signal "unknown".
    return OW_Unknown;
  }

  bool isInvisibleToCallerAfterRet(const Value *V) {
    if (isa<AllocaInst>(V))
      return true;
    auto I = InvisibleToCallerAfterRet.insert({V, false});
    if (I.second) {
      if (!isInvisibleToCallerBeforeRet(V)) {
        I.first->second = false;
      } else {
        auto *Inst = dyn_cast<Instruction>(V);
        if (Inst && isAllocLikeFn(Inst, &TLI))
          I.first->second = !PointerMayBeCaptured(V, true, false);
      }
    }
    return I.first->second;
  }

  bool isInvisibleToCallerBeforeRet(const Value *V) {
    if (isa<AllocaInst>(V))
      return true;
    auto I = InvisibleToCallerBeforeRet.insert({V, false});
    if (I.second) {
      auto *Inst = dyn_cast<Instruction>(V);
      if (Inst && isAllocLikeFn(Inst, &TLI))
        // NOTE: This could be made more precise by PointerMayBeCapturedBefore
        // with the killing MemoryDef. But we refrain from doing so for now to
        // limit compile-time and this does not cause any changes to the number
        // of stores removed on a large test set in practice.
        I.first->second = !PointerMayBeCaptured(V, false, true);
    }
    return I.first->second;
  }

  Optional<MemoryLocation> getLocForWriteEx(Instruction *I) const {
    if (!I->mayWriteToMemory())
      return None;

    if (auto *MTI = dyn_cast<AnyMemIntrinsic>(I))
      return {MemoryLocation::getForDest(MTI)};

    if (auto *CB = dyn_cast<CallBase>(I)) {
      // If the functions may write to memory we do not know about, bail out.
      if (!CB->onlyAccessesArgMemory() &&
          !CB->onlyAccessesInaccessibleMemOrArgMem())
        return None;

      LibFunc LF;
      if (TLI.getLibFunc(*CB, LF) && TLI.has(LF)) {
        switch (LF) {
        case LibFunc_strcpy:
        case LibFunc_strncpy:
        case LibFunc_strcat:
        case LibFunc_strncat:
          return {MemoryLocation::getAfter(CB->getArgOperand(0))};
        default:
          break;
        }
      }
      switch (CB->getIntrinsicID()) {
      case Intrinsic::init_trampoline:
        return {MemoryLocation::getAfter(CB->getArgOperand(0))};
      case Intrinsic::masked_store:
        return {MemoryLocation::getForArgument(CB, 1, TLI)};
      default:
        break;
      }
      return None;
    }

    return MemoryLocation::getOrNone(I);
  }

  /// Returns true if \p UseInst completely overwrites \p DefLoc
  /// (stored by \p DefInst).
  bool isCompleteOverwrite(const MemoryLocation &DefLoc, Instruction *DefInst,
                           Instruction *UseInst) {
    // UseInst has a MemoryDef associated in MemorySSA. It's possible for a
    // MemoryDef to not write to memory, e.g. a volatile load is modeled as a
    // MemoryDef.
    if (!UseInst->mayWriteToMemory())
      return false;

    if (auto *CB = dyn_cast<CallBase>(UseInst))
      if (CB->onlyAccessesInaccessibleMemory())
        return false;

    int64_t InstWriteOffset, DepWriteOffset;
    if (auto CC = getLocForWriteEx(UseInst))
      return isOverwrite(UseInst, DefInst, *CC, DefLoc, DepWriteOffset,
                         InstWriteOffset) == OW_Complete;
    return false;
  }

  /// Returns true if \p Def is not read before returning from the function.
  bool isWriteAtEndOfFunction(MemoryDef *Def) {
    LLVM_DEBUG(dbgs() << "  Check if def " << *Def << " ("
                      << *Def->getMemoryInst()
                      << ") is at the end the function \n");

    auto MaybeLoc = getLocForWriteEx(Def->getMemoryInst());
    if (!MaybeLoc) {
      LLVM_DEBUG(dbgs() << "  ... could not get location for write.\n");
      return false;
    }

    SmallVector<MemoryAccess *, 4> WorkList;
    SmallPtrSet<MemoryAccess *, 8> Visited;
    auto PushMemUses = [&WorkList, &Visited](MemoryAccess *Acc) {
      if (!Visited.insert(Acc).second)
        return;
      for (Use &U : Acc->uses())
        WorkList.push_back(cast<MemoryAccess>(U.getUser()));
    };
    PushMemUses(Def);
    for (unsigned I = 0; I < WorkList.size(); I++) {
      if (WorkList.size() >= MemorySSAScanLimit) {
        LLVM_DEBUG(dbgs() << "  ... hit exploration limit.\n");
        return false;
      }

      MemoryAccess *UseAccess = WorkList[I];
      // Simply adding the users of MemoryPhi to the worklist is not enough,
      // because we might miss read clobbers in different iterations of a loop,
      // for example.
      // TODO: Add support for phi translation to handle the loop case.
      if (isa<MemoryPhi>(UseAccess))
        return false;

      // TODO: Checking for aliasing is expensive. Consider reducing the amount
      // of times this is called and/or caching it.
      Instruction *UseInst = cast<MemoryUseOrDef>(UseAccess)->getMemoryInst();
      if (isReadClobber(*MaybeLoc, UseInst)) {
        LLVM_DEBUG(dbgs() << "  ... hit read clobber " << *UseInst << ".\n");
        return false;
      }

      if (MemoryDef *UseDef = dyn_cast<MemoryDef>(UseAccess))
        PushMemUses(UseDef);
    }
    return true;
  }

  /// If \p I is a memory  terminator like llvm.lifetime.end or free, return a
  /// pair with the MemoryLocation terminated by \p I and a boolean flag
  /// indicating whether \p I is a free-like call.
  Optional<std::pair<MemoryLocation, bool>>
  getLocForTerminator(Instruction *I) const {
    uint64_t Len;
    Value *Ptr;
    if (match(I, m_Intrinsic<Intrinsic::lifetime_end>(m_ConstantInt(Len),
                                                      m_Value(Ptr))))
      return {std::make_pair(MemoryLocation(Ptr, Len), false)};

    if (auto *CB = dyn_cast<CallBase>(I)) {
      if (isFreeCall(I, &TLI))
        return {std::make_pair(MemoryLocation::getAfter(CB->getArgOperand(0)),
                               true)};
    }

    return None;
  }

  /// Returns true if \p I is a memory terminator instruction like
  /// llvm.lifetime.end or free.
  bool isMemTerminatorInst(Instruction *I) const {
    IntrinsicInst *II = dyn_cast<IntrinsicInst>(I);
    return (II && II->getIntrinsicID() == Intrinsic::lifetime_end) ||
           isFreeCall(I, &TLI);
  }

  /// Returns true if \p MaybeTerm is a memory terminator for \p Loc from
  /// instruction \p AccessI.
  bool isMemTerminator(const MemoryLocation &Loc, Instruction *AccessI,
                       Instruction *MaybeTerm) {
    Optional<std::pair<MemoryLocation, bool>> MaybeTermLoc =
        getLocForTerminator(MaybeTerm);

    if (!MaybeTermLoc)
      return false;

    // If the terminator is a free-like call, all accesses to the underlying
    // object can be considered terminated.
    if (getUnderlyingObject(Loc.Ptr) !=
        getUnderlyingObject(MaybeTermLoc->first.Ptr))
      return false;

    auto TermLoc = MaybeTermLoc->first;
    if (MaybeTermLoc->second) {
      const Value *LocUO = getUnderlyingObject(Loc.Ptr);
      return BatchAA.isMustAlias(TermLoc.Ptr, LocUO);
    }
    int64_t InstWriteOffset, DepWriteOffset;
    return isOverwrite(MaybeTerm, AccessI, TermLoc, Loc, DepWriteOffset,
                       InstWriteOffset) == OW_Complete;
  }

  // Returns true if \p Use may read from \p DefLoc.
  bool isReadClobber(const MemoryLocation &DefLoc, Instruction *UseInst) {
    if (isNoopIntrinsic(UseInst))
      return false;

    // Monotonic or weaker atomic stores can be re-ordered and do not need to be
    // treated as read clobber.
    if (auto SI = dyn_cast<StoreInst>(UseInst))
      return isStrongerThan(SI->getOrdering(), AtomicOrdering::Monotonic);

    if (!UseInst->mayReadFromMemory())
      return false;

    if (auto *CB = dyn_cast<CallBase>(UseInst))
      if (CB->onlyAccessesInaccessibleMemory())
        return false;

    // NOTE: For calls, the number of stores removed could be slightly improved
    // by using AA.callCapturesBefore(UseInst, DefLoc, &DT), but that showed to
    // be expensive compared to the benefits in practice. For now, avoid more
    // expensive analysis to limit compile-time.
    return isRefSet(BatchAA.getModRefInfo(UseInst, DefLoc));
  }

  /// Returns true if a dependency between \p Current and \p KillingDef is
  /// guaranteed to be loop invariant for the loops that they are in. Either
  /// because they are known to be in the same block, in the same loop level or
  /// by guaranteeing that \p CurrentLoc only references a single MemoryLocation
  /// during execution of the containing function.
  bool isGuaranteedLoopIndependent(const Instruction *Current,
                                   const Instruction *KillingDef,
                                   const MemoryLocation &CurrentLoc) {
    // If the dependency is within the same block or loop level (being careful
    // of irreducible loops), we know that AA will return a valid result for the
    // memory dependency. (Both at the function level, outside of any loop,
    // would also be valid but we currently disable that to limit compile time).
    if (Current->getParent() == KillingDef->getParent())
      return true;
    const Loop *CurrentLI = LI.getLoopFor(Current->getParent());
    if (!ContainsIrreducibleLoops && CurrentLI &&
        CurrentLI == LI.getLoopFor(KillingDef->getParent()))
      return true;
    // Otherwise check the memory location is invariant to any loops.
    return isGuaranteedLoopInvariant(CurrentLoc.Ptr);
  }

  /// Returns true if \p Ptr is guaranteed to be loop invariant for any possible
  /// loop. In particular, this guarantees that it only references a single
  /// MemoryLocation during execution of the containing function.
  bool isGuaranteedLoopInvariant(const Value *Ptr) {
    auto IsGuaranteedLoopInvariantBase = [this](const Value *Ptr) {
      Ptr = Ptr->stripPointerCasts();
      if (auto *I = dyn_cast<Instruction>(Ptr)) {
        if (isa<AllocaInst>(Ptr))
          return true;

        if (isAllocLikeFn(I, &TLI))
          return true;

        return false;
      }
      return true;
    };

    Ptr = Ptr->stripPointerCasts();
    if (auto *I = dyn_cast<Instruction>(Ptr)) {
      if (I->getParent()->isEntryBlock())
        return true;
    }
    if (auto *GEP = dyn_cast<GEPOperator>(Ptr)) {
      return IsGuaranteedLoopInvariantBase(GEP->getPointerOperand()) &&
             GEP->hasAllConstantIndices();
    }
    return IsGuaranteedLoopInvariantBase(Ptr);
  }

  // Find a MemoryDef writing to \p DefLoc and dominating \p StartAccess, with
  // no read access between them or on any other path to a function exit block
  // if \p DefLoc is not accessible after the function returns. If there is no
  // such MemoryDef, return None. The returned value may not (completely)
  // overwrite \p DefLoc. Currently we bail out when we encounter an aliasing
  // MemoryUse (read).
  Optional<MemoryAccess *>
  getDomMemoryDef(MemoryDef *KillingDef, MemoryAccess *StartAccess,
                  const MemoryLocation &DefLoc, const Value *DefUO,
                  unsigned &ScanLimit, unsigned &WalkerStepLimit,
                  bool IsMemTerm, unsigned &PartialLimit) {
    if (ScanLimit == 0 || WalkerStepLimit == 0) {
      LLVM_DEBUG(dbgs() << "\n    ...  hit scan limit\n");
      return None;
    }

    MemoryAccess *Current = StartAccess;
    Instruction *KillingI = KillingDef->getMemoryInst();
    LLVM_DEBUG(dbgs() << "  trying to get dominating access\n");

    // Find the next clobbering Mod access for DefLoc, starting at StartAccess.
    Optional<MemoryLocation> CurrentLoc;
    for (;; Current = cast<MemoryDef>(Current)->getDefiningAccess()) {
      LLVM_DEBUG({
        dbgs() << "   visiting " << *Current;
        if (!MSSA.isLiveOnEntryDef(Current) && isa<MemoryUseOrDef>(Current))
          dbgs() << " (" << *cast<MemoryUseOrDef>(Current)->getMemoryInst()
                 << ")";
        dbgs() << "\n";
      });

      // Reached TOP.
      if (MSSA.isLiveOnEntryDef(Current)) {
        LLVM_DEBUG(dbgs() << "   ...  found LiveOnEntryDef\n");
        return None;
      }

      // Cost of a step. Accesses in the same block are more likely to be valid
      // candidates for elimination, hence consider them cheaper.
      unsigned StepCost = KillingDef->getBlock() == Current->getBlock()
                              ? MemorySSASameBBStepCost
                              : MemorySSAOtherBBStepCost;
      if (WalkerStepLimit <= StepCost) {
        LLVM_DEBUG(dbgs() << "   ...  hit walker step limit\n");
        return None;
      }
      WalkerStepLimit -= StepCost;

      // Return for MemoryPhis. They cannot be eliminated directly and the
      // caller is responsible for traversing them.
      if (isa<MemoryPhi>(Current)) {
        LLVM_DEBUG(dbgs() << "   ...  found MemoryPhi\n");
        return Current;
      }

      // Below, check if CurrentDef is a valid candidate to be eliminated by
      // KillingDef. If it is not, check the next candidate.
      MemoryDef *CurrentDef = cast<MemoryDef>(Current);
      Instruction *CurrentI = CurrentDef->getMemoryInst();

      if (canSkipDef(CurrentDef, !isInvisibleToCallerBeforeRet(DefUO), TLI))
        continue;

      // Before we try to remove anything, check for any extra throwing
      // instructions that block us from DSEing
      if (mayThrowBetween(KillingI, CurrentI, DefUO)) {
        LLVM_DEBUG(dbgs() << "  ... skip, may throw!\n");
        return None;
      }

      // Check for anything that looks like it will be a barrier to further
      // removal
      if (isDSEBarrier(DefUO, CurrentI)) {
        LLVM_DEBUG(dbgs() << "  ... skip, barrier\n");
        return None;
      }

      // If Current is known to be on path that reads DefLoc or is a read
      // clobber, bail out, as the path is not profitable. We skip this check
      // for intrinsic calls, because the code knows how to handle memcpy
      // intrinsics.
      if (!isa<IntrinsicInst>(CurrentI) && isReadClobber(DefLoc, CurrentI))
        return None;

      // Quick check if there are direct uses that are read-clobbers.
      if (any_of(Current->uses(), [this, &DefLoc, StartAccess](Use &U) {
            if (auto *UseOrDef = dyn_cast<MemoryUseOrDef>(U.getUser()))
              return !MSSA.dominates(StartAccess, UseOrDef) &&
                     isReadClobber(DefLoc, UseOrDef->getMemoryInst());
            return false;
          })) {
        LLVM_DEBUG(dbgs() << "   ...  found a read clobber\n");
        return None;
      }

      // If Current cannot be analyzed or is not removable, check the next
      // candidate.
      if (!hasAnalyzableMemoryWrite(CurrentI, TLI) || !isRemovable(CurrentI))
        continue;

      // If Current does not have an analyzable write location, skip it
      CurrentLoc = getLocForWriteEx(CurrentI);
      if (!CurrentLoc)
        continue;

      // AliasAnalysis does not account for loops. Limit elimination to
      // candidates for which we can guarantee they always store to the same
      // memory location and not located in different loops.
      if (!isGuaranteedLoopIndependent(CurrentI, KillingI, *CurrentLoc)) {
        LLVM_DEBUG(dbgs() << "  ... not guaranteed loop independent\n");
        WalkerStepLimit -= 1;
        continue;
      }

      if (IsMemTerm) {
        // If the killing def is a memory terminator (e.g. lifetime.end), check
        // the next candidate if the current Current does not write the same
        // underlying object as the terminator.
        if (!isMemTerminator(*CurrentLoc, CurrentI, KillingI))
          continue;
      } else {
        int64_t InstWriteOffset, DepWriteOffset;
        auto OR = isOverwrite(KillingI, CurrentI, DefLoc, *CurrentLoc,
                              DepWriteOffset, InstWriteOffset);
        // If Current does not write to the same object as KillingDef, check
        // the next candidate.
        if (OR == OW_Unknown)
          continue;
        else if (OR == OW_MaybePartial) {
          // If KillingDef only partially overwrites Current, check the next
          // candidate if the partial step limit is exceeded. This aggressively
          // limits the number of candidates for partial store elimination,
          // which are less likely to be removable in the end.
          if (PartialLimit <= 1) {
            WalkerStepLimit -= 1;
            continue;
          }
          PartialLimit -= 1;
        }
      }
      break;
    };

    // Accesses to objects accessible after the function returns can only be
    // eliminated if the access is killed along all paths to the exit. Collect
    // the blocks with killing (=completely overwriting MemoryDefs) and check if
    // they cover all paths from EarlierAccess to any function exit.
    SmallPtrSet<Instruction *, 16> KillingDefs;
    KillingDefs.insert(KillingDef->getMemoryInst());
    MemoryAccess *EarlierAccess = Current;
    Instruction *EarlierMemInst =
        cast<MemoryDef>(EarlierAccess)->getMemoryInst();
    LLVM_DEBUG(dbgs() << "  Checking for reads of " << *EarlierAccess << " ("
                      << *EarlierMemInst << ")\n");

    SmallSetVector<MemoryAccess *, 32> WorkList;
    auto PushMemUses = [&WorkList](MemoryAccess *Acc) {
      for (Use &U : Acc->uses())
        WorkList.insert(cast<MemoryAccess>(U.getUser()));
    };
    PushMemUses(EarlierAccess);

    // Optimistically collect all accesses for reads. If we do not find any
    // read clobbers, add them to the cache.
    SmallPtrSet<MemoryAccess *, 16> KnownNoReads;
    if (!EarlierMemInst->mayReadFromMemory())
      KnownNoReads.insert(EarlierAccess);
    // Check if EarlierDef may be read.
    for (unsigned I = 0; I < WorkList.size(); I++) {
      MemoryAccess *UseAccess = WorkList[I];

      LLVM_DEBUG(dbgs() << "   " << *UseAccess);
      // Bail out if the number of accesses to check exceeds the scan limit.
      if (ScanLimit < (WorkList.size() - I)) {
        LLVM_DEBUG(dbgs() << "\n    ...  hit scan limit\n");
        return None;
      }
      --ScanLimit;
      NumDomMemDefChecks++;
      KnownNoReads.insert(UseAccess);

      if (isa<MemoryPhi>(UseAccess)) {
        if (any_of(KillingDefs, [this, UseAccess](Instruction *KI) {
              return DT.properlyDominates(KI->getParent(),
                                          UseAccess->getBlock());
            })) {
          LLVM_DEBUG(dbgs() << " ... skipping, dominated by killing block\n");
          continue;
        }
        LLVM_DEBUG(dbgs() << "\n    ... adding PHI uses\n");
        PushMemUses(UseAccess);
        continue;
      }

      Instruction *UseInst = cast<MemoryUseOrDef>(UseAccess)->getMemoryInst();
      LLVM_DEBUG(dbgs() << " (" << *UseInst << ")\n");

      if (any_of(KillingDefs, [this, UseInst](Instruction *KI) {
            return DT.dominates(KI, UseInst);
          })) {
        LLVM_DEBUG(dbgs() << " ... skipping, dominated by killing def\n");
        continue;
      }

      // A memory terminator kills all preceeding MemoryDefs and all succeeding
      // MemoryAccesses. We do not have to check it's users.
      if (isMemTerminator(*CurrentLoc, EarlierMemInst, UseInst)) {
        LLVM_DEBUG(
            dbgs()
            << " ... skipping, memterminator invalidates following accesses\n");
        continue;
      }

      if (isNoopIntrinsic(cast<MemoryUseOrDef>(UseAccess)->getMemoryInst())) {
        LLVM_DEBUG(dbgs() << "    ... adding uses of intrinsic\n");
        PushMemUses(UseAccess);
        continue;
      }

      if (UseInst->mayThrow() && !isInvisibleToCallerBeforeRet(DefUO)) {
        LLVM_DEBUG(dbgs() << "  ... found throwing instruction\n");
        return None;
      }

      // Uses which may read the original MemoryDef mean we cannot eliminate the
      // original MD. Stop walk.
      if (isReadClobber(*CurrentLoc, UseInst)) {
        LLVM_DEBUG(dbgs() << "    ... found read clobber\n");
        return None;
      }

      // If this worklist walks back to the original memory access (and the
      // pointer is not guarenteed loop invariant) then we cannot assume that a
      // store kills itself.
      if (EarlierAccess == UseAccess &&
          !isGuaranteedLoopInvariant(CurrentLoc->Ptr)) {
        LLVM_DEBUG(dbgs() << "    ... found not loop invariant self access\n");
        return None;
      }
      // Otherwise, for the KillingDef and EarlierAccess we only have to check
      // if it reads the memory location.
      // TODO: It would probably be better to check for self-reads before
      // calling the function.
      if (KillingDef == UseAccess || EarlierAccess == UseAccess) {
        LLVM_DEBUG(dbgs() << "    ... skipping killing def/dom access\n");
        continue;
      }

      // Check all uses for MemoryDefs, except for defs completely overwriting
      // the original location. Otherwise we have to check uses of *all*
      // MemoryDefs we discover, including non-aliasing ones. Otherwise we might
      // miss cases like the following
      //   1 = Def(LoE) ; <----- EarlierDef stores [0,1]
      //   2 = Def(1)   ; (2, 1) = NoAlias,   stores [2,3]
      //   Use(2)       ; MayAlias 2 *and* 1, loads [0, 3].
      //                  (The Use points to the *first* Def it may alias)
      //   3 = Def(1)   ; <---- Current  (3, 2) = NoAlias, (3,1) = MayAlias,
      //                  stores [0,1]
      if (MemoryDef *UseDef = dyn_cast<MemoryDef>(UseAccess)) {
        if (isCompleteOverwrite(*CurrentLoc, EarlierMemInst, UseInst)) {
          BasicBlock *MaybeKillingBlock = UseInst->getParent();
          if (PostOrderNumbers.find(MaybeKillingBlock)->second <
              PostOrderNumbers.find(EarlierAccess->getBlock())->second) {
            if (!isInvisibleToCallerAfterRet(DefUO)) {
              LLVM_DEBUG(dbgs()
                         << "    ... found killing def " << *UseInst << "\n");
              KillingDefs.insert(UseInst);
            }
          } else {
            LLVM_DEBUG(dbgs()
                       << "    ... found preceeding def " << *UseInst << "\n");
            return None;
          }
        } else
          PushMemUses(UseDef);
      }
    }

    // For accesses to locations visible after the function returns, make sure
    // that the location is killed (=overwritten) along all paths from
    // EarlierAccess to the exit.
    if (!isInvisibleToCallerAfterRet(DefUO)) {
      SmallPtrSet<BasicBlock *, 16> KillingBlocks;
      for (Instruction *KD : KillingDefs)
        KillingBlocks.insert(KD->getParent());
      assert(!KillingBlocks.empty() &&
             "Expected at least a single killing block");

      // Find the common post-dominator of all killing blocks.
      BasicBlock *CommonPred = *KillingBlocks.begin();
      for (auto I = std::next(KillingBlocks.begin()), E = KillingBlocks.end();
           I != E; I++) {
        if (!CommonPred)
          break;
        CommonPred = PDT.findNearestCommonDominator(CommonPred, *I);
      }

      // If CommonPred is in the set of killing blocks, just check if it
      // post-dominates EarlierAccess.
      if (KillingBlocks.count(CommonPred)) {
        if (PDT.dominates(CommonPred, EarlierAccess->getBlock()))
          return {EarlierAccess};
        return None;
      }

      // If the common post-dominator does not post-dominate EarlierAccess,
      // there is a path from EarlierAccess to an exit not going through a
      // killing block.
      if (PDT.dominates(CommonPred, EarlierAccess->getBlock())) {
        SetVector<BasicBlock *> WorkList;

        // If CommonPred is null, there are multiple exits from the function.
        // They all have to be added to the worklist.
        if (CommonPred)
          WorkList.insert(CommonPred);
        else
          for (BasicBlock *R : PDT.roots())
            WorkList.insert(R);

        NumCFGTries++;
        // Check if all paths starting from an exit node go through one of the
        // killing blocks before reaching EarlierAccess.
        for (unsigned I = 0; I < WorkList.size(); I++) {
          NumCFGChecks++;
          BasicBlock *Current = WorkList[I];
          if (KillingBlocks.count(Current))
            continue;
          if (Current == EarlierAccess->getBlock())
            return None;

          // EarlierAccess is reachable from the entry, so we don't have to
          // explore unreachable blocks further.
          if (!DT.isReachableFromEntry(Current))
            continue;

          for (BasicBlock *Pred : predecessors(Current))
            WorkList.insert(Pred);

          if (WorkList.size() >= MemorySSAPathCheckLimit)
            return None;
        }
        NumCFGSuccess++;
        return {EarlierAccess};
      }
      return None;
    }

    // No aliasing MemoryUses of EarlierAccess found, EarlierAccess is
    // potentially dead.
    return {EarlierAccess};
  }

  // Delete dead memory defs
  void deleteDeadInstruction(Instruction *SI) {
    MemorySSAUpdater Updater(&MSSA);
    SmallVector<Instruction *, 32> NowDeadInsts;
    NowDeadInsts.push_back(SI);
    --NumFastOther;

    while (!NowDeadInsts.empty()) {
      Instruction *DeadInst = NowDeadInsts.pop_back_val();
      ++NumFastOther;

      // Try to preserve debug information attached to the dead instruction.
      salvageDebugInfo(*DeadInst);
      salvageKnowledge(DeadInst);

      // Remove the Instruction from MSSA.
      if (MemoryAccess *MA = MSSA.getMemoryAccess(DeadInst)) {
        if (MemoryDef *MD = dyn_cast<MemoryDef>(MA)) {
          SkipStores.insert(MD);
        }
        Updater.removeMemoryAccess(MA);
      }

      auto I = IOLs.find(DeadInst->getParent());
      if (I != IOLs.end())
        I->second.erase(DeadInst);
      // Remove its operands
      for (Use &O : DeadInst->operands())
        if (Instruction *OpI = dyn_cast<Instruction>(O)) {
          O = nullptr;
          if (isInstructionTriviallyDead(OpI, &TLI))
            NowDeadInsts.push_back(OpI);
        }

      DeadInst->eraseFromParent();
    }
  }

  // Check for any extra throws between SI and NI that block DSE.  This only
  // checks extra maythrows (those that aren't MemoryDef's). MemoryDef that may
  // throw are handled during the walk from one def to the next.
  bool mayThrowBetween(Instruction *SI, Instruction *NI,
                       const Value *SILocUnd) {
    // First see if we can ignore it by using the fact that SI is an
    // alloca/alloca like object that is not visible to the caller during
    // execution of the function.
    if (SILocUnd && isInvisibleToCallerBeforeRet(SILocUnd))
      return false;

    if (SI->getParent() == NI->getParent())
      return ThrowingBlocks.count(SI->getParent());
    return !ThrowingBlocks.empty();
  }

  // Check if \p NI acts as a DSE barrier for \p SI. The following instructions
  // act as barriers:
  //  * A memory instruction that may throw and \p SI accesses a non-stack
  //  object.
  //  * Atomic stores stronger that monotonic.
  bool isDSEBarrier(const Value *SILocUnd, Instruction *NI) {
    // If NI may throw it acts as a barrier, unless we are to an alloca/alloca
    // like object that does not escape.
    if (NI->mayThrow() && !isInvisibleToCallerBeforeRet(SILocUnd))
      return true;

    // If NI is an atomic load/store stronger than monotonic, do not try to
    // eliminate/reorder it.
    if (NI->isAtomic()) {
      if (auto *LI = dyn_cast<LoadInst>(NI))
        return isStrongerThanMonotonic(LI->getOrdering());
      if (auto *SI = dyn_cast<StoreInst>(NI))
        return isStrongerThanMonotonic(SI->getOrdering());
      if (auto *ARMW = dyn_cast<AtomicRMWInst>(NI))
        return isStrongerThanMonotonic(ARMW->getOrdering());
      if (auto *CmpXchg = dyn_cast<AtomicCmpXchgInst>(NI))
        return isStrongerThanMonotonic(CmpXchg->getSuccessOrdering()) ||
               isStrongerThanMonotonic(CmpXchg->getFailureOrdering());
      llvm_unreachable("other instructions should be skipped in MemorySSA");
    }
    return false;
  }

  /// Eliminate writes to objects that are not visible in the caller and are not
  /// accessed before returning from the function.
  bool eliminateDeadWritesAtEndOfFunction() {
    bool MadeChange = false;
    LLVM_DEBUG(
        dbgs()
        << "Trying to eliminate MemoryDefs at the end of the function\n");
    for (int I = MemDefs.size() - 1; I >= 0; I--) {
      MemoryDef *Def = MemDefs[I];
      if (SkipStores.contains(Def) || !isRemovable(Def->getMemoryInst()))
        continue;

      Instruction *DefI = Def->getMemoryInst();
      auto DefLoc = getLocForWriteEx(DefI);
      if (!DefLoc)
        continue;

      // NOTE: Currently eliminating writes at the end of a function is limited
      // to MemoryDefs with a single underlying object, to save compile-time. In
      // practice it appears the case with multiple underlying objects is very
      // uncommon. If it turns out to be important, we can use
      // getUnderlyingObjects here instead.
      const Value *UO = getUnderlyingObject(DefLoc->Ptr);
      if (!UO || !isInvisibleToCallerAfterRet(UO))
        continue;

      if (isWriteAtEndOfFunction(Def)) {
        // See through pointer-to-pointer bitcasts
        LLVM_DEBUG(dbgs() << "   ... MemoryDef is not accessed until the end "
                             "of the function\n");
        deleteDeadInstruction(DefI);
        ++NumFastStores;
        MadeChange = true;
      }
    }
    return MadeChange;
  }

  /// \returns true if \p Def is a no-op store, either because it
  /// directly stores back a loaded value or stores zero to a calloced object.
  bool storeIsNoop(MemoryDef *Def, const Value *DefUO) {
    StoreInst *Store = dyn_cast<StoreInst>(Def->getMemoryInst());
    MemSetInst *MemSet = dyn_cast<MemSetInst>(Def->getMemoryInst());
    Constant *StoredConstant = nullptr;
    if (Store)
      StoredConstant = dyn_cast<Constant>(Store->getOperand(0));
    if (MemSet)
      StoredConstant = dyn_cast<Constant>(MemSet->getValue());

    if (StoredConstant && StoredConstant->isNullValue()) {
      auto *DefUOInst = dyn_cast<Instruction>(DefUO);
      if (DefUOInst) {
        if (isCallocLikeFn(DefUOInst, &TLI)) {
          auto *UnderlyingDef =
              cast<MemoryDef>(MSSA.getMemoryAccess(DefUOInst));
          // If UnderlyingDef is the clobbering access of Def, no instructions
          // between them can modify the memory location.
          auto *ClobberDef =
              MSSA.getSkipSelfWalker()->getClobberingMemoryAccess(Def);
          return UnderlyingDef == ClobberDef;
        }

        if (MemSet) {
          if (F.hasFnAttribute(Attribute::SanitizeMemory) ||
              F.hasFnAttribute(Attribute::SanitizeAddress) ||
              F.hasFnAttribute(Attribute::SanitizeHWAddress) ||
              F.getName() == "calloc")
            return false;
          auto *Malloc = const_cast<CallInst *>(dyn_cast<CallInst>(DefUOInst));
          if (!Malloc)
            return false;
          auto *InnerCallee = Malloc->getCalledFunction();
          if (!InnerCallee)
            return false;
          LibFunc Func;
          if (!TLI.getLibFunc(*InnerCallee, Func) || !TLI.has(Func) ||
              Func != LibFunc_malloc)
            return false;
          if (Malloc->getOperand(0) == MemSet->getLength()) {
            if (DT.dominates(Malloc, MemSet) &&
                memoryIsNotModifiedBetween(Malloc, MemSet, BatchAA, DL, &DT)) {
              IRBuilder<> IRB(Malloc);
              const auto &DL = Malloc->getModule()->getDataLayout();
              if (auto *Calloc =
                      emitCalloc(ConstantInt::get(IRB.getIntPtrTy(DL), 1),
                                 Malloc->getArgOperand(0), IRB, TLI)) {
                MemorySSAUpdater Updater(&MSSA);
                auto *LastDef = cast<MemoryDef>(
                    Updater.getMemorySSA()->getMemoryAccess(Malloc));
                auto *NewAccess = Updater.createMemoryAccessAfter(
                    cast<Instruction>(Calloc), LastDef, LastDef);
                auto *NewAccessMD = cast<MemoryDef>(NewAccess);
                Updater.insertDef(NewAccessMD, /*RenameUses=*/true);
                Updater.removeMemoryAccess(Malloc);
                Malloc->replaceAllUsesWith(Calloc);
                Malloc->eraseFromParent();
                return true;
              }
              return false;
            }
          }
        }
      }
    }

    if (!Store)
      return false;

    if (auto *LoadI = dyn_cast<LoadInst>(Store->getOperand(0))) {
      if (LoadI->getPointerOperand() == Store->getOperand(1)) {
        // Get the defining access for the load.
        auto *LoadAccess = MSSA.getMemoryAccess(LoadI)->getDefiningAccess();
        // Fast path: the defining accesses are the same.
        if (LoadAccess == Def->getDefiningAccess())
          return true;

        // Look through phi accesses. Recursively scan all phi accesses by
        // adding them to a worklist. Bail when we run into a memory def that
        // does not match LoadAccess.
        SetVector<MemoryAccess *> ToCheck;
        MemoryAccess *Current =
            MSSA.getWalker()->getClobberingMemoryAccess(Def);
        // We don't want to bail when we run into the store memory def. But,
        // the phi access may point to it. So, pretend like we've already
        // checked it.
        ToCheck.insert(Def);
        ToCheck.insert(Current);
        // Start at current (1) to simulate already having checked Def.
        for (unsigned I = 1; I < ToCheck.size(); ++I) {
          Current = ToCheck[I];
          if (auto PhiAccess = dyn_cast<MemoryPhi>(Current)) {
            // Check all the operands.
            for (auto &Use : PhiAccess->incoming_values())
              ToCheck.insert(cast<MemoryAccess>(&Use));
            continue;
          }

          // If we found a memory def, bail. This happens when we have an
          // unrelated write in between an otherwise noop store.
          assert(isa<MemoryDef>(Current) &&
                 "Only MemoryDefs should reach here.");
          // TODO: Skip no alias MemoryDefs that have no aliasing reads.
          // We are searching for the definition of the store's destination.
          // So, if that is the same definition as the load, then this is a
          // noop. Otherwise, fail.
          if (LoadAccess != Current)
            return false;
        }
        return true;
      }
    }

    return false;
  }
};

static bool eliminateDeadStores(Function &F, AliasAnalysis &AA, MemorySSA &MSSA,
                                DominatorTree &DT, PostDominatorTree &PDT,
                                const TargetLibraryInfo &TLI,
                                const LoopInfo &LI) {
  bool MadeChange = false;

  DSEState State = DSEState::get(F, AA, MSSA, DT, PDT, TLI, LI);
  // For each store:
  for (unsigned I = 0; I < State.MemDefs.size(); I++) {
    MemoryDef *KillingDef = State.MemDefs[I];
    if (State.SkipStores.count(KillingDef))
      continue;
    Instruction *SI = KillingDef->getMemoryInst();

    Optional<MemoryLocation> MaybeSILoc;
    if (State.isMemTerminatorInst(SI))
      MaybeSILoc = State.getLocForTerminator(SI).map(
          [](const std::pair<MemoryLocation, bool> &P) { return P.first; });
    else
      MaybeSILoc = State.getLocForWriteEx(SI);

    if (!MaybeSILoc) {
      LLVM_DEBUG(dbgs() << "Failed to find analyzable write location for "
                        << *SI << "\n");
      continue;
    }
    MemoryLocation SILoc = *MaybeSILoc;
    assert(SILoc.Ptr && "SILoc should not be null");
    const Value *SILocUnd = getUnderlyingObject(SILoc.Ptr);

    MemoryAccess *Current = KillingDef;
    LLVM_DEBUG(dbgs() << "Trying to eliminate MemoryDefs killed by "
                      << *Current << " (" << *SI << ")\n");

    unsigned ScanLimit = MemorySSAScanLimit;
    unsigned WalkerStepLimit = MemorySSAUpwardsStepLimit;
    unsigned PartialLimit = MemorySSAPartialStoreLimit;
    // Worklist of MemoryAccesses that may be killed by KillingDef.
    SetVector<MemoryAccess *> ToCheck;

    if (SILocUnd)
      ToCheck.insert(KillingDef->getDefiningAccess());

    bool Shortend = false;
    bool IsMemTerm = State.isMemTerminatorInst(SI);
    // Check if MemoryAccesses in the worklist are killed by KillingDef.
    for (unsigned I = 0; I < ToCheck.size(); I++) {
      Current = ToCheck[I];
      if (State.SkipStores.count(Current))
        continue;

      Optional<MemoryAccess *> Next = State.getDomMemoryDef(
          KillingDef, Current, SILoc, SILocUnd, ScanLimit, WalkerStepLimit,
          IsMemTerm, PartialLimit);

      if (!Next) {
        LLVM_DEBUG(dbgs() << "  finished walk\n");
        continue;
      }

      MemoryAccess *EarlierAccess = *Next;
      LLVM_DEBUG(dbgs() << " Checking if we can kill " << *EarlierAccess);
      if (isa<MemoryPhi>(EarlierAccess)) {
        LLVM_DEBUG(dbgs() << "\n  ... adding incoming values to worklist\n");
        for (Value *V : cast<MemoryPhi>(EarlierAccess)->incoming_values()) {
          MemoryAccess *IncomingAccess = cast<MemoryAccess>(V);
          BasicBlock *IncomingBlock = IncomingAccess->getBlock();
          BasicBlock *PhiBlock = EarlierAccess->getBlock();

          // We only consider incoming MemoryAccesses that come before the
          // MemoryPhi. Otherwise we could discover candidates that do not
          // strictly dominate our starting def.
          if (State.PostOrderNumbers[IncomingBlock] >
              State.PostOrderNumbers[PhiBlock])
            ToCheck.insert(IncomingAccess);
        }
        continue;
      }
      auto *NextDef = cast<MemoryDef>(EarlierAccess);
      Instruction *NI = NextDef->getMemoryInst();
      LLVM_DEBUG(dbgs() << " (" << *NI << ")\n");
      ToCheck.insert(NextDef->getDefiningAccess());
      NumGetDomMemoryDefPassed++;

      if (!DebugCounter::shouldExecute(MemorySSACounter))
        continue;

      MemoryLocation NILoc = *State.getLocForWriteEx(NI);

      if (IsMemTerm) {
        const Value *NIUnd = getUnderlyingObject(NILoc.Ptr);
        if (SILocUnd != NIUnd)
          continue;
        LLVM_DEBUG(dbgs() << "DSE: Remove Dead Store:\n  DEAD: " << *NI
                          << "\n  KILLER: " << *SI << '\n');
        State.deleteDeadInstruction(NI);
        ++NumFastStores;
        MadeChange = true;
      } else {
        // Check if NI overwrites SI.
        int64_t InstWriteOffset, DepWriteOffset;
        OverwriteResult OR = State.isOverwrite(SI, NI, SILoc, NILoc,
                                               DepWriteOffset, InstWriteOffset);
        if (OR == OW_MaybePartial) {
          auto Iter = State.IOLs.insert(
              std::make_pair<BasicBlock *, InstOverlapIntervalsTy>(
                  NI->getParent(), InstOverlapIntervalsTy()));
          auto &IOL = Iter.first->second;
          OR = isPartialOverwrite(SILoc, NILoc, DepWriteOffset, InstWriteOffset,
                                  NI, IOL);
        }

        if (EnablePartialStoreMerging && OR == OW_PartialEarlierWithFullLater) {
          auto *Earlier = dyn_cast<StoreInst>(NI);
          auto *Later = dyn_cast<StoreInst>(SI);
          // We are re-using tryToMergePartialOverlappingStores, which requires
          // Earlier to domiante Later.
          // TODO: implement tryToMergeParialOverlappingStores using MemorySSA.
          if (Earlier && Later && DT.dominates(Earlier, Later)) {
            if (Constant *Merged = tryToMergePartialOverlappingStores(
                    Earlier, Later, InstWriteOffset, DepWriteOffset, State.DL,
                    State.BatchAA, &DT)) {

              // Update stored value of earlier store to merged constant.
              Earlier->setOperand(0, Merged);
              ++NumModifiedStores;
              MadeChange = true;

              Shortend = true;
              // Remove later store and remove any outstanding overlap intervals
              // for the updated store.
              State.deleteDeadInstruction(Later);
              auto I = State.IOLs.find(Earlier->getParent());
              if (I != State.IOLs.end())
                I->second.erase(Earlier);
              break;
            }
          }
        }

        if (OR == OW_Complete) {
          LLVM_DEBUG(dbgs() << "DSE: Remove Dead Store:\n  DEAD: " << *NI
                            << "\n  KILLER: " << *SI << '\n');
          State.deleteDeadInstruction(NI);
          ++NumFastStores;
          MadeChange = true;
        }
      }
    }

    // Check if the store is a no-op.
    if (!Shortend && isRemovable(SI) &&
        State.storeIsNoop(KillingDef, SILocUnd)) {
      LLVM_DEBUG(dbgs() << "DSE: Remove No-Op Store:\n  DEAD: " << *SI << '\n');
      State.deleteDeadInstruction(SI);
      NumRedundantStores++;
      MadeChange = true;
      continue;
    }
  }

  if (EnablePartialOverwriteTracking)
    for (auto &KV : State.IOLs)
      MadeChange |= removePartiallyOverlappedStores(State.DL, KV.second, TLI);

  MadeChange |= State.eliminateDeadWritesAtEndOfFunction();
  return MadeChange;
}
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// DSE Pass
//===----------------------------------------------------------------------===//
PreservedAnalyses DSEPass::run(Function &F, FunctionAnalysisManager &AM) {
  AliasAnalysis &AA = AM.getResult<AAManager>(F);
  const TargetLibraryInfo &TLI = AM.getResult<TargetLibraryAnalysis>(F);
  DominatorTree &DT = AM.getResult<DominatorTreeAnalysis>(F);
  MemorySSA &MSSA = AM.getResult<MemorySSAAnalysis>(F).getMSSA();
  PostDominatorTree &PDT = AM.getResult<PostDominatorTreeAnalysis>(F);
  LoopInfo &LI = AM.getResult<LoopAnalysis>(F);

  bool Changed = eliminateDeadStores(F, AA, MSSA, DT, PDT, TLI, LI);

#ifdef LLVM_ENABLE_STATS
  if (AreStatisticsEnabled())
    for (auto &I : instructions(F))
      NumRemainingStores += isa<StoreInst>(&I);
#endif

  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  PA.preserve<MemorySSAAnalysis>();
  PA.preserve<LoopAnalysis>();
  return PA;
}

namespace {

/// A legacy pass for the legacy pass manager that wraps \c DSEPass.
class DSELegacyPass : public FunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid

  DSELegacyPass() : FunctionPass(ID) {
    initializeDSELegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    AliasAnalysis &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
    DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    const TargetLibraryInfo &TLI =
        getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
    MemorySSA &MSSA = getAnalysis<MemorySSAWrapperPass>().getMSSA();
    PostDominatorTree &PDT =
        getAnalysis<PostDominatorTreeWrapperPass>().getPostDomTree();
    LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();

    bool Changed = eliminateDeadStores(F, AA, MSSA, DT, PDT, TLI, LI);

#ifdef LLVM_ENABLE_STATS
    if (AreStatisticsEnabled())
      for (auto &I : instructions(F))
        NumRemainingStores += isa<StoreInst>(&I);
#endif

    return Changed;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addRequired<PostDominatorTreeWrapperPass>();
    AU.addRequired<MemorySSAWrapperPass>();
    AU.addPreserved<PostDominatorTreeWrapperPass>();
    AU.addPreserved<MemorySSAWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addPreserved<LoopInfoWrapperPass>();
  }
};

} // end anonymous namespace

char DSELegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(DSELegacyPass, "dse", "Dead Store Elimination", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(PostDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(GlobalsAAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MemorySSAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MemoryDependenceWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(DSELegacyPass, "dse", "Dead Store Elimination", false,
                    false)

FunctionPass *llvm::createDeadStoreEliminationPass() {
  return new DSELegacyPass();
}
