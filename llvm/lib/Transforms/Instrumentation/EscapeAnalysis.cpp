//===- EscapeAnalysis.cpp - Intraprocedural Escape Analysis Implementation ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the EscapeAnalysis helper class. It uses a worklist-
// based, backward dataflow analysis to determine if an allocation can escape.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/EscapeAnalysis.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "escape-analysis"

using namespace llvm;

STATISTIC(NumAllocationsAnalyzed, "Number of allocation sites analyzed");
STATISTIC(NumAllocationsEscaped, "Number of allocation sites found to escape");

/// Per-allocation worklist cap (safety valve). If the number of processed
/// worklist nodes exceeds this limit, the analysis bails out conservatively and
/// considers the allocation as escaping.
static cl::opt<unsigned> WorklistLimit(
    "escape-analysis-worklist-limit", cl::init(1000), cl::Hidden,
    cl::desc("Max number of worklist nodes processed per allocation; "
             "if exceeded, assume the allocation escapes"));

// getUnderlyingObjects(..., MaxLookup = 0) is assumed to mean "unbounded".
// If upstream changes semantics, this must be revisited.
static const unsigned VTMaxLookup = 0;

//===----------------------------------------------------------------------===//
// MemorySSA-related utils
//===----------------------------------------------------------------------===//

namespace llvm {
/// Add P to Worklist if it doesn't exist in Seen
template <typename PtrT, typename SetT, typename WorklistT>
static bool tryEnqueueIfNew(PtrT *P, SetT &Seen, WorklistT &Worklist) {
  if (P && Seen.insert(P).second) {
    Worklist.push_back(P);
    return true;
  }
  return false;
}

/// Add incoming unvisited MemoryAccesses of a MemoryPhi to MAWorkList.
static void appendIncomingMAs(const MemoryPhi *MPhi,
                              SmallPtrSetImpl<MemoryAccess *> &VisitedMA,
                              SmallVectorImpl<MemoryAccess *> &MAWorkList,
                              MemoryLocation Loc, MemorySSAWalker *Walker,
                              bool &IsComplete) {
  for (unsigned i = 0, N = MPhi->getNumIncomingValues(); i != N; ++i) {
    MemoryAccess *InMA = MPhi->getIncomingValue(i);
    MemoryAccess *EdgeCl = Walker->getClobberingMemoryAccess(InMA, Loc);
    if (!EdgeCl) {
      IsComplete = false;
      continue;
    }
    tryEnqueueIfNew(EdgeCl, VisitedMA, MAWorkList);
  }
}

enum class EdgeWalkStep { Recurse, SkipSuccessors, Stop };

/// Walk edge clobbering definitions starting from Start MemoryAccess.
template <typename VisitT>
static void walkEdgeClobbers(MemoryAccess *Start, MemorySSAWalker *Walker,
                             MemoryLocation Loc, unsigned Limit,
                             const VisitT &Visit, bool &IsComplete) {
  IsComplete = true;
  if (!Start) {
    IsComplete = false;
    return;
  }

  SmallVector<MemoryAccess *, 32> MAWorklist;
  SmallPtrSet<MemoryAccess *, 32> MAVisited;
  tryEnqueueIfNew(Start, MAVisited, MAWorklist);
  unsigned Steps = 0;

  while (!MAWorklist.empty()) {
    if (++Steps > Limit) {
      IsComplete = false;
      return;
    }

    MemoryAccess *MA = MAWorklist.pop_back_val();

    const EdgeWalkStep Act = Visit(MA);
    if (Act == EdgeWalkStep::Stop)
      return;
    if (Act == EdgeWalkStep::SkipSuccessors)
      continue;

    if (auto *MDef = dyn_cast<MemoryDef>(MA)) {
      MemoryAccess *EdgeCl = Walker->getClobberingMemoryAccess(MDef, Loc);
      if (!EdgeCl) {
        IsComplete = false;
        return;
      }
      tryEnqueueIfNew(EdgeCl, MAVisited, MAWorklist);
    } else if (const auto *MPhi = dyn_cast<MemoryPhi>(MA)) {
      appendIncomingMAs(MPhi, MAVisited, MAWorklist, Loc, Walker, IsComplete);
      if (!IsComplete)
        return;
    } else {
      llvm_unreachable("Unexpected MemoryAccess kind");
    }
  }
}

/// Try to use ValueTracking to find underlying objects.
static bool tryValueTracking(const Value *V, LoopInfo *LI,
                             SmallVectorImpl<const Value *> &Work,
                             SmallPtrSetImpl<const Value *> &Enqueued) {
  SmallVector<const Value *, 4> Bases;
  if (!V->getType()->isPointerTy())
    return false; // Only pointers have underlying objects.

  getUnderlyingObjects(V, Bases, LI, VTMaxLookup);

  if (Bases.empty() || (Bases.size() == 1 && Bases[0] == V))
    return false;

  for (const Value *B : Bases)
    tryEnqueueIfNew(B, Enqueued, Work);
  return true;
}

bool isHeapAllocation(const CallBase *CB, const TargetLibraryInfo &TLI) {
  // Try standard path first (works for C++ new and modern IR with allockind)
  if (isAllocationFn(CB, &TLI) || isNewLikeFn(CB, &TLI))
    return true;

  // Fallback: check directly via TLI for malloc/calloc/etc
  const Function *Callee = CB->getCalledFunction();
  if (!Callee || !Callee->getReturnType()->isPointerTy())
    return false;

  LibFunc Func;
  if (!TLI.getLibFunc(*Callee, Func) || !TLI.has(Func))
    return false;

  // List of known heap allocation functions from libc
  switch (Func) {
  case LibFunc_malloc:
  case LibFunc_calloc:
  case LibFunc_realloc:
  case LibFunc_reallocf:
  case LibFunc_reallocarray:
  case LibFunc_valloc:
  case LibFunc_pvalloc:
  case LibFunc_aligned_alloc:
  case LibFunc_memalign:
  case LibFunc_vec_malloc:
  case LibFunc_vec_calloc:
  case LibFunc_vec_realloc:
  case LibFunc_strdup:
  case LibFunc_strndup:
    return true;
  default:
    return false;
  }
}

void getUnderlyingObjectsThroughLoads(const Value *Ptr, MemorySSA *MSSA,
                                      SmallPtrSetImpl<const Value *> &Result,
                                      const TargetLibraryInfo *TLI,
                                      LoopInfo *LI, bool *IsComplete,
                                      unsigned MaxSteps) {
  LLVM_DEBUG(dbgs() << "getUnderlyingObjectsThroughLoads: " << Ptr->getName()
                    << "\n");

  if (!Ptr->getType()->isPointerTy()) {
    LLVM_DEBUG(dbgs() << "Input is not a pointer: " << *Ptr << "\n");
    return; // Only pointers have underlying objects.
  }

  if (!MSSA) {
    LLVM_DEBUG(dbgs() << "MSSA is null, marking analysis as incomplete\n");
    if (IsComplete)
      *IsComplete = false;
    return;
  }

  auto addTerminal = [&](const Value *Term,
                         bool MarkIncompleteIfNotBase = true) {
    if (!Term || !Term->getType()->isPointerTy())
      return;
    bool IsBase = isa<AllocaInst>(Term) || isa<Argument>(Term) ||
                  isa<GlobalVariable>(Term) || isa<GlobalAlias>(Term) ||
                  isa<ConstantPointerNull>(Term);
    if (!IsBase && TLI) { // Check if it's heap allocation call
      if (const auto *CB = dyn_cast<CallBase>(Term))
        IsBase = isHeapAllocation(CB, *TLI);
    }
    LLVM_DEBUG(dbgs() << "Mark terminal: " << *Term
                      << " IsBase=" << (IsBase ? "yes" : "no") << "\n");
    Result.insert(Term);
    if (IsComplete && !IsBase && MarkIncompleteIfNotBase) {
      *IsComplete = false;
      LLVM_DEBUG(dbgs() << "Marking incomplete due to non-base\n");
    }
  };

  SmallPtrSet<const Value *, 32> ValueTrackingSeen;
  SmallPtrSet<const Value *, 32> Seen;
  SmallVector<const Value *, 32> Worklist;

  auto bail = [&]() {
    if (IsComplete)
      *IsComplete = false;
    for (const Value *WV : Worklist)
      addTerminal(WV);
  };

  tryEnqueueIfNew(Ptr, Seen, Worklist);

  unsigned Step = 0;
  if (IsComplete)
    *IsComplete = true;

  MemorySSAWalker *Walker = MSSA->getSkipSelfWalker();

  while (!Worklist.empty()) {
    const Value *CurrPtr = Worklist.pop_back_val();

    // Safety valve: if we exceed MaxSteps, bail out conservatively.
    if (++Step > MaxSteps) {
      LLVM_DEBUG(dbgs() << "MaxSteps exceeded at: " << *CurrPtr << "\n");
      addTerminal(CurrPtr);
      bail();
      return;
    }

    // Try ValueTracking first (only once per value)
    if (!isa<LoadInst>(CurrPtr) && ValueTrackingSeen.insert(CurrPtr).second &&
        tryValueTracking(CurrPtr, LI, Worklist, Seen))
      continue; // Successfully expanded via ValueTracking;

    const auto *Load = dyn_cast<LoadInst>(CurrPtr);
    if (!Load || !Load->isSimple()) {
      addTerminal(CurrPtr);
      continue;
    }

    // Use MemorySSA's API to get the clobbering MemoryAccess.
    MemoryAccess *Clobber = Walker->getClobberingMemoryAccess(Load);
    const auto LoadLoc = MemoryLocation::get(Load);

    // Local accumulators for Load
    SmallVector<const Value *, 8> LocalWorklist;
    SmallPtrSet<const Value *, 8> LocalSeen;

    LocalSeen.insert(Load);
    bool Fallback = false;
    bool MAWalkComplete = false;
    // Limit MemorySSA walk to half of the budget
    const unsigned MAIterationLimit = std::max(1u, MaxSteps / 2);

    walkEdgeClobbers(
        Clobber, Walker, LoadLoc, MAIterationLimit,
        [&](MemoryAccess *MA) -> EdgeWalkStep {
          if (MSSA->isLiveOnEntryDef(MA)) {
            LLVM_DEBUG(dbgs() << "LiveOnEntryDef reached, fallback\n");
            Fallback = true;
            return EdgeWalkStep::Stop;
          }

          if (const auto *MDef = dyn_cast<MemoryDef>(MA)) {
            const Instruction *I = MDef->getMemoryInst();
            assert(I && "MemoryDef must have an instruction");

            if (const auto *Store = dyn_cast<StoreInst>(I)) {
              if (!Store->isSimple()) {
                Fallback = true;
                return EdgeWalkStep::Stop;
              }
              const Value *SV = Store->getValueOperand();
              if (SV->getType()->isPointerTy()) {
                tryEnqueueIfNew(SV, LocalSeen, LocalWorklist);
                // Reached defining store for LoadLoc — stop this path here.
                return EdgeWalkStep::SkipSuccessors;
              }
              LLVM_DEBUG(dbgs() << "Non-pointer store: " << *Store << "\n");
              Fallback = true;
              return EdgeWalkStep::Stop;
            }
            // NOTE: We intentionally don't consider the source in memintrinsics
            // (memmove/memcpy): they are not semantically underlying objects.
            // Conservatively assume escape.
            LLVM_DEBUG(dbgs() << "Unrecognized defining write, fallback\n");
            Fallback = true;
            return EdgeWalkStep::Stop;
          }
          return EdgeWalkStep::Recurse;
        },
        MAWalkComplete);

    if (!MAWalkComplete)
      Fallback = true;

    if (Fallback) {
      LLVM_DEBUG(dbgs() << "Fallback: mark Load as term: " << *Load << "\n");
      addTerminal(Load);
    } else {
      for (const auto *WV : LocalWorklist)
        tryEnqueueIfNew(WV, Seen, Worklist);
    }
  } // end while for Work
  LLVM_DEBUG(dbgs() << "getUnderlyingObjectsThroughLoads: " << Ptr->getName()
                    << " -- end\n");
}

} // end namespace llvm

//===----------------------------------------------------------------------===//
// EscapeCaptureTracker Implementation
//===----------------------------------------------------------------------===//

bool EscapeAnalysisInfo::EscapeCaptureTracker::shouldExplore(const Use *U) {
  // Always explore, but we can add optimizations here later
  return true;
}

bool EscapeAnalysisInfo::isExternalObject(const Value *Base) {
  return isa<GlobalVariable>(Base) || isa<GlobalAlias>(Base) ||
         isa<Argument>(Base);
}

bool EscapeAnalysisInfo::EscapeCaptureTracker::doesStoreDestEscape(
    const Value *Dest) {
  // Find base objects for the storage location
  SmallPtrSet<const Value *, 8> BaseObjects;
  bool IsComplete = false;
  getUnderlyingObjectsThroughLoads(Dest, EAI.MSSA, BaseObjects, EAI.TLI, EAI.LI,
                                   &IsComplete);

  // If bases are unknown or the walk is incomplete, be conservative.
  if (BaseObjects.empty() || !IsComplete) {
    LLVM_DEBUG(dbgs() << "  Store destination unknown/incomplete, escapes\n");
    return true;
  }

  for (const Value *Base : BaseObjects) {
    LLVM_DEBUG(dbgs() << "  Store destination Base: " << *Base << "\n");
    if (isExternalObject(Base)) {
      LLVM_DEBUG(dbgs() << "  Stored to external object, escapes\n");
      return true;
    }

    // If storing to another local allocation, recursively check if it escapes
    if (const auto *Alloca = dyn_cast<AllocaInst>(Base)) {
      // Recurse to decide whether the target alloca itself escapes.
      if (EAI.solveEscapeFor(*Alloca, ProcessingSet)) {
        LLVM_DEBUG(dbgs() << "  Stored to escaping alloca, escapes\n");
        return true;
      }
    } else if (const auto *CB = dyn_cast<CallBase>(Base)) {
      // Store to malloc/new call result (heap allocation).
      if (isHeapAllocation(CB, *EAI.TLI)) {
        if (EAI.solveEscapeFor(*CB, ProcessingSet)) {
          LLVM_DEBUG(dbgs() << "  Stored into escaping heap alloc, escapes\n");
          return true; // Stored into escaping heap allocation — escapes.
        }
        continue; // Stored into non-escaping heap allocation — OK.
      }
      LLVM_DEBUG(dbgs() << "  Stored to unknown call result, escapes\n");
      return true; // Unknown call result — escapes.
    } else {
      // Any other/unknown terminal means the destination is not proven local.
      LLVM_DEBUG(dbgs() << "  Stored to unknown location, escapes\n");
      return true;
    }
  }
  return false;
}

SmallVector<unsigned, 8>
EscapeAnalysisInfo::EscapeCaptureTracker::getNoCapturePointerArgIndices(
    const CallBase *CB) const {
  SmallVector<unsigned, 8> Indices;
  for (unsigned ArgNo = 0, End = CB->arg_size(); ArgNo != End; ++ArgNo) {
    const auto *Opnd = CB->getArgOperand(ArgNo);
    if (!Opnd || !Opnd->getType()->isPointerTy() ||
        CB->paramHasAttr(ArgNo, Attribute::WriteOnly))
      continue;
    CaptureInfo CI = CB->getCaptureInfo(ArgNo);
    if (capturesNothing(
            UseCaptureInfo(CI.getOtherComponents(), CI.getRetComponents())))
      Indices.push_back(ArgNo);
  }
  return Indices;
}

bool EscapeAnalysisInfo::EscapeCaptureTracker::canEscapeViaNocaptureArgs(
    const CallBase &CB, ArrayRef<unsigned> NoCapPtrArgs,
    SmallPtrSetImpl<const Value *> &StorePtrOpndBases) const {
  // For each nocapture arg, compute bases and check if it is the QueryObj.
  for (unsigned ArgIdx : NoCapPtrArgs) {
    const Value *OpV = CB.getArgOperand(ArgIdx);
    SmallPtrSet<const Value *, 8> ArgBases;
    bool Complete = false;
    getUnderlyingObjectsThroughLoads(OpV, EAI.MSSA, ArgBases, EAI.TLI, EAI.LI,
                                     &Complete);
    if (!Complete)
      return true;
    for (const Value *StoreBaseObj : StorePtrOpndBases) {
      if (ArgBases.contains(StoreBaseObj))
        return true;
    }
  }
  return false;
}

bool EscapeAnalysisInfo::EscapeCaptureTracker::stemsFromStartStore(
    MemoryUseOrDef *MUOD, const MemoryDef *StartMDef, MemoryLocation Loc,
    bool &IsComplete, MemorySSAWalker *Walker) const {
  LLVM_DEBUG(dbgs() << "  stemsFromStartStore: clobber " << *MUOD << "\n");
  MemoryAccess *Clobber =
      Walker->getClobberingMemoryAccess(MUOD->getDefiningAccess(), Loc);
  if (!Clobber) {
    IsComplete = false;
    return true;
  }
  bool WalkComplete = false;
  bool Found = false;

  walkEdgeClobbers(
      Clobber, Walker, Loc, WorklistLimit,
      [&](MemoryAccess *MA) {
        LLVM_DEBUG(dbgs() << "    inspect MA: " << *MA << "\n");
        if (MA == StartMDef) {
          Found = true;
          return EdgeWalkStep::Stop;
        }
        return EdgeWalkStep::Recurse;
      },
      WalkComplete);

  if (!WalkComplete)
    IsComplete = false;
  return Found || !WalkComplete; // conservative on incompleteness
}

SmallVector<const LoadInst *, 32>
EscapeAnalysisInfo::EscapeCaptureTracker::findStoreReadersAndExports(
    const StoreInst *StartStore, bool &ContentMayEscape, bool &IsComplete) {
  IsComplete = true;
  ContentMayEscape = false;
  auto *StartMDef = cast<MemoryDef>(EAI.MSSA->getMemoryAccess(StartStore));
  LLVM_DEBUG(dbgs() << "Collecting Loads reading from Store: " << *StartStore
                    << "\t" << *StartMDef << "\n");
  const MemoryLocation LocDest = MemoryLocation::get(StartStore);
  MemorySSAWalker *Walker = EAI.MSSA->getSkipSelfWalker();

  // A call may export memory and cause its escape.
  auto mayReadMemory = [](const CallBase *CB) {
    if (!CB || CB->doesNotAccessMemory() || CB->onlyWritesMemory())
      return false;
    return true;
  };

  // Need for nocapture arguments inquire
  SmallPtrSet<const Value *, 8> StorePtrBases;
  getUnderlyingObjectsThroughLoads(StartStore->getPointerOperand(), EAI.MSSA,
                                   StorePtrBases, EAI.TLI, EAI.LI, &IsComplete);

  SmallVector<const LoadInst *, 32> ResLoads;
  SmallVector<MemoryAccess *, 32> MAWorklist;
  SmallPtrSet<MemoryAccess *, 32> MAVisited;
  tryEnqueueIfNew(StartMDef, MAVisited, MAWorklist);

  unsigned Steps = 0;
  while (!MAWorklist.empty()) {
    if (++Steps > WorklistLimit) {
      IsComplete = false;
      return {};
    }

    MemoryAccess *MA = MAWorklist.pop_back_val();
    for (User *U : MA->users()) {
      if (auto *MUOD = dyn_cast<MemoryUseOrDef>(U)) {
        if (const Instruction *I = MUOD->getMemoryInst()) {
          LLVM_DEBUG(dbgs() << "  UseOrDef: " << *I << "\n");
          // Consider functions which can export the content behind LocDest
          if (const auto *CB = dyn_cast<CallBase>(I); CB && mayReadMemory(CB)) {
            if (auto NoCapArgs = getNoCapturePointerArgIndices(CB);
                canEscapeViaNocaptureArgs(*CB, NoCapArgs, StorePtrBases) &&
                stemsFromStartStore(MUOD, StartMDef, LocDest, IsComplete,
                                    Walker)) {
              LLVM_DEBUG(dbgs() << "  Call may export bytes: " << *CB << "\n");
              ContentMayEscape = true;
              return {};
            }
          } else if (auto *MU = dyn_cast<MemoryUse>(MUOD)) {
            if (const auto *Load = dyn_cast<LoadInst>(I);
                Load && stemsFromStartStore(MU, StartMDef, LocDest, IsComplete,
                                            Walker)) {
              LLVM_DEBUG(dbgs() << "  Load read from Store: " << *Load << "\n");
              ResLoads.push_back(Load);
            }
            continue; // No need to enqueue further for MemoryUse
          }
        }
      }
      tryEnqueueIfNew(cast<MemoryAccess>(U), MAVisited, MAWorklist);
    }
  }
  return ResLoads;
}

bool EscapeAnalysisInfo::EscapeCaptureTracker::doesStoredPointerEscapeViaLoads(
    const StoreInst *Store) {
  LLVM_DEBUG(dbgs() << "\n---- doesStoredPointerEscapeViaLoads " << *Store
                    << " ----\n");
  bool IsComplete = false;
  bool ContentMayEscape = false;
  const auto Loads =
      findStoreReadersAndExports(Store, ContentMayEscape, IsComplete);
  if (!IsComplete) {
    LLVM_DEBUG(dbgs() << "  Incomplete load collection, escapes\n");
    return true; // We may have missed loads -> conservatively escape
  }
  if (ContentMayEscape) {
    LLVM_DEBUG(dbgs() << "  Bytes exported by call, escapes\n");
    return true; // Bytes stored may have been exported -> conservatively escape
  }

  for (const LoadInst *Load : Loads) {
    if (!Load->isSimple())
      return true;
    if (!Load->getType()->isPointerTy())
      continue; // Loading non-pointer cannot cause escape
    if (EAI.solveEscapeFor(*Load, ProcessingSet)) {
      LLVM_DEBUG(dbgs() << "  -> escapes via load\n");
      return true;
    }
  }
  LLVM_DEBUG(dbgs() << "  -> does not escape via loads\n");
  return false;
}

CaptureTracker::Action
EscapeAnalysisInfo::EscapeCaptureTracker::captured(const Use *U,
                                                   UseCaptureInfo CI) {
  LLVM_DEBUG(dbgs() << "\n--> Analyzing capture use: " << *U->get() << " in "
                    << *U->getUser() << "\n");
  const auto *I = cast<Instruction>(U->getUser());

  // If CaptureTracking says this use does not capture, continue exploring.
  if (capturesNothing(CI.UseCC)) {
    LLVM_DEBUG(dbgs() << "    Use doesn't capture, continue\n");
    return Continue; // CaptureTracking says it's not captured, continue
  }

  // Passthrough ops (gep/bitcast/select/phi..) should be explored transitively.
  if (CI.isPassthrough()) {
    LLVM_DEBUG(dbgs() << "    Passthrough operation, continue to result\n");
    return Continue;
  }

  // Now handle special cases where CaptureTracking says it's captured,
  // but we need more sophisticated escape analysis

  if (const auto *Store = dyn_cast<StoreInst>(I)) {
    // Check if we're storing the pointer (not storing to it)
    if (Store->getValueOperand() == U->get()) {
      LLVM_DEBUG(dbgs() << "==> Storing pointer value, analyze destination "
                        << *Store->getPointerOperand() << "\n");
      if (!Store->isSimple() ||
          doesStoreDestEscape(Store->getPointerOperand()) ||
          doesStoredPointerEscapeViaLoads(Store)) {
        LLVM_DEBUG(dbgs() << "  Store to escaping destination, escapes\n");
        Escaped = true;
        return Stop;
      }
      return ContinueIgnoringReturn;
    }
    // If we are the destination pointer, this use does not capture the value.
    LLVM_DEBUG(dbgs() << "    Used as store destination, doesn't escape\n");
    return ContinueIgnoringReturn;
  }

  if (isa<ICmpInst>(I)) { // Pure comparisons of addresses do not cause escape.
    LLVM_DEBUG(dbgs() << "    Pointer comparison, doesn't escape\n");
    return ContinueIgnoringReturn;
  }

  // Default: if CaptureTracking still indicates capture, treat as escape.
  if (capturesAnything(CI.UseCC)) {
    LLVM_DEBUG(dbgs() << "  Captured by: " << *I << "\n");
    Escaped = true;
    return Stop;
  }

  llvm_unreachable("Unhandled case in EscapeCaptureTracker::captured");
}

bool EscapeAnalysisInfo::solveEscapeFor(
    const Value &Ptr, SmallPtrSet<const Value *, 32> &ProcessingSet) {
  LLVM_DEBUG(dbgs() << "Solving escape for "
                    << (Ptr.hasName() ? Ptr.getName() : "Load") << "\n");
  if (const auto CacheIt = Cache.find(&Ptr); CacheIt != Cache.end()) {
    LLVM_DEBUG(dbgs() << "  Cached result: "
                      << (CacheIt->second ? "escaped" : "not escaped") << "\n");
    return CacheIt->second;
  }

  if (ProcessingSet.contains(&Ptr)) { // Cycle
    LLVM_DEBUG(dbgs() << "  Cycle detected for " << Ptr.getName()
                      << ", assume escapes\n");
    return true;
  }
  ProcessingSet.insert(&Ptr);

  EscapeCaptureTracker Tracker(*this, ProcessingSet);

  // Use the CaptureTracking infrastructure to analyze the allocation
  PointerMayBeCaptured(&Ptr, &Tracker, /*MaxUsesToExplore=*/WorklistLimit);
  Cache[&Ptr] = Tracker.hasEscaped();

  LLVM_DEBUG(dbgs() << "  Result: "
                    << (Tracker.hasEscaped() ? "escaped" : "not escaped")
                    << "\n");
  return Tracker.hasEscaped();
}

//===----------------------------------------------------------------------===//
// EscapeAnalysis Core Implementation
//===----------------------------------------------------------------------===//

bool EscapeAnalysisInfo::isAllocationSite(const Value *V) {
  if (isa<AllocaInst>(V))
    return true;
  if (const auto *CB = dyn_cast<CallBase>(V))
    return isHeapAllocation(CB, *TLI);
  return false;
}

bool EscapeAnalysisInfo::isEscaping(const Value &Alloc) {
  if (!isAllocationSite(&Alloc)) { // Validate input
    LLVM_DEBUG(dbgs() << "EscapeAnalysis: Not an allocation: " << Alloc
                      << "\n");
    return true; // Conservative: unknown things "escape"
  }

  LLVM_DEBUG(dbgs() << "EscapeAnalysis: Analyzing " << Alloc << "\n");
  NumAllocationsAnalyzed++;

  // Track allocations being processed to detect cycles
  SmallPtrSet<const Value *, 32> ProcessingSet;
  const bool IsEscaped = solveEscapeFor(Alloc, ProcessingSet);

  if (IsEscaped)
    NumAllocationsEscaped++;

  return IsEscaped;
}

void EscapeAnalysisInfo::print(raw_ostream &OS) {
  bool Any = false;
  unsigned UnnamedCount = 0;

  for (Instruction &I : instructions(F)) {
    LLVM_DEBUG(OS << "\nI: " << I << "\n");
    if (!isAllocationSite(&I))
      continue;

    Any = true;

    // Stable symbol: use SSA name if exists, otherwise "unnamed#N".
    StringRef Name = I.hasName() ? I.getName() : StringRef();
    SmallString<32> Gen;
    if (Name.empty()) {
      ++UnnamedCount;
      Gen += "unnamed#";
      Gen += Twine(UnnamedCount).str();
      Name = Gen;
    }

    const bool Esc = isEscaping(I);
    OS << "  " << Name << " escapes: " << (Esc ? "yes" : "no") << "\n";
  }

  if (!Any)
    OS << "  none\n";
  OS << "\n";
}

bool EscapeAnalysisInfo::invalidate(Function &F, const PreservedAnalyses &PA,
                                    FunctionAnalysisManager::Invalidator &Inv) {
  if (auto PAC = PA.getChecker<EscapeAnalysis>();
      PAC.preserved() || PAC.preservedSet<AllAnalysesOn<Function>>())
    return false;

  if (Inv.invalidate<MemorySSAAnalysis>(F, PA) ||
      Inv.invalidate<LoopAnalysis>(F, PA) ||
      Inv.invalidate<TargetLibraryAnalysis>(F, PA)) {
    Cache.clear();
    MSSA = nullptr;
    LI = nullptr;
    TLI = nullptr;
    return true;
  }

  return false;
}

AnalysisKey EscapeAnalysis::Key;

EscapeAnalysis::Result EscapeAnalysis::run(Function &F,
                                           FunctionAnalysisManager &FAM) {
  EscapeAnalysisInfo EAI(F, FAM);
  return EAI;
}

//===----------------------------------------------------------------------===//
// Printing Pass for Verification
//===----------------------------------------------------------------------===//

PreservedAnalyses
EscapeAnalysisPrinterPass::run(Function &F,
                               FunctionAnalysisManager &FAM) const {
  if (F.isDeclaration())
    return PreservedAnalyses::all();
  OS << "Printing analysis 'Escape Analysis' for function '" << F.getName()
     << "':\n";
  FAM.getResult<EscapeAnalysis>(F).print(OS);
  return PreservedAnalyses::all();
}