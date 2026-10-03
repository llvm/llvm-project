//===-- IndirectCallPromotionAnalysis.cpp - Find promotion candidates ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper methods for identifying profitable indirect call promotion
// candidates for an instruction when the indirect-call value profile metadata
// is available.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/IndirectCallPromotionAnalysis.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "pgo-icall-prom-analysis"

namespace llvm {

// The percent threshold for the direct-call target (this call site vs the
// remaining call count) for it to be considered as the promotion target.
static cl::opt<unsigned> ICPRemainingPercentThreshold(
    "icp-remaining-percent-threshold", cl::init(30), cl::Hidden,
    cl::desc("The percentage threshold against remaining unpromoted indirect "
             "call count for the promotion"));

// The percent threshold for the direct-call target (this call site vs the
// total call count) for it to be considered as the promotion target.
static cl::opt<uint64_t>
    ICPTotalPercentThreshold("icp-total-percent-threshold", cl::init(5),
                             cl::Hidden,
                             cl::desc("The percentage threshold against total "
                                      "count for the promotion"));

// Set the minimum absolute count threshold for indirect call promotion.
// Candidates with counts below this threshold will not be promoted.
static cl::opt<unsigned> ICPMinimumCountThreshold(
    "icp-minimum-count-threshold", cl::init(0), cl::Hidden,
    cl::desc("Minimum absolute count for promotion candidate"));

// Set the maximum number of targets to promote for a single indirect-call
// callsite.
static cl::opt<unsigned>
    MaxNumPromotions("icp-max-prom", cl::init(3), cl::Hidden,
                     cl::desc("Max number of promotions for a single indirect "
                              "call callsite"));

static cl::opt<unsigned> MaxStaticTargetTraversal(
    "icp-max-static-target-traversal", cl::init(64), cl::Hidden,
    cl::desc("Max number of values traversed when collecting static indirect "
             "call targets"));

cl::opt<unsigned> MaxNumVTableAnnotations(
    "icp-max-num-vtables", cl::init(6), cl::Hidden,
    cl::desc("Max number of vtables annotated for a vtable load instruction."));

} // end namespace llvm

bool llvm::collectStaticIndirectCallTargets(
    const CallBase &CB, SmallVectorImpl<Function *> &Targets) {
  assert(!CB.getCalledFunction() && "expected an indirect call");
  Targets.clear();

  auto Fail = [&]() {
    Targets.clear();
    return false;
  };

  SmallPtrSet<Value *, 16> Visited;
  SmallVector<Value *, 16> Worklist;
  unsigned NumEnqueued = 0;

  // Count queued operands, including duplicates, so a PHI with arbitrarily
  // many incoming edges cannot consume unbounded time or worklist storage.
  auto Enqueue = [&](Value *V) {
    if (NumEnqueued >= MaxStaticTargetTraversal)
      return false;
    ++NumEnqueued;
    Worklist.push_back(V);
    return true;
  };

  if (!Enqueue(CB.getCalledOperand()))
    return Fail();

  while (!Worklist.empty()) {
    Value *V = Worklist.pop_back_val();

    // Call promotion compares the indirect callee with the function symbol
    // after converting them to the same pointer type. Do not look through
    // address-space casts that may change the pointer representation, since the
    // function symbol cannot then safely stand in for the original value.
    V = V->stripPointerCastsSameRepresentation();
    if (!Visited.insert(V).second)
      continue;

    if (auto *F = dyn_cast<Function>(V)) {
      if (!is_contained(Targets, F))
        Targets.push_back(F);
      if (Targets.size() > MaxNumPromotions)
        return Fail();
      continue;
    }

    if (auto *SI = dyn_cast<SelectInst>(V)) {
      if (!Enqueue(SI->getFalseValue()) || !Enqueue(SI->getTrueValue()))
        return Fail();
      continue;
    }

    if (auto *PN = dyn_cast<PHINode>(V)) {
      for (Value *Incoming : reverse(PN->incoming_values()))
        if (!Enqueue(Incoming))
          return Fail();
      continue;
    }

    return Fail();
  }

  if (Targets.empty())
    return Fail();
  return true;
}

bool ICallPromotionAnalysis::isPromotionProfitable(uint64_t Count,
                                                   uint64_t TotalCount,
                                                   uint64_t RemainingCount) {
  return Count >= ICPMinimumCountThreshold &&
         Count * 100 >= ICPRemainingPercentThreshold * RemainingCount &&
         Count * 100 >= ICPTotalPercentThreshold * TotalCount;
}

// Indirect-call promotion heuristic. The direct targets are sorted based on
// the count. Stop at the first target that is not promoted. Returns the
// number of candidates deemed profitable.
uint32_t ICallPromotionAnalysis::getProfitablePromotionCandidates(
    const Instruction *Inst, uint64_t TotalCount) {
  LLVM_DEBUG(dbgs() << " \nWork on callsite " << *Inst
                    << " Num_targets: " << ValueDataArray.size() << "\n");

  uint32_t I = 0;
  uint64_t RemainingCount = TotalCount;
  for (; I < MaxNumPromotions && I < ValueDataArray.size(); I++) {
    uint64_t Count = ValueDataArray[I].Count;
    assert(Count <= RemainingCount);
    LLVM_DEBUG(dbgs() << " Candidate " << I << " Count=" << Count
                      << "  Target_func: " << ValueDataArray[I].Value << "\n");

    if (!isPromotionProfitable(Count, TotalCount, RemainingCount)) {
      LLVM_DEBUG(dbgs() << " Not promote: Cold target.\n");
      return I;
    }
    RemainingCount -= Count;
  }
  return I;
}

MutableArrayRef<InstrProfValueData>
ICallPromotionAnalysis::getPromotionCandidatesForInstruction(
    const Instruction *I, uint64_t &TotalCount, uint32_t &NumCandidates,
    unsigned MaxNumValueData) {
  // Use the max of the values specified by -icp-max-prom and the provided
  // MaxNumValueData parameter.
  if (MaxNumPromotions > MaxNumValueData)
    MaxNumValueData = MaxNumPromotions;
  ValueDataArray = getValueProfDataFromInst(*I, IPVK_IndirectCallTarget,
                                            MaxNumValueData, TotalCount);
  if (ValueDataArray.empty()) {
    NumCandidates = 0;
    return MutableArrayRef<InstrProfValueData>();
  }
  NumCandidates = getProfitablePromotionCandidates(I, TotalCount);
  return ValueDataArray;
}
