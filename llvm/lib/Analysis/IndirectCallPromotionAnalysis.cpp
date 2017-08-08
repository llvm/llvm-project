//===-- IndirectCallPromotionAnalysis.cpp - Find promotion candidates ===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/Analysis/IndirectCallSiteVisitor.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/Debug.h"
#include <string>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "pgo-icall-prom-analysis"

// The minimum call count for the direct-call target to be considered as the
// promotion candidate.
static cl::opt<unsigned>
    ICPCountThreshold("icp-count-threshold", cl::Hidden, cl::ZeroOrMore,
                      cl::init(1000),
                      cl::desc("The minimum count to the direct call target "
                               "for the promotion"));

// The percent threshold for the direct-call target (this call site vs the
// total call count) for it to be considered as the promotion target.
static cl::opt<unsigned>
    ICPPercentThreshold("icp-percent-threshold", cl::init(30), cl::Hidden,
                        cl::ZeroOrMore,
                        cl::desc("The percentage threshold for the promotion"));

// Set the maximum number of targets to promote for a single indirect-call
// callsite.
static cl::opt<unsigned>
    MaxNumPromotions("icp-max-prom", cl::init(2), cl::Hidden, cl::ZeroOrMore,
                     cl::desc("Max number of promotions for a single indirect "
                              "call callsite"));

ICallPromotionAnalysis::ICallPromotionAnalysis() {
  ValueDataArray = llvm::make_unique<InstrProfValueData[]>(MaxNumPromotions);
}

bool ICallPromotionAnalysis::isPromotionProfitable(uint64_t Count,
                                                   uint64_t TotalCount) {
  if (Count < ICPCountThreshold)
    return false;

  unsigned Percentage = (Count * 100) / TotalCount;
  return (Percentage >= ICPPercentThreshold);
}

// Indirect-call promotion heuristic. The direct targets are sorted based on
// the count. Stop at the first target that is not promoted. Returns the
// number of candidates deemed profitable.
uint32_t ICallPromotionAnalysis::getProfitablePromotionCandidates(
    const Instruction *Inst, uint32_t NumVals, uint64_t TotalCount) {
  ArrayRef<InstrProfValueData> ValueDataRef(ValueDataArray.get(), NumVals);

  DEBUG(dbgs() << " \nWork on callsite " << *Inst << " Num_targets: " << NumVals
               << "\n");

  uint32_t I = 0;
  for (; I < MaxNumPromotions && I < NumVals; I++) {
    uint64_t Count = ValueDataRef[I].Count;
    assert(Count <= TotalCount);
    DEBUG(dbgs() << " Candidate " << I << " Count=" << Count
                 << "  Target_func: " << ValueDataRef[I].Value << "\n");

    if (!isPromotionProfitable(Count, TotalCount)) {
      DEBUG(dbgs() << " Not promote: Cold target.\n");
      return I;
    }
    TotalCount -= Count;
  }
  return I;
}

ArrayRef<InstrProfValueData>
ICallPromotionAnalysis::getPromotionCandidatesForInstruction(
    const Instruction *I, uint32_t &NumVals, uint64_t &TotalCount,
    uint32_t &NumCandidates) {
  bool Res =
      getValueProfDataFromInst(*I, IPVK_IndirectCallTarget, MaxNumPromotions,
                               ValueDataArray.get(), NumVals, TotalCount);
  if (!Res) {
    NumCandidates = 0;
    return ArrayRef<InstrProfValueData>();
  }
  NumCandidates = getProfitablePromotionCandidates(I, NumVals, TotalCount);
  return ArrayRef<InstrProfValueData>(ValueDataArray.get(), NumVals);
}
