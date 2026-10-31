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
#include "AnalysisOptions.h"
#include "llvm/IR/Instruction.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "pgo-icall-prom-analysis"

unsigned llvm::getMaxNumVTableAnnotations(const LLVMContext &C) {
  return getAnalysisOptions(C).icp_max_num_vtables;
}

bool ICallPromotionAnalysis::isPromotionProfitable(const AnalysisOptions &Opts,
                                                   uint64_t Count,
                                                   uint64_t TotalCount,
                                                   uint64_t RemainingCount) {
  return Count >= Opts.icp_minimum_count_threshold &&
         Count * 100 >= Opts.icp_remaining_percent_threshold * RemainingCount &&
         Count * 100 >= Opts.icp_total_percent_threshold * TotalCount;
}

// Indirect-call promotion heuristic. The direct targets are sorted based on
// the count. Stop at the first target that is not promoted. Returns the
// number of candidates deemed profitable.
uint32_t ICallPromotionAnalysis::getProfitablePromotionCandidates(
    const AnalysisOptions &Opts, const Instruction *Inst, uint64_t TotalCount) {
  LLVM_DEBUG(dbgs() << " \nWork on callsite " << *Inst
                    << " Num_targets: " << ValueDataArray.size() << "\n");

  uint32_t I = 0;
  uint64_t RemainingCount = TotalCount;
  for (; I < Opts.icp_max_prom && I < ValueDataArray.size(); I++) {
    uint64_t Count = ValueDataArray[I].Count;
    assert(Count <= RemainingCount);
    LLVM_DEBUG(dbgs() << " Candidate " << I << " Count=" << Count
                      << "  Target_func: " << ValueDataArray[I].Value << "\n");

    if (!isPromotionProfitable(Opts, Count, TotalCount, RemainingCount)) {
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
  const AnalysisOptions &Opts = getAnalysisOptions(*I->getFunction());
  // Use the max of the values specified by -icp-max-prom and the provided
  // MaxNumValueData parameter.
  if (Opts.icp_max_prom > MaxNumValueData)
    MaxNumValueData = Opts.icp_max_prom;
  ValueDataArray = getValueProfDataFromInst(*I, IPVK_IndirectCallTarget,
                                            MaxNumValueData, TotalCount);
  if (ValueDataArray.empty()) {
    NumCandidates = 0;
    return MutableArrayRef<InstrProfValueData>();
  }
  NumCandidates = getProfitablePromotionCandidates(Opts, I, TotalCount);
  return ValueDataArray;
}
