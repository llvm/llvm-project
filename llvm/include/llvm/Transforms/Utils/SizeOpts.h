//===- llvm/Transforms/Utils/SizeOpts.h - size optimization -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains some shared code size optimization related code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_SIZEOPTS_H
#define LLVM_TRANSFORMS_UTILS_SIZEOPTS_H

#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Support/CommandLine.h"

extern llvm::cl::opt<bool> EnablePGSO;
extern llvm::cl::opt<bool> PGSOLargeWorkingSetSizeOnly;
extern llvm::cl::opt<bool> PGSOIRPassOrTestOnly;
extern llvm::cl::opt<bool> PGSOColdCodeOnly;
extern llvm::cl::opt<bool> ForcePGSO;
extern llvm::cl::opt<int> PgsoCutoffInstrProf;
extern llvm::cl::opt<int> PgsoCutoffSampleProf;

namespace llvm {

class BasicBlock;
class BlockFrequencyInfo;
class Function;
class ProfileSummaryInfo;

enum class PGSOQueryType {
  IRPass, // A query call from an IR-level transform pass.
  Test,   // A query call from a unit test.
  Other,  // Others.
};

template<typename AdapterT, typename FuncT, typename BFIT>
bool shouldFuncOptimizeForSizeImpl(const FuncT *F, ProfileSummaryInfo *PSI,
                                   BFIT *BFI, PGSOQueryType QueryType) {
  assert(F);
  if (!PSI || !BFI || !PSI->hasProfileSummary())
    return false;
  if (ForcePGSO)
    return true;
  if (!EnablePGSO)
    return false;
  // Temporarily enable size optimizations only for the IR pass or test query
  // sites for gradual commit/rollout. This is to be removed later.
  if (PGSOIRPassOrTestOnly && !(QueryType == PGSOQueryType::IRPass ||
                                QueryType == PGSOQueryType::Test))
    return false;
  if (PGSOColdCodeOnly ||
      (PGSOLargeWorkingSetSizeOnly && !PSI->hasLargeWorkingSetSize())) {
    // Even if the working set size isn't large, size-optimize cold code.
    return AdapterT::isFunctionColdInCallGraph(F, PSI, *BFI);
  }
  return !AdapterT::isFunctionHotInCallGraphNthPercentile(
      PSI->hasSampleProfile() ? PgsoCutoffSampleProf : PgsoCutoffInstrProf,
      F, PSI, *BFI);
}

template<typename AdapterT, typename BlockT, typename BFIT>
bool shouldOptimizeForSizeImpl(const BlockT *BB, ProfileSummaryInfo *PSI,
                               BFIT *BFI, PGSOQueryType QueryType) {
  assert(BB);
  if (!PSI || !BFI || !PSI->hasProfileSummary())
    return false;
  if (ForcePGSO)
    return true;
  if (!EnablePGSO)
    return false;
  // Temporarily enable size optimizations only for the IR pass or test query
  // sites for gradual commit/rollout. This is to be removed later.
  if (PGSOIRPassOrTestOnly && !(QueryType == PGSOQueryType::IRPass ||
                                QueryType == PGSOQueryType::Test))
    return false;
  if (PGSOColdCodeOnly ||
      (PGSOLargeWorkingSetSizeOnly && !PSI->hasLargeWorkingSetSize())) {
    // Even if the working set size isn't large, size-optimize cold code.
    return AdapterT::isColdBlock(BB, PSI, BFI);
  }
  return !AdapterT::isHotBlockNthPercentile(
      PSI->hasSampleProfile() ? PgsoCutoffSampleProf : PgsoCutoffInstrProf,
      BB, PSI, BFI);
}

/// Returns true if function \p F is suggested to be size-optimized based on the
/// profile.
bool shouldOptimizeForSize(const Function *F, ProfileSummaryInfo *PSI,
                           BlockFrequencyInfo *BFI,
                           PGSOQueryType QueryType = PGSOQueryType::Other);

/// Returns true if basic block \p BB is suggested to be size-optimized based on
/// the profile.
bool shouldOptimizeForSize(const BasicBlock *BB, ProfileSummaryInfo *PSI,
                           BlockFrequencyInfo *BFI,
                           PGSOQueryType QueryType = PGSOQueryType::Other);

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_SIZEOPTS_H
