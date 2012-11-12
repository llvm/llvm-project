//========-------- BlockFrequencyInfo.h - Block Frequency Analysis -------========//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Loops should be simplified before this analysis.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_BLOCKFREQUENCYINFO_H
#define LLVM_ANALYSIS_BLOCKFREQUENCYINFO_H

#include "llvm/Pass.h"
#include "llvm/Support/BlockFrequency.h"
#include <climits>

namespace llvm {

class BranchProbabilityInfo;
template<class BlockT, class FunctionT, class BranchProbInfoT>
class BlockFrequencyImpl;

/// BlockFrequencyInfo pass uses BlockFrequencyImpl implementation to estimate
/// IR basic block frequencies.
class BlockFrequencyInfo : public FunctionPass {

  BlockFrequencyImpl<BasicBlock, Function, BranchProbabilityInfo> *BFI;

public:
  static char ID;

  BlockFrequencyInfo();

  ~BlockFrequencyInfo();

  void getAnalysisUsage(AnalysisUsage &AU) const;

  bool runOnFunction(Function &F);
  void print(raw_ostream &O, const Module *M) const;

  /// getblockFreq - Return block frequency. Return 0 if we don't have the
  /// information. Please note that initial frequency is equal to 1024. It means
  /// that we should not rely on the value itself, but only on the comparison to
  /// the other block frequencies. We do this to avoid using of floating points.
  ///
  BlockFrequency getBlockFreq(const BasicBlock *BB) const;
};

}

#endif
