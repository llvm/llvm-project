//===- BlockFrequencyInfo.h - Block Frequency Analysis ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Loops should be simplified before this analysis.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_BLOCKFREQUENCYINFO_H
#define LLVM_ANALYSIS_BLOCKFREQUENCYINFO_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/BlockFrequency.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Printable.h"
#include <cstdint>
#include <memory>
#include <optional>

namespace llvm {

class BasicBlock;
class BranchProbabilityInfo;
class LoopInfo;
class Module;
class raw_ostream;
template <class BlockT> class BlockFrequencyInfoImpl;

enum PGOViewCountsType { PGOVCT_None, PGOVCT_Graph, PGOVCT_Text };

/// BlockFrequencyInfo pass uses BlockFrequencyInfoImpl implementation to
/// estimate IR basic block frequencies.
class BlockFrequencyInfo {
  using ImplType = BlockFrequencyInfoImpl<BasicBlock>;

  std::unique_ptr<ImplType> BFI;

public:
  LLVM_ABI BlockFrequencyInfo();
  LLVM_ABI BlockFrequencyInfo(const Function &F,
                              const BranchProbabilityInfo &BPI,
                              const LoopInfo &LI);
  BlockFrequencyInfo(const BlockFrequencyInfo &) = delete;
  BlockFrequencyInfo &operator=(const BlockFrequencyInfo &) = delete;
  LLVM_ABI BlockFrequencyInfo(BlockFrequencyInfo &&Arg);
  LLVM_ABI BlockFrequencyInfo &operator=(BlockFrequencyInfo &&RHS);
  LLVM_ABI ~BlockFrequencyInfo();

  /// Handle invalidation explicitly.
  LLVM_ABI bool invalidate(Function &F, const PreservedAnalyses &PA,
                           FunctionAnalysisManager::Invalidator &);

  LLVM_ABI const Function *getFunction() const;
  LLVM_ABI const BranchProbabilityInfo *getBPI() const;
  LLVM_ABI void view(StringRef = "BlockFrequencyDAGs") const;

  /// getblockFreq - Return block frequency. Return 0 if we don't have the
  /// information. Please note that initial frequency is equal to ENTRY_FREQ. It
  /// means that we should not rely on the value itself, but only on the
  /// comparison to the other block frequencies. We do this to avoid using of
  /// floating points.
  LLVM_ABI BlockFrequency getBlockFreq(const BasicBlock *BB) const;

  /// Returns the estimated profile count of \p BB.
  /// This computes the relative block frequency of \p BB and multiplies it by
  /// the enclosing function's count (if available) and returns the value.
  LLVM_ABI std::optional<uint64_t>
  getBlockProfileCount(const BasicBlock *BB, bool AllowSynthetic = false) const;

  /// Returns the estimated profile count of \p Freq.
  /// This uses the frequency \p Freq and multiplies it by
  /// the enclosing function's count (if available) and returns the value.
  LLVM_ABI std::optional<uint64_t>
  getProfileCountFromFreq(BlockFrequency Freq) const;

  /// Returns true if \p BB is an irreducible loop header
  /// block. Otherwise false.
  LLVM_ABI bool isIrrLoopHeader(const BasicBlock *BB);

  // Set the frequency of the given basic block.
  LLVM_ABI void setBlockFreq(const BasicBlock *BB, BlockFrequency Freq);

  /// Set the frequency of \p ReferenceBB to \p Freq and scale the frequencies
  /// of the blocks in \p BlocksToScale such that their frequencies relative
  /// to \p ReferenceBB remain unchanged.
  LLVM_ABI void
  setBlockFreqAndScale(const BasicBlock *ReferenceBB, BlockFrequency Freq,
                       SmallPtrSetImpl<BasicBlock *> &BlocksToScale);

  /// calculate - compute block frequency info for the given function.
  LLVM_ABI void calculate(const Function &F, const BranchProbabilityInfo &BPI,
                          const LoopInfo &LI);

  LLVM_ABI BlockFrequency getEntryFreq() const;
  LLVM_ABI void releaseMemory();
  LLVM_ABI void print(raw_ostream &OS) const;

  // Compare to the other BFI and verify they match.
  LLVM_ABI void verifyMatch(BlockFrequencyInfo &Other) const;
};

/// Print the block frequency @p Freq relative to the current functions entry
/// frequency. Returns a Printable object that can be piped via `<<` to a
/// `raw_ostream`.
LLVM_ABI Printable printBlockFreq(const BlockFrequencyInfo &BFI,
                                  BlockFrequency Freq);

/// Convenience function equivalent to calling
/// `printBlockFreq(BFI, BFI.getBlocakFreq(&BB))`.
LLVM_ABI Printable printBlockFreq(const BlockFrequencyInfo &BFI,
                                  const BasicBlock &BB);

/// Analysis pass which computes \c BlockFrequencyInfo.
class BlockFrequencyAnalysis
    : public AnalysisInfoMixin<BlockFrequencyAnalysis> {
  friend AnalysisInfoMixin<BlockFrequencyAnalysis>;

  LLVM_ABI static AnalysisKey Key;

public:
  /// Provide the result type for this analysis pass.
  using Result = BlockFrequencyInfo;

  /// Run the analysis pass over a function and produce BFI.
  LLVM_ABI Result run(Function &F, FunctionAnalysisManager &AM);
};

/// Printer pass for the \c BlockFrequencyInfo results.
class BlockFrequencyPrinterPass
    : public PassInfoMixin<BlockFrequencyPrinterPass> {
  raw_ostream &OS;

public:
  explicit BlockFrequencyPrinterPass(raw_ostream &OS) : OS(OS) {}

  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  static bool isRequired() { return true; }
};

/// Legacy analysis pass which computes \c BlockFrequencyInfo.
class LLVM_ABI BlockFrequencyInfoWrapperPass : public FunctionPass {
  BlockFrequencyInfo BFI;

public:
  static char ID;

  BlockFrequencyInfoWrapperPass();
  ~BlockFrequencyInfoWrapperPass() override;

  BlockFrequencyInfo &getBFI() { return BFI; }
  const BlockFrequencyInfo &getBFI() const { return BFI; }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool runOnFunction(Function &F) override;
  void releaseMemory() override;
  void print(raw_ostream &OS, const Module *M) const override;
};

} // end namespace llvm

#endif // LLVM_ANALYSIS_BLOCKFREQUENCYINFO_H
