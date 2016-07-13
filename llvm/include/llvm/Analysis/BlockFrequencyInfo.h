//===- BlockFrequencyInfo.h - Block Frequency Analysis ----------*- C++ -*-===//
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

#include "llvm/ADT/Optional.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/BlockFrequency.h"
#include <climits>

namespace llvm {

class BranchProbabilityInfo;
class LoopInfo;
template <class BlockT> class BlockFrequencyInfoImpl;

/// BlockFrequencyInfo pass uses BlockFrequencyInfoImpl implementation to
/// estimate IR basic block frequencies.
class BlockFrequencyInfo {
  typedef BlockFrequencyInfoImpl<BasicBlock> ImplType;
  std::unique_ptr<ImplType> BFI;

  void operator=(const BlockFrequencyInfo &) = delete;
  BlockFrequencyInfo(const BlockFrequencyInfo &) = delete;

public:
  BlockFrequencyInfo();
  BlockFrequencyInfo(const Function &F, const BranchProbabilityInfo &BPI,
                     const LoopInfo &LI);
  BlockFrequencyInfo(BlockFrequencyInfo &&Arg);

  BlockFrequencyInfo &operator=(BlockFrequencyInfo &&RHS);

  ~BlockFrequencyInfo();

  const Function *getFunction() const;
  const BranchProbabilityInfo *getBPI() const;
  void view() const;

  /// getblockFreq - Return block frequency. Return 0 if we don't have the
  /// information. Please note that initial frequency is equal to ENTRY_FREQ. It
  /// means that we should not rely on the value itself, but only on the
  /// comparison to the other block frequencies. We do this to avoid using of
  /// floating points.
  BlockFrequency getBlockFreq(const BasicBlock *BB) const;

  /// \brief Returns the estimated profile count of \p BB.
  /// This computes the relative block frequency of \p BB and multiplies it by
  /// the enclosing function's count (if available) and returns the value.
  Optional<uint64_t> getBlockProfileCount(const BasicBlock *BB) const;

  // Set the frequency of the given basic block.
  void setBlockFreq(const BasicBlock *BB, uint64_t Freq);

  /// calculate - compute block frequency info for the given function.
  void calculate(const Function &F, const BranchProbabilityInfo &BPI,
                 const LoopInfo &LI);

  // Print the block frequency Freq to OS using the current functions entry
  // frequency to convert freq into a relative decimal form.
  raw_ostream &printBlockFreq(raw_ostream &OS, const BlockFrequency Freq) const;

  // Convenience method that attempts to look up the frequency associated with
  // BB and print it to OS.
  raw_ostream &printBlockFreq(raw_ostream &OS, const BasicBlock *BB) const;

  uint64_t getEntryFreq() const;
  void releaseMemory();
  void print(raw_ostream &OS) const;
};

/// \brief Analysis pass which computes \c BlockFrequencyInfo.
class BlockFrequencyAnalysis
    : public AnalysisInfoMixin<BlockFrequencyAnalysis> {
  friend AnalysisInfoMixin<BlockFrequencyAnalysis>;
  static char PassID;

public:
  /// \brief Provide the result typedef for this analysis pass.
  typedef BlockFrequencyInfo Result;

  /// \brief Run the analysis pass over a function and produce BFI.
  Result run(Function &F, AnalysisManager<Function> &AM);
};

/// \brief Printer pass for the \c BlockFrequencyInfo results.
class BlockFrequencyPrinterPass
    : public PassInfoMixin<BlockFrequencyPrinterPass> {
  raw_ostream &OS;

public:
  explicit BlockFrequencyPrinterPass(raw_ostream &OS) : OS(OS) {}
  PreservedAnalyses run(Function &F, AnalysisManager<Function> &AM);
};

/// \brief Legacy analysis pass which computes \c BlockFrequencyInfo.
class BlockFrequencyInfoWrapperPass : public FunctionPass {
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

}

#endif
