//===- MachineBlockFrequencyInfo.h - MBB Frequency Analysis -----*- C++ -*-===//
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

#ifndef LLVM_CODEGEN_MACHINEBLOCKFREQUENCYINFO_H
#define LLVM_CODEGEN_MACHINEBLOCKFREQUENCYINFO_H

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/Support/BlockFrequency.h"
#include "llvm/Support/Compiler.h"
#include <cstdint>
#include <memory>
#include <optional>

namespace llvm {

template <class BlockT> class BlockFrequencyInfoImpl;
class MachineBasicBlock;
class MachineBranchProbabilityInfo;
class MachineFunction;
class MachineLoopInfo;
class raw_ostream;

/// MachineBlockFrequencyInfo pass uses BlockFrequencyInfoImpl implementation
/// to estimate machine basic block frequencies.
class MachineBlockFrequencyInfo {
  using ImplType = BlockFrequencyInfoImpl<MachineBasicBlock>;
  std::unique_ptr<ImplType> MBFI;

public:
  LLVM_ABI MachineBlockFrequencyInfo(); // Legacy pass manager only.
  LLVM_ABI explicit MachineBlockFrequencyInfo(
      MachineFunction &F, MachineBranchProbabilityInfo &MBPI,
      MachineLoopInfo &MLI);
  LLVM_ABI MachineBlockFrequencyInfo(MachineBlockFrequencyInfo &&);
  LLVM_ABI ~MachineBlockFrequencyInfo();

  /// Handle invalidation explicitly.
  LLVM_ABI bool invalidate(MachineFunction &F, const PreservedAnalyses &PA,
                           MachineFunctionAnalysisManager::Invalidator &);

  /// calculate - compute block frequency info for the given function.
  LLVM_ABI void calculate(const MachineFunction &F,
                          const MachineBranchProbabilityInfo &MBPI,
                          const MachineLoopInfo &MLI);

  LLVM_ABI void print(raw_ostream &OS);

  LLVM_ABI void releaseMemory();

  /// getblockFreq - Return block frequency. Return 0 if we don't have the
  /// information. Please note that initial frequency is equal to 1024. It means
  /// that we should not rely on the value itself, but only on the comparison to
  /// the other block frequencies. We do this to avoid using of floating points.
  /// For example, to get the frequency of a block relative to the entry block,
  /// divide the integral value returned by this function (the
  /// BlockFrequency::getFrequency() value) by getEntryFreq().
  LLVM_ABI BlockFrequency getBlockFreq(const MachineBasicBlock *MBB) const;

  /// Compute the frequency of the block, relative to the entry block.
  /// This API assumes getEntryFreq() is non-zero.
  double getBlockFreqRelativeToEntryBlock(const MachineBasicBlock *MBB) const {
    assert(getEntryFreq() != BlockFrequency(0) &&
           "getEntryFreq() should not return 0 here!");
    return static_cast<double>(getBlockFreq(MBB).getFrequency()) /
           static_cast<double>(getEntryFreq().getFrequency());
  }

  LLVM_ABI std::optional<uint64_t>
  getBlockProfileCount(const MachineBasicBlock *MBB) const;
  LLVM_ABI std::optional<uint64_t>
  getProfileCountFromFreq(BlockFrequency Freq) const;

  LLVM_ABI bool isIrrLoopHeader(const MachineBasicBlock *MBB) const;

  /// incrementally calculate block frequencies when we split edges, to avoid
  /// full CFG traversal.
  LLVM_ABI void onEdgeSplit(const MachineBasicBlock &NewPredecessor,
                            const MachineBasicBlock &NewSuccessor,
                            const MachineBranchProbabilityInfo &MBPI);

  LLVM_ABI const MachineFunction *getFunction() const;
  LLVM_ABI const MachineBranchProbabilityInfo *getMBPI() const;

  /// Pop up a ghostview window with the current block frequency propagation
  /// rendered using dot.
  LLVM_ABI void view(const Twine &Name, bool isSimple = true) const;

  /// Divide a block's BlockFrequency::getFrequency() value by this value to
  /// obtain the entry block - relative frequency of said block.
  LLVM_ABI BlockFrequency getEntryFreq() const;
};

/// Print the block frequency @p Freq relative to the current functions entry
/// frequency. Returns a Printable object that can be piped via `<<` to a
/// `raw_ostream`.
LLVM_ABI Printable printBlockFreq(const MachineBlockFrequencyInfo &MBFI,
                                  BlockFrequency Freq);

/// Convenience function equivalent to calling
/// `printBlockFreq(MBFI, MBFI.getBlockFreq(&MBB))`.
LLVM_ABI Printable printBlockFreq(const MachineBlockFrequencyInfo &MBFI,
                                  const MachineBasicBlock &MBB);

class MachineBlockFrequencyAnalysis
    : public AnalysisInfoMixin<MachineBlockFrequencyAnalysis> {
  friend AnalysisInfoMixin<MachineBlockFrequencyAnalysis>;
  LLVM_ABI static AnalysisKey Key;

public:
  using Result = MachineBlockFrequencyInfo;

  LLVM_ABI Result run(MachineFunction &MF,
                      MachineFunctionAnalysisManager &MFAM);
};

/// Printer pass for the \c MachineBlockFrequencyInfo results.
class MachineBlockFrequencyPrinterPass
    : public PassInfoMixin<MachineBlockFrequencyPrinterPass> {
  raw_ostream &OS;

public:
  explicit MachineBlockFrequencyPrinterPass(raw_ostream &OS) : OS(OS) {}

  LLVM_ABI PreservedAnalyses run(MachineFunction &MF,
                                 MachineFunctionAnalysisManager &MFAM);

  static bool isRequired() { return true; }
};

class LLVM_ABI MachineBlockFrequencyInfoWrapperPass
    : public MachineFunctionPass {
  MachineBlockFrequencyInfo MBFI;

public:
  static char ID;

  MachineBlockFrequencyInfoWrapperPass();

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool runOnMachineFunction(MachineFunction &F) override;

  void releaseMemory() override { MBFI.releaseMemory(); }

  MachineBlockFrequencyInfo &getMBFI() { return MBFI; }

  const MachineBlockFrequencyInfo &getMBFI() const { return MBFI; }
};
} // end namespace llvm

#endif // LLVM_CODEGEN_MACHINEBLOCKFREQUENCYINFO_H
