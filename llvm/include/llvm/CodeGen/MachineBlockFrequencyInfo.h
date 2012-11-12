//====----- MachineBlockFrequencyInfo.h - MachineBlock Frequency Analysis ----====//
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

#ifndef LLVM_CODEGEN_MACHINEBLOCKFREQUENCYINFO_H
#define LLVM_CODEGEN_MACHINEBLOCKFREQUENCYINFO_H

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/BlockFrequency.h"
#include <climits>

namespace llvm {

class MachineBasicBlock;
class MachineBranchProbabilityInfo;
template<class BlockT, class FunctionT, class BranchProbInfoT>
class BlockFrequencyImpl;

/// MachineBlockFrequencyInfo pass uses BlockFrequencyImpl implementation to estimate
/// machine basic block frequencies.
class MachineBlockFrequencyInfo : public MachineFunctionPass {

  BlockFrequencyImpl<MachineBasicBlock, MachineFunction,
                     MachineBranchProbabilityInfo> *MBFI;

public:
  static char ID;

  MachineBlockFrequencyInfo();

  ~MachineBlockFrequencyInfo();

  void getAnalysisUsage(AnalysisUsage &AU) const;

  bool runOnMachineFunction(MachineFunction &F);

  /// getblockFreq - Return block frequency. Return 0 if we don't have the
  /// information. Please note that initial frequency is equal to 1024. It means
  /// that we should not rely on the value itself, but only on the comparison to
  /// the other block frequencies. We do this to avoid using of floating points.
  ///
  BlockFrequency getBlockFreq(const MachineBasicBlock *MBB) const;
};

}

#endif
