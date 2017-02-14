///===- LazyMachineBlockFrequencyInfo.h - Lazy Block Frequency -*- C++ -*--===//
///
///                     The LLVM Compiler Infrastructure
///
/// This file is distributed under the University of Illinois Open Source
/// License. See LICENSE.TXT for details.
///
///===---------------------------------------------------------------------===//
/// \file
/// This is an alternative analysis pass to MachineBlockFrequencyInfo.  The
/// difference is that with this pass the block frequencies are not computed
/// when the analysis pass is executed but rather when the BFI result is
/// explicitly requested by the analysis client.
///
///===---------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LAZYMACHINEBLOCKFREQUENCYINFO_H
#define LLVM_ANALYSIS_LAZYMACHINEBLOCKFREQUENCYINFO_H

#include "llvm/Analysis/LazyBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineLoopInfo.h"

namespace llvm {
/// \brief This is an alternative analysis pass to MachineBlockFrequencyInfo.
/// The difference is that with this pass, the block frequencies are not
/// computed when the analysis pass is executed but rather when the BFI result
/// is explicitly requested by the analysis client.
///
/// There are some additional requirements for any client pass that wants to use
/// the analysis:
///
/// 1. The pass needs to initialize dependent passes with:
///
///   INITIALIZE_PASS_DEPENDENCY(LazyMachineBFIPass)
///
/// 2. Similarly, getAnalysisUsage should call:
///
///   LazyMachineBlockFrequencyInfoPass::getLazyMachineBFIAnalysisUsage(AU)
///
/// 3. The computed MachineBFI should be requested with
///    getAnalysis<LazyMachineBlockFrequencyInfoPass>().getBFI() before
///    MachineLoopInfo could be invalidated for example by changing the CFG.
///
/// Note that it is expected that we wouldn't need this functionality for the
/// new PM since with the new PM, analyses are executed on demand.

class LazyMachineBlockFrequencyInfoPass : public MachineFunctionPass {
private:
  /// \brief Machine BPI is an immutable pass, no need to use it lazily.
  LazyBlockFrequencyInfo<MachineFunction, MachineBranchProbabilityInfo,
                         MachineLoopInfo, MachineBlockFrequencyInfo>
      LMBFI;

public:
  static char ID;

  LazyMachineBlockFrequencyInfoPass();

  /// \brief Compute and return the block frequencies.
  MachineBlockFrequencyInfo &getBFI() { return LMBFI.getCalculated(); }

  /// \brief Compute and return the block frequencies.
  const MachineBlockFrequencyInfo &getBFI() const {
    return LMBFI.getCalculated();
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  /// Helper for client passes to set up the analysis usage on behalf of this
  /// pass.
  static void getLazyMachineBFIAnalysisUsage(AnalysisUsage &AU);

  bool runOnMachineFunction(MachineFunction &F) override;
  void releaseMemory() override;
  void print(raw_ostream &OS, const Module *M) const override;
};

/// \brief Helper for client passes to initialize dependent passes for LMBFI.
void initializeLazyMachineBFIPassPass(PassRegistry &Registry);
}
#endif
