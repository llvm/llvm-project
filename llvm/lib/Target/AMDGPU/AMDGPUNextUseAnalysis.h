//===---------------------- AMDGPUNextUseAnalysis.h  ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Next Use Analysis.
// For each register it goes over all uses and returns the estimated distance of
// the nearest use. This will be used for selecting which registers to spill
// before register allocation.
//
// This is based on ideas from the paper:
// "Register Spilling and Live-Range Splitting for SSA-Form Programs"
// Matthias Braun and Sebastian Hack, CC'09
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUNEXTUSEANALYSIS_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUNEXTUSEANALYSIS_H

#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include <optional>

using namespace llvm;

class AMDGPUNextUseAnalysis {
  const SIRegisterInfo *TRI = nullptr;
  const SIInstrInfo *TII = nullptr;
  const MachineLoopInfo *MLI = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  /// Instruction to instruction-id map.
  DenseMap<MachineInstr *, double> InstrToId;
  /// Returns MI's instruction ID. It renumbers (part of) the BB if MI is not
  /// found in the map.
  double getInstrId(MachineInstr *MI) {
    auto It = InstrToId.find(MI);
    if (It != InstrToId.end())
      return It->second;
    // Renumber the MBB.
    // TODO: Renumber from MI onwards.
    auto *MBB = MI->getParent();
    double Id = 0.0;
    for (auto &MBBMI : *MBB)
      InstrToId[&MBBMI] = Id++;
    return InstrToId.find(MI)->second;
  }
  /// [FromMBB, ToMBB] to shortest distance map.
  DenseMap<std::pair<MachineBasicBlock *, MachineBasicBlock *>, double>
      ShortestPathTable;
  /// We assume an approximate trip count of 1000 for all loops.
  static constexpr const double LoopWeight = 1000.0;
  bool isBackedge(MachineBasicBlock *From, MachineBasicBlock *To) const;
  double getShortestPath(MachineBasicBlock *From, MachineBasicBlock *To);
  /// Goes over all MBB pairs in \p MF, calculates the shortest path between
  /// them and fills in \p ShortestPathTable.
  void calculateShortestPaths(MachineFunction &MF);
  /// If the path from \p MI to \p UseMI does not cross any loops, then this
  /// \returns the shortest instruction distance between them.
  double calculateShortestDistance(MachineInstr *MI, MachineInstr *UseMI);
  /// /Returns the shortest distance between a given basic block \p CurMBB and
  /// its closest exiting latch of \p CurLoop.
  std::pair<double, MachineBasicBlock *>
  getShortestDistanceToExitingLatch(MachineBasicBlock *CurMBB,
                                    MachineLoop *CurLoop);
  /// Helper function for calculating the minimum instruction distance from the
  /// outer loop header to the outer loop latch.
  std::pair<double, MachineBasicBlock *> getNestedLoopDistanceAndExitingLatch(
      MachineBasicBlock *CurMBB, MachineBasicBlock *UseMBB,
      bool IsUseOutsideOfTheCurLoopNest = false,
      bool IsUseInParentLoop = false);
  /// Given \p CurMI in a loop and \p UseMI outside the loop, this function
  /// returns the minimum instruction path between \p CurMI and \p UseMI.
  /// Please note that since \p CurMI is in a loop we don't care about the
  /// exact position of the instruction in the block because we are making a
  /// rough estimate of the dynamic instruction path length, given that the loop
  /// iterates multiple times.
  double calculateCurLoopDistance(Register DefReg, MachineInstr *CurMI,
                                  MachineInstr *UseMI);
  /// \Returns the shortest path distance from \p CurMI to the end of the loop
  /// latch plus the distance from the top of the loop header to the PHI use.
  double calculateBackedgeDistance(MachineInstr *CurMI, MachineInstr *UseMI);
  /// \Returns true if the use of \p DefReg (\p UseMI) is a PHI in the loop
  /// header, i.e., DefReg is flowing through the back-edge.
  bool isIncomingValFromBackedge(MachineInstr *CurMI, MachineInstr *UseMI,
                                 Register DefReg) const;

  void dumpShortestPaths() const;

  void printAllDistances(MachineFunction &);

  void clearTables() {
    InstrToId.clear();
    ShortestPathTable.clear();
  }

public:
  AMDGPUNextUseAnalysis() = default;

  ~AMDGPUNextUseAnalysis() { clearTables(); }

  bool run(MachineFunction &, const MachineLoopInfo *);

  /// \Returns the next-use distance for \p DefReg.
  std::optional<double> getNextUseDistance(Register DefReg);

  std::optional<double> getNextUseDistance(Register DefReg, MachineInstr *CurMI,
                                           SmallVector<MachineInstr *> &Uses);

  /// Helper function that finds the shortest instruction path in \p CurMMB's
  /// loop that includes \p CurMBB and starts from the loop header and ends at
  /// the earliest loop latch. \Returns the path cost and the earliest latch
  /// MBB.
  std::pair<double, MachineBasicBlock *>
  getLoopDistanceAndExitingLatch(MachineBasicBlock *CurMBB);
  /// Calculates the shortest distance and caches it.
  double getShortestDistance(MachineBasicBlock *FromMBB,
                             MachineBasicBlock *ToMBB) {
    auto It = ShortestPathTable.find({FromMBB, ToMBB});
    if (It != ShortestPathTable.end())
      return It->second;
    double ShortestDistance = 0.0;
    ShortestDistance = getShortestPath(FromMBB, ToMBB);
    ShortestPathTable[std::make_pair(FromMBB, ToMBB)] = ShortestDistance;
    return ShortestDistance;
  }
};

class AMDGPUNextUseAnalysisPass : public MachineFunctionPass {
  const MachineLoopInfo *MLI = nullptr;

public:
  static char ID;

  AMDGPUNextUseAnalysisPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &) override;

  StringRef getPassName() const override { return "Next Use Analysis"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveVariablesWrapperPass>();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addPreserved<LiveVariablesWrapperPass>();
    AU.addPreserved<MachineLoopInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUNEXTUSEANALYSIS_H
