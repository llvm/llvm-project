//===-- AMDGPUVMemDegreeDAG.h - Build degree about VMem on DAG --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Build degree about VMem to help balance latency and pressure inside a
/// block.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <vector>
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/ScheduleDAG.h"  // For SUnit.

namespace llvm {
class MachineBasicBlock;
class SUnit;
class SIInstrInfo;
class MachineInstr;

class SimpleDAG {
public:
  SimpleDAG(llvm::MachineBasicBlock &MBB, const llvm::SIInstrInfo *TII)
      : SIII(TII), MBB(MBB) {}
  std::vector<llvm::SUnit> SUnits;
  // InstrInfo.
  const llvm::SIInstrInfo *SIII;
  llvm::DenseMap<llvm::MachineInstr *, llvm::SUnit *> MISUnitMap;
  llvm::DenseMap<llvm::SUnit *, llvm::MachineInstr *> SUnitMIMap;
  llvm::MachineBasicBlock &MBB;
  void build();

private:
  void initNodes();
  void addDependence();
  void addCtrlDep();
};


// Collect height/depth for high latency mem ld, which only update height/depth
// when cross high latency mem ld. Call the height/depth as VMem degree here.
// The rule is sample and its user should has different degree.
// For example
// a = sample     // a has depth 0, height 3
// b = sample a   // b has depth 1, height 2
// c = sample c   // c has depth 2, height 1
//   user of c    // user of c has depth 2, height 0
//
// For the purpose of in block reorder/remat, nothing will move/clone cross the
// block. So do this after cross blk remat? In the middle of cross block remat
// to help reach target when only move things cross blk cannot reach the target.
// Reorder at the beginning? No pressure at that time? After get pressure, might
// need to update max pressure.

class VMemDegreeDAG {
public:
  VMemDegreeDAG(std::vector<llvm::SUnit> &Units,
              const llvm::SIInstrInfo *TII)
      : SUnits(Units), SIII(TII) {}
  std::vector<llvm::SUnit> &SUnits;
  // InstrInfo.
  const llvm::SIInstrInfo *SIII;
  void build();


  bool isHighLatency(const llvm::SUnit *SU) const;
  bool isHighLatency(const llvm::MachineInstr *MI) const;
  // height/depth based on Long latency inst.
  std::vector<unsigned> VMemDataHeight;
  std::vector<unsigned> VMemDataDepth;
  // Full height/depth count non-data dependent too.
  std::vector<unsigned> VMemFullHeight;
  std::vector<unsigned> VMemFullDepth;
  llvm::SmallVector<llvm::SUnit *, 16> VMemSUs;
  llvm::SmallVector<llvm::SmallVector<llvm::SUnit *, 8>, 16> GroupedVMemSUs;
  llvm::SmallVector<llvm::SmallVector<llvm::SUnit *, 8>, 16> GroupedVMemSUsByDepth;


  void dump();

private:
  static constexpr unsigned kNoReg = -1;


  std::pair<unsigned, unsigned> buildVMemDepthHeight(std::vector<unsigned> &VMemHeight,
                            std::vector<unsigned> &VMemDepth, bool bDataOnly);
  // Compute vmem height/depth.
  void buildVMemDepthHeight();
  void buildVMemDataDepthHeight();
  void groupVmemSUnits();

};



// Split block based on vmem depth.
void buildVMemDepth(llvm::MachineBasicBlock &MBB, llvm::VMemDegreeDAG &dag);

}

