//===------- AMDGPUMIRUtils.h - Helpers for MIR passes --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Helper functions for MIR passes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUMIRUTILS_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUMIRUTILS_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/CodeGen/MachineBasicBlock.h"

namespace llvm {

class LiveInterval;
class SlotIndexes;
class MachineRegisterInfo;
class SIRegisterInfo;
class MachineDominatorTree;
class MachinePostDominatorTree;

constexpr unsigned RegForVCC = 2;

bool getNonDebugMBBEnd(llvm::MachineBasicBlock::reverse_iterator &BBEnd,
                       llvm::MachineBasicBlock &MBB);

// Check if LI live cross basic blocks, save all touched basic block if is
// local.
bool isLocalLiveInterval(
    const llvm::LiveInterval &LI, llvm::SlotIndexes *Indexes,
    llvm::SmallDenseSet<llvm::MachineBasicBlock *, 2> &TouchedMBBSet);
bool isLocalLiveInterval(const llvm::LiveInterval &LI,
                         llvm::SlotIndexes *Indexes);

bool isSub0Sub1SingleDef(unsigned Reg, const llvm::MachineRegisterInfo &MRI);

using LiveSet = llvm::DenseMap<unsigned, llvm::LaneBitmask>;
void dumpLiveSet(const LiveSet &LiveSet, const llvm::SIRegisterInfo *SIRI);

unsigned getRegSize(unsigned Reg, llvm::LaneBitmask &Mask,
  const llvm::MachineRegisterInfo &MRI,
  const llvm::SIRegisterInfo *SIRI);
void collectLiveSetPressure(const LiveSet &LiveSet,
                            const llvm::MachineRegisterInfo &MRI,
                            const llvm::SIRegisterInfo *SIRI,
                            unsigned &VPressure, unsigned &SPressure);

bool reach_block(llvm::MachineBasicBlock *FromBB,
                 llvm::MachineDominatorTree *DT,
                 llvm::MachinePostDominatorTree *PDT, llvm::MachineLoopInfo *LI,
                 llvm::MachineBasicBlock *ToBB);
}

#endif
