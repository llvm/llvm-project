//===------- AMDGPUMIRUtils.cpp - Helpers for MIR passes ------------------===//
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

#include "AMDGPUMIRUtils.h"
#include "SIRegisterInfo.h"
#include "SIInstrInfo.h"

#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachinePostDominators.h"

#define DEBUG_TYPE "xb-mir-util"
using namespace llvm;

namespace llvm {
bool getNonDebugMBBEnd(MachineBasicBlock::reverse_iterator &BBEnd,
                       MachineBasicBlock &MBB) {
  // R.End doesn't point to the boundary instruction.
  // Skip Debug instr.
  while (BBEnd != MBB.rend() && BBEnd->isDebugInstr())
    BBEnd++;
  return BBEnd != MBB.rend();
}
} // namespace llvm

namespace {
bool isLocalSegment(const LiveRange::Segment *Seg, SlotIndexes *Indexes,
                    SmallDenseSet<MachineBasicBlock *, 2> &TouchedMBBSet) {
  MachineInstr *StartMI = Indexes->getInstructionFromIndex(Seg->start);
  MachineInstr *EndMI = Indexes->getInstructionFromIndex(Seg->end);
  // Treat non inst as not local.
  if (!StartMI || !EndMI)
    return false;
  // is local when parent MBB the same.
  bool IsSameMBB = StartMI->getParent() == EndMI->getParent();
  if (!IsSameMBB)
    return false;
  // Collect touched MBB.
  MachineBasicBlock *MBB = StartMI->getParent();
  TouchedMBBSet.insert(MBB);
  return true;
}

bool isLocalLiveRange(const LiveRange *Range, SlotIndexes *Indexes,
                      SmallDenseSet<MachineBasicBlock *, 2> &TouchedMBBSet) {
  for (const LiveRange::Segment &Seg : Range->segments) {
    if (!isLocalSegment(&Seg, Indexes, TouchedMBBSet))
      return false;
  }
  return true;
}

bool isLocalSegment(const LiveRange::Segment *Seg, SlotIndexes *Indexes) {
  MachineInstr *StartMI = Indexes->getInstructionFromIndex(Seg->start);
  MachineInstr *EndMI = Indexes->getInstructionFromIndex(Seg->end);
  // Treat non inst as not local.
  if (!StartMI || !EndMI)
    return false;
  // is local when parent MBB the same.
  return StartMI->getParent() == EndMI->getParent();
}

bool isLocalLiveRange(const LiveRange *Range, SlotIndexes *Indexes) {
  for (const LiveRange::Segment &Seg : Range->segments) {
    if (!isLocalSegment(&Seg, Indexes))
      return false;
  }
  return true;
}

} // namespace

// In case like float4 v, v.x used and defined in one block, v.y used and define
// in another block, one live interval could touch more than one MBB.
// TouchedMBBSet is used for scheduling where local live interval could cross
// multiple regions, need to calculate livereg for each region inside touched
// MBB.
bool llvm::isLocalLiveInterval(
    const LiveInterval &LI, SlotIndexes *Indexes,
    SmallDenseSet<MachineBasicBlock *, 2> &TouchedMBBSet) {
  if (LI.hasSubRanges()) {
    for (const auto &S : LI.subranges()) {
      if (!isLocalLiveRange(&S, Indexes, TouchedMBBSet))
        return false;
    }
  }
  return isLocalLiveRange(&LI, Indexes, TouchedMBBSet);
}

bool llvm::isLocalLiveInterval(const LiveInterval &LI, SlotIndexes *Indexes) {
  if (LI.hasSubRanges()) {
    for (const auto &S : LI.subranges()) {
      if (!isLocalLiveRange(&S, Indexes))
        return false;
    }
  }
  return isLocalLiveRange(&LI, Indexes);
}

void llvm::dumpLiveSet(const LiveSet &LiveSet, const SIRegisterInfo *SIRI) {

  dbgs() << "\n live set: \n";
  for (auto It : LiveSet) {
    int Reg = It.first;
    dbgs() << printReg(Reg, SIRI);
    if (It.second.any()) {
      dbgs() << " mask:" << It.second.getAsInteger();
    }
    dbgs() << "\n";
  }
}

namespace llvm {
unsigned getRegSize(unsigned Reg, llvm::LaneBitmask &Mask,
                    const llvm::MachineRegisterInfo &MRI,
                    const llvm::SIRegisterInfo *SIRI) {
  unsigned Size = SIRI->getRegSizeInBits(*MRI.getRegClass(Reg));
  Size >>= 5;
  if (Mask.any()) {
    if (unsigned MaskSize = Mask.getNumLanes()) {
      if (MaskSize < Size)
        Size = MaskSize;
    }
  }
  return Size;
}

void collectLiveSetPressure(const LiveSet &LiveSet,
                            const MachineRegisterInfo &MRI,
                            const SIRegisterInfo *SIRI, unsigned &VPressure,
                            unsigned &SPressure) {
  VPressure = 0;
  SPressure = 0;
  for (auto LiveIt : LiveSet) {
    unsigned Reg = LiveIt.first;
    unsigned Size = getRegSize(Reg, LiveIt.second, MRI, SIRI);
    if (SIRI->isVGPR(MRI, Reg)) {
      VPressure += Size;
    } else {
      SPressure += Size;
    }
  }
}

bool isSub0Sub1SingleDef(unsigned Reg, const MachineRegisterInfo &MRI) {
  // Support multi def for pattern of pointer:
  // undef_ %808.sub0:sgpr_64 = COPY killed %795:sgpr_32
  // %808.sub1:sgpr_64 = S_MOV_B32 0
  bool HasSub0 = false;
  bool HasSub1 = false;
  for (MachineOperand &UserDefMO : MRI.def_operands(Reg)) {
    if (unsigned SubReg = UserDefMO.getSubReg()) {
      bool IsSingleSubReg = false;
      switch (SubReg) {
      default:
        break;
      case AMDGPU::sub0:
        if (!HasSub0) {
          HasSub0 = true;
          IsSingleSubReg = true;
        }
        break;
      case AMDGPU::sub1:
        if (!HasSub1) {
          HasSub1 = true;
          IsSingleSubReg = true;
        }
        break;
      }
      if (!IsSingleSubReg) {
        HasSub0 = false;
        break;
      }
    } else {
      HasSub0 = false;
      break;
    }
  }

  return (HasSub0 && HasSub1);
}

bool reach_block(MachineBasicBlock *FromBB, MachineDominatorTree *DT,
                 MachinePostDominatorTree *PDT, MachineLoopInfo *LI,
                 MachineBasicBlock *ToBB) {
  if (FromBB == ToBB) {
    return true;
  }

  if (DT->dominates(FromBB, ToBB)) {
    return true;
  }

  if (PDT->dominates(ToBB, FromBB)) {
    return true;
  }

  if (loopContainsBoth(LI, ToBB, FromBB)) {
    return true;
  }
  // TODO: cover case hotBB in loop,
  //       one block in that loop dom BB or
  //       BB post dom one block in that loop.
  return false;
}
} // namespace llvm
