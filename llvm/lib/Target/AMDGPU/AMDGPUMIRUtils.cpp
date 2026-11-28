//===------- AMDGPUMIRUtils.cpp - Helpers for MIR passes -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Helper functions shared by AMDGPU MIR passes.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUMIRUtils.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"

#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "amdgpu-mir-utils"

using namespace llvm;

namespace {

// LoopInfo maps a block to its innermost loop. Find the outermost loop in the
// nest that contains BB.
const MachineLoop *getOutermostLoop(const MachineLoopInfo *LI,
                                    const MachineBasicBlock *BB) {
  const MachineLoop *L = LI->getLoopFor(BB);
  if (L) {
    while (const MachineLoop *Parent = L->getParentLoop())
      L = Parent;
  }
  return L;
}

bool loopContainsBoth(const MachineLoopInfo *LI, const MachineBasicBlock *BB1,
                      const MachineBasicBlock *BB2) {
  const MachineLoop *L1 = getOutermostLoop(LI, BB1);
  const MachineLoop *L2 = getOutermostLoop(LI, BB2);
  return L1 != nullptr && L1 == L2;
}

// After control-flow lowering a single source region can be split into a
// dominator chain joined by exec-mask manipulation. findNearestCommonDominator
// then returns the lowered join block whose leading instruction consumes the
// exec mask, when the true common dominator is the predecessor that set the
// mask up. Detect that shape and step up to the predecessor.
MachineBasicBlock *fixupExecSplit(MachineBasicBlock *DomB) {
  if (!DomB || DomB->empty())
    return DomB;
  MachineInstr &Lead = *DomB->begin();
  unsigned Opc = Lead.getOpcode();
  bool IsExecSplitLead = Opc == AMDGPU::S_CBRANCH_EXECZ ||
                         Opc == AMDGPU::S_AND_SAVEEXEC_B64 ||
                         Opc == AMDGPU::S_AND_SAVEEXEC_B32;
  if (!IsExecSplitLead || DomB->pred_size() != 1)
    return DomB;
  MachineBasicBlock *Pred = *DomB->pred_begin();
  if (Pred->succ_size() == 1 && (Pred->empty() || !Pred->back().isBranch()))
    return Pred;
  return DomB;
}

} // namespace

namespace llvm {

bool getNonDebugMBBEnd(MachineBasicBlock::reverse_iterator &BBEnd,
                       MachineBasicBlock &MBB) {
  while (BBEnd != MBB.rend() && BBEnd->isDebugInstr())
    ++BBEnd;
  return BBEnd != MBB.rend();
}

unsigned getNumLanesIn32BitReg(bool IsVGPR) {
  const TargetRegisterClass *RC =
      IsVGPR ? &AMDGPU::VGPR_32RegClass : &AMDGPU::SGPR_32RegClass;
  return RC->getLaneMask().getNumLanes();
}

unsigned getRegSize(unsigned Reg, LaneBitmask &Mask,
                    const MachineRegisterInfo &MRI, const SIRegisterInfo *SIRI) {
  const TargetRegisterClass *RC = MRI.getRegClass(Reg);
  unsigned Size = SIRI->getRegSizeInBits(*RC) >> 5;
  if (Mask.any()) {
    // Convert the live lane mask to dwords using this file's lanes-per-dword
    // ratio, rather than treating the raw lane count as dwords (lane masks may
    // be at 16-bit granularity).
    unsigned LanesPerDword = getNumLanesIn32BitReg(SIRI->isVGPR(MRI, Reg));
    unsigned MaskDwords =
        (Mask.getNumLanes() + LanesPerDword - 1) / LanesPerDword;
    if (MaskDwords < Size)
      Size = MaskDwords;
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
    LaneBitmask Mask = LiveIt.second;
    unsigned Size = getRegSize(Reg, Mask, MRI, SIRI);
    if (SIRI->isVGPR(MRI, Reg))
      VPressure += Size;
    else
      SPressure += Size;
  }
}

bool isConsistentMultiDef(Register Reg, const MachineRegisterInfo &MRI,
                          const TargetRegisterInfo *TRI) {
  const TargetRegisterClass *RC = MRI.getRegClass(Reg);
  LaneBitmask FullMask = RC->getLaneMask();
  LaneBitmask CoveredLanes = LaneBitmask::getNone();

  bool HasAnyDef = false;
  for (const MachineOperand &DefMO : MRI.def_operands(Reg)) {
    HasAnyDef = true;
    unsigned SubReg = DefMO.getSubReg();
    // A full-register def should be handled by getUniqueVRegDef instead.
    if (SubReg == 0)
      return false;
    LaneBitmask SubLanes = TRI->getSubRegIndexLaneMask(SubReg);
    // Two defs writing the same lanes: not a clean construction.
    if ((CoveredLanes & SubLanes).any())
      return false;
    CoveredLanes |= SubLanes;
  }

  return HasAnyDef && CoveredLanes == FullMask;
}

MachineBasicBlock *findNonLoopDominator(MachineBasicBlock *BB,
                                        MachineDominatorTree *DT,
                                        MachineLoopInfo *LI) {
  while (LI->getLoopDepth(BB) > 0) {
    MachineDomTreeNode *N = DT->getNode(BB);
    if (N == nullptr)
      return nullptr;
    MachineDomTreeNode *IDom = N->getIDom();
    if (IDom == nullptr)
      return nullptr;
    BB = IDom->getBlock();
  }
  return BB;
}

MachineBasicBlock *
nearestCommonDominator(MachineDominatorTree *DT,
                       SmallPtrSetImpl<MachineBasicBlock *> &Blocks) {
  auto I = Blocks.begin(), E = Blocks.end();
  if (I == E)
    return nullptr;

  MachineBasicBlock *DomB = *(I++);
  while (I != E) {
    MachineBasicBlock *B = *(I++);
    DomB = DT->findNearestCommonDominator(DomB, B);
    if (DomB == nullptr)
      return nullptr;
  }

  return fixupExecSplit(DomB);
}

bool reachBlock(MachineBasicBlock *FromBB, MachineDominatorTree *DT,
                MachinePostDominatorTree *PDT, MachineLoopInfo *LI,
                MachineBasicBlock *ToBB) {
  if (FromBB == ToBB)
    return true;
  if (DT->dominates(FromBB, ToBB))
    return true;
  if (PDT->dominates(ToBB, FromBB))
    return true;
  if (loopContainsBoth(LI, ToBB, FromBB))
    return true;
  return false;
}

bool isSccLiveAt(const MachineInstr &MI, LiveIntervals *LIS) {
  if (!LIS)
    return true;
  const TargetRegisterInfo *TRI = MI.getMF()->getSubtarget().getRegisterInfo();
  LiveRange &LR =
      LIS->getRegUnit(*TRI->regunits(MCRegister::from(AMDGPU::SCC)).begin());
  SlotIndex Idx = LIS->getInstructionIndex(MI);
  return LR.liveAt(Idx);
}

MachineBasicBlock::iterator findOrCreateInsertionPointForSccDef(
    MachineBasicBlock *MBB, MachineBasicBlock::iterator MI,
    const TargetRegisterInfo *TRI, const SIInstrInfo *TII,
    MachineRegisterInfo *MRI, LiveIntervals *LIS) {
  // If SCC is already dead at MI, MI itself is a safe insert point.
  if (!isSccLiveAt(*MI, LIS))
    return MI;

  // Otherwise, scan backwards for a point where SCC is dead.
  for (MachineBasicBlock::reverse_iterator It = MI.getReverse(),
                                           End = MBB->rend();
       It != End; ++It) {
    if (!isSccLiveAt(*It, LIS))
      return It->getIterator();
  }

  // No safe location in the block: save and restore SCC around MI so MI
  // becomes a safe insert point. There is no direct read/write of SCC, so use
  // COPY, which lowers to the appropriate s_cselect/s_cmp sequence.
  //
  //      %SavedSCC = COPY $scc   # save
  //      <-- returned safe insert point
  //      MI
  //      $scc = COPY %SavedSCC   # restore
  Register TmpScc = MRI->createVirtualRegister(&AMDGPU::SReg_32_XM0RegClass);
  DebugLoc DL = MI->getDebugLoc();
  auto CopyFrom =
      BuildMI(*MBB, MI, DL, TII->get(AMDGPU::COPY), TmpScc).addReg(AMDGPU::SCC);
  auto CopyTo = BuildMI(*MBB, std::next(MI->getIterator()), DL,
                        TII->get(AMDGPU::COPY), AMDGPU::SCC)
                    .addReg(TmpScc);

  SlotIndexes *SI = LIS->getSlotIndexes();
  SI->insertMachineInstrInMaps(*CopyFrom);
  SI->insertMachineInstrInMaps(*CopyTo);
  LiveRange &LR =
      LIS->getRegUnit(*TRI->regunits(MCRegister::from(AMDGPU::SCC)).begin());
  auto OldSegment = *LR.getSegmentContaining(LIS->getInstructionIndex(*MI));
  LiveRange::Segment NewSegA(OldSegment.start,
                             SI->getInstructionIndex(*CopyFrom).getRegSlot(),
                             OldSegment.valno);
  LiveRange::Segment NewSegB(SI->getInstructionIndex(*CopyTo).getRegSlot(),
                             OldSegment.end, OldSegment.valno);
  LR.removeSegment(OldSegment);
  LR.addSegment(NewSegA);
  LR.addSegment(NewSegB);

  return MI;
}

void dumpLiveSet(const LiveSet &LiveSet, const SIRegisterInfo *SIRI) {
  dbgs() << "\n live set: \n";
  for (auto It : LiveSet) {
    dbgs() << printReg(It.first, SIRI);
    if (It.second.any())
      dbgs() << " mask:" << It.second.getAsInteger();
    dbgs() << "\n";
  }
}

} // namespace llvm
