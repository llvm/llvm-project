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
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"

#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

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

// LoopInfo contains a mapping from basic block to the innermost loop. Find
// the outermost loop in the loop nest that contains BB.
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

} // namespace

namespace llvm {

bool isSccLiveAt(const MachineInstr &MI, LiveIntervals *LIS) {
  if (!LIS)
    return true;
  const TargetRegisterInfo *TRI = MI.getMF()->getSubtarget().getRegisterInfo();
  LiveRange &LR =
      LIS->getRegUnit(*MCRegUnitIterator(MCRegister::from(AMDGPU::SCC), TRI));
  SlotIndex Idx = LIS->getInstructionIndex(MI);
  return LR.liveAt(Idx);
}

//
// This function is useful for when we need to insert a new
// instruction that defines scc in a block and we need to find
// a location that will not smash the existing value.
//
// Starting at `BeforeInst` it will look backwards to try to find
// a place in the block where scc is dead so we can insert our new
// def there. If no location can be found it will save and restore
// scc around BeforeInst. This way BeforeInst can safely be used
// as the new insert location.
//
MachineBasicBlock::iterator findOrCreateInsertionPointForSccDef(
    MachineBasicBlock *MBB, MachineBasicBlock::iterator MI,
    const TargetRegisterInfo *TRI, const SIInstrInfo *TII,
    MachineRegisterInfo *MRI, LiveIntervals *LIS,
    SccDefInsertPointConstraintFlags Constraints) {
  // If SCC is dead at MI when we can use MI as the insert point.
  if (!llvm::isSccLiveAt(*MI, LIS))
    return MI;

  const bool CheckForExecWrite =
      Constraints & SccDefInsertPointConstraintFlags::NoExecWrite;

  MachineBasicBlock::reverse_iterator Start = MI.getReverse();

  // Otherwise, walk backwards through the block looking for a location where
  // SCC is dead.
  for (MachineBasicBlock::reverse_iterator It = Start, End = MBB->rend();
       It != End; ++It) {
    // If the instruction modifies exec then we cannot use it as
    // an insertion point (if that is a constraint from the caller).
    // The check for EXEC works for both wave64 and wave32 because
    // it will also catch Writes to the subregisters (e.g. exec_lo).
    if (CheckForExecWrite && It->modifiesRegister(AMDGPU::EXEC, TRI))
      break;

    if (!llvm::isSccLiveAt(*It, LIS))
      return It->getIterator();
  }

  // If no safe location can be found in the block we can save and restore
  // SCC around MI. There is no way to directly read or Write SCC so we use
  // s_cselect to read the current value of SCC and s_cmp to Write the saved
  // value back to SCC.
  //
  // The generated code will look like this;
  //
  //      %SavedSCC = COPY $scc  # Save SCC
  //      <----- Newly created safe insert point.
  //      MI
  //      $scc = COPY %SavedSCC  # Restore SCC
  //
  Register TmpScc = MRI->createVirtualRegister(&AMDGPU::SReg_32_XM0RegClass);
  DebugLoc DL = MI->getDebugLoc();
  auto CopyFrom =
      BuildMI(*MBB, MI, DL, TII->get(AMDGPU::COPY), TmpScc).addReg(AMDGPU::SCC);
  auto CopyTo = BuildMI(*MBB, std::next(MI->getIterator()), DL,
                        TII->get(AMDGPU::COPY), AMDGPU::SCC)
                    .addReg(TmpScc);

  // Cut the live segment.
  auto SlotIndexes = LIS->getSlotIndexes();
  SlotIndexes->insertMachineInstrInMaps(*CopyFrom);
  SlotIndexes->insertMachineInstrInMaps(*CopyTo);
  LiveRange &LR =
      LIS->getRegUnit(*MCRegUnitIterator(MCRegister::from(AMDGPU::SCC), TRI));
  auto OldSegment = *LR.getSegmentContaining(LIS->getInstructionIndex(*MI));
  LiveRange::Segment NewSegA(
      OldSegment.start,
      SlotIndexes->getInstructionIndex(*CopyFrom).getRegSlot(),
      OldSegment.valno);
  LiveRange::Segment NewSegB(LIS->getInstructionIndex(*CopyTo).getRegSlot(),
                             OldSegment.end, OldSegment.valno);
  LR.removeSegment(OldSegment);
  LR.addSegment(NewSegA);
  LR.addSegment(NewSegB);

  return MI;
}

void dumpLiveSet(const LiveSet &LiveSet, const SIRegisterInfo *SIRI) {

  dbgs() << "\n live set: \n";
  for (auto It : LiveSet) {
    int Reg = It.first;
    dbgs() << printReg(Reg, SIRI);
    if (It.second.any())
      dbgs() << " mask:" << It.second.getAsInteger();
    dbgs() << "\n";
  }
}

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
    if (SIRI->isVGPR(MRI, Reg))
      VPressure += Size;
    else
      SPressure += Size;
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
  if (FromBB == ToBB)
    return true;

  if (DT->dominates(FromBB, ToBB))
    return true;

  if (PDT->dominates(ToBB, FromBB))
    return true;

  if (loopContainsBoth(LI, ToBB, FromBB))
    return true;

  // TODO: cover case hotBB in loop,
  //       one block in that loop dom BB or
  //       BB post dom one block in that loop.
  return false;
}
} // namespace llvm
