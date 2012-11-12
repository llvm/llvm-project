//===-- HexagonFrameLowering.cpp - Define frame lowering ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//
//===----------------------------------------------------------------------===//

#include "HexagonFrameLowering.h"
#include "Hexagon.h"
#include "HexagonInstrInfo.h"
#include "HexagonRegisterInfo.h"
#include "HexagonSubtarget.h"
#include "HexagonTargetMachine.h"
#include "HexagonMachineFunctionInfo.h"
#include "llvm/Function.h"
#include "llvm/Type.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/MC/MachineLocation.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

static cl::opt<bool> DisableDeallocRet(
                       "disable-hexagon-dealloc-ret",
                       cl::Hidden,
                       cl::desc("Disable Dealloc Return for Hexagon target"));

/// determineFrameLayout - Determine the size of the frame and maximum call
/// frame size.
void HexagonFrameLowering::determineFrameLayout(MachineFunction &MF) const {
  MachineFrameInfo *MFI = MF.getFrameInfo();

  // Get the number of bytes to allocate from the FrameInfo.
  unsigned FrameSize = MFI->getStackSize();

  // Get the alignments provided by the target.
  unsigned TargetAlign = MF.getTarget().getFrameLowering()->getStackAlignment();
  // Get the maximum call frame size of all the calls.
  unsigned maxCallFrameSize = MFI->getMaxCallFrameSize();

  // If we have dynamic alloca then maxCallFrameSize needs to be aligned so
  // that allocations will be aligned.
  if (MFI->hasVarSizedObjects())
    maxCallFrameSize = RoundUpToAlignment(maxCallFrameSize, TargetAlign);

  // Update maximum call frame size.
  MFI->setMaxCallFrameSize(maxCallFrameSize);

  // Include call frame size in total.
  FrameSize += maxCallFrameSize;

  // Make sure the frame is aligned.
  FrameSize = RoundUpToAlignment(FrameSize, TargetAlign);

  // Update frame info.
  MFI->setStackSize(FrameSize);
}


void HexagonFrameLowering::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineModuleInfo &MMI = MF.getMMI();
  MachineBasicBlock::iterator MBBI = MBB.begin();
  const HexagonRegisterInfo *QRI =
    static_cast<const HexagonRegisterInfo *>(MF.getTarget().getRegisterInfo());
  DebugLoc dl = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();
  determineFrameLayout(MF);

  // Check if frame moves are needed for EH.
  bool needsFrameMoves = MMI.hasDebugInfo() ||
    !MF.getFunction()->needsUnwindTableEntry();

  // Get the number of bytes to allocate from the FrameInfo.
  int NumBytes = (int) MFI->getStackSize();

  // LLVM expects allocframe not to be the first instruction in the
  // basic block.
  MachineBasicBlock::iterator InsertPt = MBB.begin();

  //
  // ALLOCA adjust regs.  Iterate over ADJDYNALLOC nodes and change the offset.
  //
  HexagonMachineFunctionInfo *FuncInfo =
    MF.getInfo<HexagonMachineFunctionInfo>();
  const std::vector<MachineInstr*>& AdjustRegs =
    FuncInfo->getAllocaAdjustInsts();
  for (std::vector<MachineInstr*>::const_iterator i = AdjustRegs.begin(),
         e = AdjustRegs.end();
       i != e; ++i) {
    MachineInstr* MI = *i;
    assert((MI->getOpcode() == Hexagon::ADJDYNALLOC) &&
           "Expected adjust alloca node");

    MachineOperand& MO = MI->getOperand(2);
    assert(MO.isImm() && "Expected immediate");
    MO.setImm(MFI->getMaxCallFrameSize());
  }

 std::vector<MachineMove> &Moves = MMI.getFrameMoves();

 if (needsFrameMoves) {
   // Advance CFA. DW_CFA_def_cfa
   unsigned FPReg = QRI->getFrameRegister();
   unsigned RAReg = QRI->getRARegister();

   MachineLocation Dst(MachineLocation::VirtualFP);
   MachineLocation Src(FPReg, -8);
   Moves.push_back(MachineMove(0, Dst, Src));

   // R31 = (R31 - #4)
   MachineLocation LRDst(RAReg, -4);
   MachineLocation LRSrc(RAReg);
   Moves.push_back(MachineMove(0, LRDst, LRSrc));

   // R30 = (R30 - #8)
   MachineLocation SPDst(FPReg, -8);
   MachineLocation SPSrc(FPReg);
   Moves.push_back(MachineMove(0, SPDst, SPSrc));
 }

  //
  // Only insert ALLOCFRAME if we need to.
  //
  if (hasFP(MF)) {
    // Check for overflow.
    // Hexagon_TODO: Ugh! hardcoding. Is there an API that can be used?
    const int ALLOCFRAME_MAX = 16384;
    const TargetInstrInfo &TII = *MF.getTarget().getInstrInfo();

    if (NumBytes >= ALLOCFRAME_MAX) {
      // Emit allocframe(#0).
      BuildMI(MBB, InsertPt, dl, TII.get(Hexagon::ALLOCFRAME)).addImm(0);

      // Subtract offset from frame pointer.
      BuildMI(MBB, InsertPt, dl, TII.get(Hexagon::CONST32_Int_Real),
                                      HEXAGON_RESERVED_REG_1).addImm(NumBytes);
      BuildMI(MBB, InsertPt, dl, TII.get(Hexagon::SUB_rr),
                                      QRI->getStackRegister()).
                                      addReg(QRI->getStackRegister()).
                                      addReg(HEXAGON_RESERVED_REG_1);
    } else {
      BuildMI(MBB, InsertPt, dl, TII.get(Hexagon::ALLOCFRAME)).addImm(NumBytes);
    }
  }
}
// Returns true if MBB has a machine instructions that indicates a tail call
// in the block.
bool HexagonFrameLowering::hasTailCall(MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  unsigned RetOpcode = MBBI->getOpcode();

  return RetOpcode == Hexagon::TCRETURNtg || RetOpcode == Hexagon::TCRETURNtext;}

void HexagonFrameLowering::emitEpilogue(MachineFunction &MF,
                                     MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = prior(MBB.end());
  DebugLoc dl = MBBI->getDebugLoc();
  //
  // Only insert deallocframe if we need to.
  //
  if (hasFP(MF)) {
    MachineBasicBlock::iterator MBBI = prior(MBB.end());
    MachineBasicBlock::iterator MBBI_end = MBB.end();
    //
    // For Hexagon, we don't need the frame size.
    //
    MachineFrameInfo *MFI = MF.getFrameInfo();
    int NumBytes = (int) MFI->getStackSize();

    const TargetInstrInfo &TII = *MF.getTarget().getInstrInfo();

    // Replace 'jumpr r31' instruction with dealloc_return for V4 and higher
    // versions.
    if (STI.hasV4TOps() && MBBI->getOpcode() == Hexagon::JMPR
                        && !DisableDeallocRet) {
      // Remove jumpr node.
      MBB.erase(MBBI);
      // Add dealloc_return.
      BuildMI(MBB, MBBI_end, dl, TII.get(Hexagon::DEALLOC_RET_V4))
        .addImm(NumBytes);
    } else { // Add deallocframe for V2 and V3.
      BuildMI(MBB, MBBI, dl, TII.get(Hexagon::DEALLOCFRAME)).addImm(NumBytes);
    }
  }
}

bool HexagonFrameLowering::hasFP(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  const HexagonMachineFunctionInfo *FuncInfo =
    MF.getInfo<HexagonMachineFunctionInfo>();
  return (MFI->hasCalls() || (MFI->getStackSize() > 0) ||
          FuncInfo->hasClobberLR() );
}

static inline
unsigned uniqueSuperReg(unsigned Reg, const TargetRegisterInfo *TRI) {
  MCSuperRegIterator SRI(Reg, TRI);
  assert(SRI.isValid() && "Expected a superreg");
  unsigned SuperReg = *SRI;
  ++SRI;
  assert(!SRI.isValid() && "Expected exactly one superreg");
  return SuperReg;
}

bool
HexagonFrameLowering::spillCalleeSavedRegisters(
                                        MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MI,
                                        const std::vector<CalleeSavedInfo> &CSI,
                                        const TargetRegisterInfo *TRI) const {
  MachineFunction *MF = MBB.getParent();
  const TargetInstrInfo &TII = *MF->getTarget().getInstrInfo();

  if (CSI.empty()) {
    return false;
  }

  // We can only schedule double loads if we spill contiguous callee-saved regs
  // For instance, we cannot scheduled double-word loads if we spill r24,
  // r26, and r27.
  // Hexagon_TODO: We can try to double-word align odd registers for -O2 and
  // above.
  bool ContiguousRegs = true;

  for (unsigned i = 0; i < CSI.size(); ++i) {
    unsigned Reg = CSI[i].getReg();

    //
    // Check if we can use a double-word store.
    //
    unsigned SuperReg = uniqueSuperReg(Reg, TRI);
    bool CanUseDblStore = false;
    const TargetRegisterClass* SuperRegClass = 0;

    if (ContiguousRegs && (i < CSI.size()-1)) {
      unsigned SuperRegNext = uniqueSuperReg(CSI[i+1].getReg(), TRI);
      SuperRegClass = TRI->getMinimalPhysRegClass(SuperReg);
      CanUseDblStore = (SuperRegNext == SuperReg);
    }


    if (CanUseDblStore) {
      TII.storeRegToStackSlot(MBB, MI, SuperReg, true,
                              CSI[i+1].getFrameIdx(), SuperRegClass, TRI);
      MBB.addLiveIn(SuperReg);
      ++i;
    } else {
      // Cannot use a double-word store.
      ContiguousRegs = false;
      const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
      TII.storeRegToStackSlot(MBB, MI, Reg, true, CSI[i].getFrameIdx(), RC,
                              TRI);
      MBB.addLiveIn(Reg);
    }
  }
  return true;
}


bool HexagonFrameLowering::restoreCalleeSavedRegisters(
                                        MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MI,
                                        const std::vector<CalleeSavedInfo> &CSI,
                                        const TargetRegisterInfo *TRI) const {

  MachineFunction *MF = MBB.getParent();
  const TargetInstrInfo &TII = *MF->getTarget().getInstrInfo();

  if (CSI.empty()) {
    return false;
  }

  // We can only schedule double loads if we spill contiguous callee-saved regs
  // For instance, we cannot scheduled double-word loads if we spill r24,
  // r26, and r27.
  // Hexagon_TODO: We can try to double-word align odd registers for -O2 and
  // above.
  bool ContiguousRegs = true;

  for (unsigned i = 0; i < CSI.size(); ++i) {
    unsigned Reg = CSI[i].getReg();

    //
    // Check if we can use a double-word load.
    //
    unsigned SuperReg = uniqueSuperReg(Reg, TRI);
    const TargetRegisterClass* SuperRegClass = 0;
    bool CanUseDblLoad = false;
    if (ContiguousRegs && (i < CSI.size()-1)) {
      unsigned SuperRegNext = uniqueSuperReg(CSI[i+1].getReg(), TRI);
      SuperRegClass = TRI->getMinimalPhysRegClass(SuperReg);
      CanUseDblLoad = (SuperRegNext == SuperReg);
    }


    if (CanUseDblLoad) {
      TII.loadRegFromStackSlot(MBB, MI, SuperReg, CSI[i+1].getFrameIdx(),
                               SuperRegClass, TRI);
      MBB.addLiveIn(SuperReg);
      ++i;
    } else {
      // Cannot use a double-word load.
      ContiguousRegs = false;
      const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
      TII.loadRegFromStackSlot(MBB, MI, Reg, CSI[i].getFrameIdx(), RC, TRI);
      MBB.addLiveIn(Reg);
    }
  }
  return true;
}

int HexagonFrameLowering::getFrameIndexOffset(const MachineFunction &MF,
                                              int FI) const {
  return MF.getFrameInfo()->getObjectOffset(FI);
}
