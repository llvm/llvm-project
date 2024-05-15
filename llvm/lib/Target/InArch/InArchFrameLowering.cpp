#include "InArchFrameLowering.h"
#include "InArchMachineFunctionInfo.h"
#include "InArchSubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>

#define DEBUG_TYPE "InArch-frame-lowering"

using namespace llvm;

// seems done
void InArchFrameLowering::determineCalleeSaves(MachineFunction &MF,
                                             BitVector &SavedRegs,
                                             RegScavenger *RS) const {
  TargetFrameLowering::determineCalleeSaves(MF, SavedRegs, RS);
  // Unconditionally spill RA and FP only if the function uses a frame
  // pointer.
  if (hasFP(MF)) {
    SavedRegs.set(InArch::RA);
    SavedRegs.set(InArch::FP);
  }
  // Mark BP as used if function has dedicated base pointer.
  if (hasBP(MF))
    SavedRegs.set(InArch::BP);
}


void InArchFrameLowering::adjustReg(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator MBBI,
                                  const DebugLoc &DL, Register DestReg,
                                  Register SrcReg, int64_t Val,
                                  MachineInstr::MIFlag Flag) const {
  // MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  const InArchInstrInfo *TII = STI.getInstrInfo();

  if (DestReg == SrcReg && Val == 0)
    return;

  if (isInt<16>(Val)) {
    BuildMI(MBB, MBBI, DL, TII->get(InArch::ADDI), DestReg)
        .addReg(SrcReg)
        .addImm(Val)
        .setMIFlag(Flag);
  } else {
    // alloc vreg, load imm, add
    llvm_unreachable("");
  }
}

void InArchFrameLowering::adjustStackToMatchRecords(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    bool Allocate) const {
  llvm_unreachable("");
}

// TODO: seems ok
void InArchFrameLowering::emitPrologue(MachineFunction &MF,
                                     MachineBasicBlock &MBB) const {
  MachineFrameInfo &MFI = MF.getFrameInfo();
  auto *FI = MF.getInfo<InArchFunctionInfo>();
  const InArchRegisterInfo *RI = STI.getRegisterInfo();
  MachineBasicBlock::iterator MBBI = MBB.begin();

  Register FPReg = InArch::FP;
  Register SPReg = InArch::SP;
  // Register BPReg = InArch::BP;
  // Debug location must be unknown since the first debug location is used
  // to determine the end of the prologue.
  DebugLoc DL;

  uint64_t StackSize = alignTo(MFI.getStackSize(), getStackAlign());
  MFI.setStackSize(StackSize);


  if (!isInt<16>(StackSize)) {
    llvm_unreachable("Stack offs won't fit in InArch::LDi");
  }

  // Early exit if there is no need to allocate on the stack
  if (StackSize == 0 && !MFI.adjustsStack())
    return;

  // Allocate space on the stack if necessary.
  adjustReg(MBB, MBBI, DL, SPReg, SPReg, -StackSize, MachineInstr::FrameSetup);

  const auto &CSI = MFI.getCalleeSavedInfo();

  // The frame pointer is callee-saved, and code has been generated for us to
  // save it to the stack. We need to skip over the storing of callee-saved
  // registers as the frame pointer must be modified after it has been saved
  // to the stack, not before.
  // FIXME: assumes exactly one instruction is used to save each callee-saved
  // register.
  std::advance(MBBI, CSI.size());

  if (!hasFP(MF)) {
    return;
  }

  // Generate new FP.
  adjustReg(MBB, MBBI, DL, FPReg, SPReg, StackSize - FI->getVarArgsSaveSize(),
            MachineInstr::FrameSetup);

  if (RI->hasStackRealignment(MF)) {
    llvm_unreachable(""); // TODO: realigned stack
  }
}

// TODO: seems ok
void InArchFrameLowering::emitEpilogue(MachineFunction &MF,
                                     MachineBasicBlock &MBB) const {
    const InArchRegisterInfo *RI = STI.getRegisterInfo();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  auto *UFI = MF.getInfo<InArchFunctionInfo>();
  Register FPReg = InArch::FP;
  Register SPReg = InArch::SP;

  // Get the insert location for the epilogue. If there were no terminators in
  // the block, get the last instruction.
  MachineBasicBlock::iterator MBBI = MBB.end();
  DebugLoc DL;
  if (!MBB.empty()) {
    MBBI = MBB.getFirstTerminator();
    if (MBBI == MBB.end())
      MBBI = MBB.getLastNonDebugInstr();
    DL = MBBI->getDebugLoc();

    // If this is not a terminator, the actual insert location should be after
    // the last instruction.
    if (!MBBI->isTerminator())
      MBBI = std::next(MBBI);

    // TODO: is it necessary?
    while (MBBI != MBB.begin() &&
           std::prev(MBBI)->getFlag(MachineInstr::FrameDestroy))
      --MBBI;
  }

  const auto &CSI = MFI.getCalleeSavedInfo();

  // Skip to before the restores of callee-saved registers
  // FIXME: assumes exactly one instruction is used to restore each
  // callee-saved register.
  auto LastFrameDestroy = MBBI;
  if (!CSI.empty())
    LastFrameDestroy = std::prev(MBBI, CSI.size());

  uint64_t StackSize = MFI.getStackSize();
  uint64_t FPOffset = StackSize - UFI->getVarArgsSaveSize();

  // Restore the stack pointer using the value of the frame pointer. Only
  // necessary if the stack pointer was modified, meaning the stack size is
  // unknown.
  if (RI->hasStackRealignment(MF) || MFI.hasVarSizedObjects()) {
    llvm_unreachable("");
    assert(hasFP(MF) && "frame pointer should not have been eliminated");
    adjustReg(MBB, LastFrameDestroy, DL, SPReg, FPReg, -FPOffset,
              MachineInstr::FrameDestroy);
  }

  // Deallocate stack
  adjustReg(MBB, MBBI, DL, SPReg, SPReg, StackSize, MachineInstr::FrameDestroy);
}

// TODO: seems ok
bool InArchFrameLowering::spillCalleeSavedRegisters(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
    ArrayRef<CalleeSavedInfo> CSI, const TargetRegisterInfo *TRI) const {
  if (CSI.empty())
    return true;

  MachineFunction *MF = MBB.getParent();
  const TargetInstrInfo &TII = *MF->getSubtarget().getInstrInfo();

  for (auto &CS : CSI) {
    // Insert the spill to the stack frame.
    Register Reg = CS.getReg();
    const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
    TII.storeRegToStackSlot(MBB, MI, Reg, !MBB.isLiveIn(Reg), CS.getFrameIdx(),
                            RC, TRI, Register());
  }

  return true;
}

// TODO: seems ok
bool InArchFrameLowering::restoreCalleeSavedRegisters(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
    MutableArrayRef<CalleeSavedInfo> CSI, const TargetRegisterInfo *TRI) const {
  if (CSI.empty())
    return true;

  MachineFunction *MF = MBB.getParent();
  const TargetInstrInfo &TII = *MF->getSubtarget().getInstrInfo();

  // Insert in reverse order.
  // loadRegFromStackSlot can insert multiple instructions.
  for (auto &CS : reverse(CSI)) {
    Register Reg = CS.getReg();
    const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
    TII.loadRegFromStackSlot(MBB, MI, Reg, CS.getFrameIdx(), RC, TRI, Register());
    assert(MI != MBB.begin() && "loadRegFromStackSlot didn't insert any code!");
  }

  return true;
}

// TODO: seems ok
void InArchFrameLowering::processFunctionBeforeFrameFinalized(
    MachineFunction &MF, RegScavenger *RS) const {
  MachineFrameInfo &MFI = MF.getFrameInfo();
  auto *UFI = MF.getInfo<InArchFunctionInfo>();
  return;

  if (MFI.getCalleeSavedInfo().empty()) {
    UFI->setCalleeSavedStackSize(0);
    return;
  }

  unsigned Size = 0;
  for (const auto &Info : MFI.getCalleeSavedInfo()) {
    int FrameIdx = Info.getFrameIdx();
    if (MFI.getStackID(FrameIdx) != TargetStackID::Default)
      continue;

    Size += MFI.getObjectSize(FrameIdx);
  }
  UFI->setCalleeSavedStackSize(Size);
}

// Eliminate ADJCALLSTACKDOWN, ADJCALLSTACKUP pseudo instructions.
// TODO: seems ok
MachineBasicBlock::iterator InArchFrameLowering::eliminateCallFramePseudoInstr(
    MachineFunction &MF, MachineBasicBlock &MBB,
    MachineBasicBlock::iterator MI) const {
  Register SPReg = InArch::SP;
  DebugLoc DL = MI->getDebugLoc();

  if (!hasReservedCallFrame(MF)) {
    // If space has not been reserved for a call frame, ADJCALLSTACKDOWN and
    // ADJCALLSTACKUP must be converted to instructions manipulating the stack
    // pointer. This is necessary when there is a variable length stack
    // allocation (e.g. alloca), which means it's not possible to allocate
    // space for outgoing arguments from within the function prologue.
    int64_t Amount = MI->getOperand(0).getImm();

    if (Amount != 0) {
      // Ensure the stack remains aligned after adjustment.
      Amount = alignSPAdjust(Amount);

      if (MI->getOpcode() == InArch::ADJCALLSTACKDOWN)
         Amount = -Amount;

      adjustReg(MBB, MI, DL, SPReg, SPReg, Amount, MachineInstr::NoFlags);
    }
  }

  return MBB.erase(MI);
}

bool InArchFrameLowering::hasFP(const MachineFunction &MF) const {
  const TargetRegisterInfo *RegInfo = MF.getSubtarget().getRegisterInfo();

  const MachineFrameInfo &MFI = MF.getFrameInfo();
  return MF.getTarget().Options.DisableFramePointerElim(MF) || // -fomit-frame-pointer
         RegInfo->hasStackRealignment(MF) || MFI.hasVarSizedObjects() ||
         MFI.isFrameAddressTaken();
}

bool InArchFrameLowering::hasBP(const MachineFunction &MF) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();

  return MFI.hasVarSizedObjects() && TRI->hasStackRealignment(MF);
}

// TODO: rewrite!
StackOffset
InArchFrameLowering::getFrameIndexReference(const MachineFunction &MF, int FI,
                                          Register &FrameReg) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const TargetRegisterInfo *RI = MF.getSubtarget().getRegisterInfo();
  const auto *UFI = MF.getInfo<InArchFunctionInfo>();

  // Callee-saved registers should be referenced relative to the stack
  // pointer (positive offset), otherwise use the frame pointer (negative
  // offset).
  const auto &CSI = MFI.getCalleeSavedInfo();
  int MinCSFI = 0;
  int MaxCSFI = -1;
  int Offset;

  Offset = MFI.getObjectOffset(FI) - getOffsetOfLocalArea() +
           MFI.getOffsetAdjustment();

  if (CSI.size()) {
    MinCSFI = CSI[0].getFrameIdx();
    MaxCSFI = CSI[CSI.size() - 1].getFrameIdx();
  }

  if (FI >= MinCSFI && FI <= MaxCSFI) {
    FrameReg = InArch::SP;
    Offset += MFI.getStackSize();
  } else if (RI->hasStackRealignment(MF) && !MFI.isFixedObjectIndex(FI)) {
    // TODO: realigned stack
    llvm_unreachable("");
  } else {
    // TODO: what's going on here
    FrameReg = RI->getFrameRegister(MF);
    if (hasFP(MF)) {
      Offset += UFI->getVarArgsSaveSize();
    } else {
      Offset += MFI.getStackSize();
    }
  }

  return StackOffset::getFixed(Offset);
}

// Not preserve stack space within prologue for outgoing variables when the
// function contains variable size objects or there are vector objects accessed
// by the frame pointer.
// Let eliminateCallFramePseudoInstr preserve stack space for it.
bool InArchFrameLowering::hasReservedCallFrame(const MachineFunction &MF) const {
  return !MF.getFrameInfo().hasVarSizedObjects();
}