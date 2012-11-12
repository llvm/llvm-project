//===-- Thumb1FrameLowering.cpp - Thumb1 Frame Information ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Thumb1 implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "Thumb1FrameLowering.h"
#include "ARMMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

bool Thumb1FrameLowering::hasReservedCallFrame(const MachineFunction &MF) const{
  const MachineFrameInfo *FFI = MF.getFrameInfo();
  unsigned CFSize = FFI->getMaxCallFrameSize();
  // It's not always a good idea to include the call frame as part of the
  // stack frame. ARM (especially Thumb) has small immediate offset to
  // address the stack frame. So a large call frame can cause poor codegen
  // and may even makes it impossible to scavenge a register.
  if (CFSize >= ((1 << 8) - 1) * 4 / 2) // Half of imm8 * 4
    return false;

  return !MF.getFrameInfo()->hasVarSizedObjects();
}

static void
emitSPUpdate(MachineBasicBlock &MBB,
             MachineBasicBlock::iterator &MBBI,
             const TargetInstrInfo &TII, DebugLoc dl,
             const Thumb1RegisterInfo &MRI,
             int NumBytes, unsigned MIFlags = MachineInstr::NoFlags)  {
  emitThumbRegPlusImmediate(MBB, MBBI, dl, ARM::SP, ARM::SP, NumBytes, TII,
                            MRI, MIFlags);
}

void Thumb1FrameLowering::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front();
  MachineBasicBlock::iterator MBBI = MBB.begin();
  MachineFrameInfo  *MFI = MF.getFrameInfo();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  const Thumb1RegisterInfo *RegInfo =
    static_cast<const Thumb1RegisterInfo*>(MF.getTarget().getRegisterInfo());
  const Thumb1InstrInfo &TII =
    *static_cast<const Thumb1InstrInfo*>(MF.getTarget().getInstrInfo());

  unsigned VARegSaveSize = AFI->getVarArgsRegSaveSize();
  unsigned NumBytes = MFI->getStackSize();
  const std::vector<CalleeSavedInfo> &CSI = MFI->getCalleeSavedInfo();
  DebugLoc dl = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();
  unsigned FramePtr = RegInfo->getFrameRegister(MF);
  unsigned BasePtr = RegInfo->getBaseRegister();

  // Thumb add/sub sp, imm8 instructions implicitly multiply the offset by 4.
  NumBytes = (NumBytes + 3) & ~3;
  MFI->setStackSize(NumBytes);

  // Determine the sizes of each callee-save spill areas and record which frame
  // belongs to which callee-save spill areas.
  unsigned GPRCS1Size = 0, GPRCS2Size = 0, DPRCSSize = 0;
  int FramePtrSpillFI = 0;

  if (VARegSaveSize)
    emitSPUpdate(MBB, MBBI, TII, dl, *RegInfo, -VARegSaveSize,
                 MachineInstr::FrameSetup);

  if (!AFI->hasStackFrame()) {
    if (NumBytes != 0)
      emitSPUpdate(MBB, MBBI, TII, dl, *RegInfo, -NumBytes,
                   MachineInstr::FrameSetup);
    return;
  }

  for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
    unsigned Reg = CSI[i].getReg();
    int FI = CSI[i].getFrameIdx();
    switch (Reg) {
    case ARM::R4:
    case ARM::R5:
    case ARM::R6:
    case ARM::R7:
    case ARM::LR:
      if (Reg == FramePtr)
        FramePtrSpillFI = FI;
      AFI->addGPRCalleeSavedArea1Frame(FI);
      GPRCS1Size += 4;
      break;
    case ARM::R8:
    case ARM::R9:
    case ARM::R10:
    case ARM::R11:
      if (Reg == FramePtr)
        FramePtrSpillFI = FI;
      if (STI.isTargetIOS()) {
        AFI->addGPRCalleeSavedArea2Frame(FI);
        GPRCS2Size += 4;
      } else {
        AFI->addGPRCalleeSavedArea1Frame(FI);
        GPRCS1Size += 4;
      }
      break;
    default:
      AFI->addDPRCalleeSavedAreaFrame(FI);
      DPRCSSize += 8;
    }
  }

  if (MBBI != MBB.end() && MBBI->getOpcode() == ARM::tPUSH) {
    ++MBBI;
    if (MBBI != MBB.end())
      dl = MBBI->getDebugLoc();
  }

  // Determine starting offsets of spill areas.
  unsigned DPRCSOffset  = NumBytes - (GPRCS1Size + GPRCS2Size + DPRCSSize);
  unsigned GPRCS2Offset = DPRCSOffset + DPRCSSize;
  unsigned GPRCS1Offset = GPRCS2Offset + GPRCS2Size;
  AFI->setFramePtrSpillOffset(MFI->getObjectOffset(FramePtrSpillFI) + NumBytes);
  AFI->setGPRCalleeSavedArea1Offset(GPRCS1Offset);
  AFI->setGPRCalleeSavedArea2Offset(GPRCS2Offset);
  AFI->setDPRCalleeSavedAreaOffset(DPRCSOffset);
  NumBytes = DPRCSOffset;

  // Adjust FP so it point to the stack slot that contains the previous FP.
  if (hasFP(MF)) {
    AddDefaultPred(BuildMI(MBB, MBBI, dl, TII.get(ARM::tADDrSPi), FramePtr)
      .addFrameIndex(FramePtrSpillFI).addImm(0)
      .setMIFlags(MachineInstr::FrameSetup));
    if (NumBytes > 508)
      // If offset is > 508 then sp cannot be adjusted in a single instruction,
      // try restoring from fp instead.
      AFI->setShouldRestoreSPFromFP(true);
  }

  if (NumBytes)
    // Insert it after all the callee-save spills.
    emitSPUpdate(MBB, MBBI, TII, dl, *RegInfo, -NumBytes,
                 MachineInstr::FrameSetup);

  if (STI.isTargetELF() && hasFP(MF))
    MFI->setOffsetAdjustment(MFI->getOffsetAdjustment() -
                             AFI->getFramePtrSpillOffset());

  AFI->setGPRCalleeSavedArea1Size(GPRCS1Size);
  AFI->setGPRCalleeSavedArea2Size(GPRCS2Size);
  AFI->setDPRCalleeSavedAreaSize(DPRCSSize);

  // Thumb1 does not currently support dynamic stack realignment.  Report a
  // fatal error rather then silently generate bad code.
  if (RegInfo->needsStackRealignment(MF))
      report_fatal_error("Dynamic stack realignment not supported for thumb1.");

  // If we need a base pointer, set it up here. It's whatever the value
  // of the stack pointer is at this point. Any variable size objects
  // will be allocated after this, so we can still use the base pointer
  // to reference locals.
  if (RegInfo->hasBasePointer(MF))
    AddDefaultPred(BuildMI(MBB, MBBI, dl, TII.get(ARM::tMOVr), BasePtr)
                   .addReg(ARM::SP));

  // If the frame has variable sized objects then the epilogue must restore
  // the sp from fp. We can assume there's an FP here since hasFP already
  // checks for hasVarSizedObjects.
  if (MFI->hasVarSizedObjects())
    AFI->setShouldRestoreSPFromFP(true);
}

static bool isCalleeSavedRegister(unsigned Reg, const uint16_t *CSRegs) {
  for (unsigned i = 0; CSRegs[i]; ++i)
    if (Reg == CSRegs[i])
      return true;
  return false;
}

static bool isCSRestore(MachineInstr *MI, const uint16_t *CSRegs) {
  if (MI->getOpcode() == ARM::tLDRspi &&
      MI->getOperand(1).isFI() &&
      isCalleeSavedRegister(MI->getOperand(0).getReg(), CSRegs))
    return true;
  else if (MI->getOpcode() == ARM::tPOP) {
    // The first two operands are predicates. The last two are
    // imp-def and imp-use of SP. Check everything in between.
    for (int i = 2, e = MI->getNumOperands() - 2; i != e; ++i)
      if (!isCalleeSavedRegister(MI->getOperand(i).getReg(), CSRegs))
        return false;
    return true;
  }
  return false;
}

void Thumb1FrameLowering::emitEpilogue(MachineFunction &MF,
                                   MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  assert((MBBI->getOpcode() == ARM::tBX_RET ||
          MBBI->getOpcode() == ARM::tPOP_RET) &&
         "Can only insert epilog into returning blocks");
  DebugLoc dl = MBBI->getDebugLoc();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  const Thumb1RegisterInfo *RegInfo =
    static_cast<const Thumb1RegisterInfo*>(MF.getTarget().getRegisterInfo());
  const Thumb1InstrInfo &TII =
    *static_cast<const Thumb1InstrInfo*>(MF.getTarget().getInstrInfo());

  unsigned VARegSaveSize = AFI->getVarArgsRegSaveSize();
  int NumBytes = (int)MFI->getStackSize();
  const uint16_t *CSRegs = RegInfo->getCalleeSavedRegs();
  unsigned FramePtr = RegInfo->getFrameRegister(MF);

  if (!AFI->hasStackFrame()) {
    if (NumBytes != 0)
      emitSPUpdate(MBB, MBBI, TII, dl, *RegInfo, NumBytes);
  } else {
    // Unwind MBBI to point to first LDR / VLDRD.
    if (MBBI != MBB.begin()) {
      do
        --MBBI;
      while (MBBI != MBB.begin() && isCSRestore(MBBI, CSRegs));
      if (!isCSRestore(MBBI, CSRegs))
        ++MBBI;
    }

    // Move SP to start of FP callee save spill area.
    NumBytes -= (AFI->getGPRCalleeSavedArea1Size() +
                 AFI->getGPRCalleeSavedArea2Size() +
                 AFI->getDPRCalleeSavedAreaSize());

    if (AFI->shouldRestoreSPFromFP()) {
      NumBytes = AFI->getFramePtrSpillOffset() - NumBytes;
      // Reset SP based on frame pointer only if the stack frame extends beyond
      // frame pointer stack slot, the target is ELF and the function has FP, or
      // the target uses var sized objects.
      if (NumBytes) {
        assert(MF.getRegInfo().isPhysRegUsed(ARM::R4) &&
               "No scratch register to restore SP from FP!");
        emitThumbRegPlusImmediate(MBB, MBBI, dl, ARM::R4, FramePtr, -NumBytes,
                                  TII, *RegInfo);
        AddDefaultPred(BuildMI(MBB, MBBI, dl, TII.get(ARM::tMOVr),
                               ARM::SP)
          .addReg(ARM::R4));
      } else
        AddDefaultPred(BuildMI(MBB, MBBI, dl, TII.get(ARM::tMOVr),
                               ARM::SP)
          .addReg(FramePtr));
    } else {
      if (MBBI->getOpcode() == ARM::tBX_RET &&
          &MBB.front() != MBBI &&
          prior(MBBI)->getOpcode() == ARM::tPOP) {
        MachineBasicBlock::iterator PMBBI = prior(MBBI);
        emitSPUpdate(MBB, PMBBI, TII, dl, *RegInfo, NumBytes);
      } else
        emitSPUpdate(MBB, MBBI, TII, dl, *RegInfo, NumBytes);
    }
  }

  if (VARegSaveSize) {
    // Unlike T2 and ARM mode, the T1 pop instruction cannot restore
    // to LR, and we can't pop the value directly to the PC since
    // we need to update the SP after popping the value. Therefore, we
    // pop the old LR into R3 as a temporary.

    // Move back past the callee-saved register restoration
    while (MBBI != MBB.end() && isCSRestore(MBBI, CSRegs))
      ++MBBI;
    // Epilogue for vararg functions: pop LR to R3 and branch off it.
    AddDefaultPred(BuildMI(MBB, MBBI, dl, TII.get(ARM::tPOP)))
      .addReg(ARM::R3, RegState::Define);

    emitSPUpdate(MBB, MBBI, TII, dl, *RegInfo, VARegSaveSize);

    MachineInstrBuilder MIB =
      BuildMI(MBB, MBBI, dl, TII.get(ARM::tBX_RET_vararg))
      .addReg(ARM::R3, RegState::Kill);
    AddDefaultPred(MIB);
    MIB->copyImplicitOps(&*MBBI);
    // erase the old tBX_RET instruction
    MBB.erase(MBBI);
  }
}

bool Thumb1FrameLowering::
spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator MI,
                          const std::vector<CalleeSavedInfo> &CSI,
                          const TargetRegisterInfo *TRI) const {
  if (CSI.empty())
    return false;

  DebugLoc DL;
  MachineFunction &MF = *MBB.getParent();
  const TargetInstrInfo &TII = *MF.getTarget().getInstrInfo();

  if (MI != MBB.end()) DL = MI->getDebugLoc();

  MachineInstrBuilder MIB = BuildMI(MBB, MI, DL, TII.get(ARM::tPUSH));
  AddDefaultPred(MIB);
  for (unsigned i = CSI.size(); i != 0; --i) {
    unsigned Reg = CSI[i-1].getReg();
    bool isKill = true;

    // Add the callee-saved register as live-in unless it's LR and
    // @llvm.returnaddress is called. If LR is returned for @llvm.returnaddress
    // then it's already added to the function and entry block live-in sets.
    if (Reg == ARM::LR) {
      MachineFunction &MF = *MBB.getParent();
      if (MF.getFrameInfo()->isReturnAddressTaken() &&
          MF.getRegInfo().isLiveIn(Reg))
        isKill = false;
    }

    if (isKill)
      MBB.addLiveIn(Reg);

    MIB.addReg(Reg, getKillRegState(isKill));
  }
  MIB.setMIFlags(MachineInstr::FrameSetup);
  return true;
}

bool Thumb1FrameLowering::
restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MI,
                            const std::vector<CalleeSavedInfo> &CSI,
                            const TargetRegisterInfo *TRI) const {
  if (CSI.empty())
    return false;

  MachineFunction &MF = *MBB.getParent();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  const TargetInstrInfo &TII = *MF.getTarget().getInstrInfo();

  bool isVarArg = AFI->getVarArgsRegSaveSize() > 0;
  DebugLoc DL = MI->getDebugLoc();
  MachineInstrBuilder MIB = BuildMI(MF, DL, TII.get(ARM::tPOP));
  AddDefaultPred(MIB);

  bool NumRegs = false;
  for (unsigned i = CSI.size(); i != 0; --i) {
    unsigned Reg = CSI[i-1].getReg();
    if (Reg == ARM::LR) {
      // Special epilogue for vararg functions. See emitEpilogue
      if (isVarArg)
        continue;
      Reg = ARM::PC;
      (*MIB).setDesc(TII.get(ARM::tPOP_RET));
      MIB->copyImplicitOps(&*MI);
      MI = MBB.erase(MI);
    }
    MIB.addReg(Reg, getDefRegState(true));
    NumRegs = true;
  }

  // It's illegal to emit pop instruction without operands.
  if (NumRegs)
    MBB.insert(MI, &*MIB);
  else
    MF.DeleteMachineInstr(MIB);

  return true;
}
