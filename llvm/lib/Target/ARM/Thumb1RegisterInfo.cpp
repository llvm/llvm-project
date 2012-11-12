//===-- Thumb1RegisterInfo.cpp - Thumb-1 Register Information -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Thumb-1 implementation of the TargetRegisterInfo
// class.
//
//===----------------------------------------------------------------------===//

#include "Thumb1RegisterInfo.h"
#include "ARM.h"
#include "ARMBaseInstrInfo.h"
#include "ARMMachineFunctionInfo.h"
#include "ARMSubtarget.h"
#include "MCTargetDesc/ARMAddressingModes.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/LLVMContext.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
extern cl::opt<bool> ReuseFrameIndexVals;
}

using namespace llvm;

Thumb1RegisterInfo::Thumb1RegisterInfo(const ARMBaseInstrInfo &tii,
                                       const ARMSubtarget &sti)
  : ARMBaseRegisterInfo(tii, sti) {
}

const TargetRegisterClass*
Thumb1RegisterInfo::getLargestLegalSuperClass(const TargetRegisterClass *RC)
                                                                         const {
  if (ARM::tGPRRegClass.hasSubClassEq(RC))
    return &ARM::tGPRRegClass;
  return ARMBaseRegisterInfo::getLargestLegalSuperClass(RC);
}

const TargetRegisterClass *
Thumb1RegisterInfo::getPointerRegClass(const MachineFunction &MF, unsigned Kind)
                                                                         const {
  return &ARM::tGPRRegClass;
}

/// emitLoadConstPool - Emits a load from constpool to materialize the
/// specified immediate.
void
Thumb1RegisterInfo::emitLoadConstPool(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator &MBBI,
                                      DebugLoc dl,
                                      unsigned DestReg, unsigned SubIdx,
                                      int Val,
                                      ARMCC::CondCodes Pred, unsigned PredReg,
                                      unsigned MIFlags) const {
  MachineFunction &MF = *MBB.getParent();
  MachineConstantPool *ConstantPool = MF.getConstantPool();
  const Constant *C = ConstantInt::get(
          Type::getInt32Ty(MBB.getParent()->getFunction()->getContext()), Val);
  unsigned Idx = ConstantPool->getConstantPoolIndex(C, 4);

  BuildMI(MBB, MBBI, dl, TII.get(ARM::tLDRpci))
    .addReg(DestReg, getDefRegState(true), SubIdx)
    .addConstantPoolIndex(Idx).addImm(Pred).addReg(PredReg)
    .setMIFlags(MIFlags);
}


/// emitThumbRegPlusImmInReg - Emits a series of instructions to materialize
/// a destreg = basereg + immediate in Thumb code. Materialize the immediate
/// in a register using mov / mvn sequences or load the immediate from a
/// constpool entry.
static
void emitThumbRegPlusImmInReg(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator &MBBI,
                              DebugLoc dl,
                              unsigned DestReg, unsigned BaseReg,
                              int NumBytes, bool CanChangeCC,
                              const TargetInstrInfo &TII,
                              const ARMBaseRegisterInfo& MRI,
                              unsigned MIFlags = MachineInstr::NoFlags) {
    MachineFunction &MF = *MBB.getParent();
    bool isHigh = !isARMLowRegister(DestReg) ||
                  (BaseReg != 0 && !isARMLowRegister(BaseReg));
    bool isSub = false;
    // Subtract doesn't have high register version. Load the negative value
    // if either base or dest register is a high register. Also, if do not
    // issue sub as part of the sequence if condition register is to be
    // preserved.
    if (NumBytes < 0 && !isHigh && CanChangeCC) {
      isSub = true;
      NumBytes = -NumBytes;
    }
    unsigned LdReg = DestReg;
    if (DestReg == ARM::SP) {
      assert(BaseReg == ARM::SP && "Unexpected!");
      LdReg = MF.getRegInfo().createVirtualRegister(&ARM::tGPRRegClass);
    }

    if (NumBytes <= 255 && NumBytes >= 0)
      AddDefaultT1CC(BuildMI(MBB, MBBI, dl, TII.get(ARM::tMOVi8), LdReg))
        .addImm(NumBytes).setMIFlags(MIFlags);
    else if (NumBytes < 0 && NumBytes >= -255) {
      AddDefaultT1CC(BuildMI(MBB, MBBI, dl, TII.get(ARM::tMOVi8), LdReg))
        .addImm(NumBytes).setMIFlags(MIFlags);
      AddDefaultT1CC(BuildMI(MBB, MBBI, dl, TII.get(ARM::tRSB), LdReg))
        .addReg(LdReg, RegState::Kill).setMIFlags(MIFlags);
    } else
      MRI.emitLoadConstPool(MBB, MBBI, dl, LdReg, 0, NumBytes,
                            ARMCC::AL, 0, MIFlags);

    // Emit add / sub.
    int Opc = (isSub) ? ARM::tSUBrr : (isHigh ? ARM::tADDhirr : ARM::tADDrr);
    MachineInstrBuilder MIB =
      BuildMI(MBB, MBBI, dl, TII.get(Opc), DestReg);
    if (Opc != ARM::tADDhirr)
      MIB = AddDefaultT1CC(MIB);
    if (DestReg == ARM::SP || isSub)
      MIB.addReg(BaseReg).addReg(LdReg, RegState::Kill);
    else
      MIB.addReg(LdReg).addReg(BaseReg, RegState::Kill);
    AddDefaultPred(MIB);
}

/// calcNumMI - Returns the number of instructions required to materialize
/// the specific add / sub r, c instruction.
static unsigned calcNumMI(int Opc, int ExtraOpc, unsigned Bytes,
                          unsigned NumBits, unsigned Scale) {
  unsigned NumMIs = 0;
  unsigned Chunk = ((1 << NumBits) - 1) * Scale;

  if (Opc == ARM::tADDrSPi) {
    unsigned ThisVal = (Bytes > Chunk) ? Chunk : Bytes;
    Bytes -= ThisVal;
    NumMIs++;
    NumBits = 8;
    Scale = 1;  // Followed by a number of tADDi8.
    Chunk = ((1 << NumBits) - 1) * Scale;
  }

  NumMIs += Bytes / Chunk;
  if ((Bytes % Chunk) != 0)
    NumMIs++;
  if (ExtraOpc)
    NumMIs++;
  return NumMIs;
}

/// emitThumbRegPlusImmediate - Emits a series of instructions to materialize
/// a destreg = basereg + immediate in Thumb code.
void llvm::emitThumbRegPlusImmediate(MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator &MBBI,
                                     DebugLoc dl,
                                     unsigned DestReg, unsigned BaseReg,
                                     int NumBytes, const TargetInstrInfo &TII,
                                     const ARMBaseRegisterInfo& MRI,
                                     unsigned MIFlags) {
  bool isSub = NumBytes < 0;
  unsigned Bytes = (unsigned)NumBytes;
  if (isSub) Bytes = -NumBytes;
  bool isMul4 = (Bytes & 3) == 0;
  bool isTwoAddr = false;
  bool DstNotEqBase = false;
  unsigned NumBits = 1;
  unsigned Scale = 1;
  int Opc = 0;
  int ExtraOpc = 0;
  bool NeedCC = false;

  if (DestReg == BaseReg && BaseReg == ARM::SP) {
    assert(isMul4 && "Thumb sp inc / dec size must be multiple of 4!");
    NumBits = 7;
    Scale = 4;
    Opc = isSub ? ARM::tSUBspi : ARM::tADDspi;
    isTwoAddr = true;
  } else if (!isSub && BaseReg == ARM::SP) {
    // r1 = add sp, 403
    // =>
    // r1 = add sp, 100 * 4
    // r1 = add r1, 3
    if (!isMul4) {
      Bytes &= ~3;
      ExtraOpc = ARM::tADDi3;
    }
    NumBits = 8;
    Scale = 4;
    Opc = ARM::tADDrSPi;
  } else {
    // sp = sub sp, c
    // r1 = sub sp, c
    // r8 = sub sp, c
    if (DestReg != BaseReg)
      DstNotEqBase = true;
    NumBits = 8;
    if (DestReg == ARM::SP) {
      Opc = isSub ? ARM::tSUBspi : ARM::tADDspi;
      assert(isMul4 && "Thumb sp inc / dec size must be multiple of 4!");
      NumBits = 7;
      Scale = 4;
    } else {
      Opc = isSub ? ARM::tSUBi8 : ARM::tADDi8;
      NumBits = 8;
      NeedCC = true;
    }
    isTwoAddr = true;
  }

  unsigned NumMIs = calcNumMI(Opc, ExtraOpc, Bytes, NumBits, Scale);
  unsigned Threshold = (DestReg == ARM::SP) ? 3 : 2;
  if (NumMIs > Threshold) {
    // This will expand into too many instructions. Load the immediate from a
    // constpool entry.
    emitThumbRegPlusImmInReg(MBB, MBBI, dl,
                             DestReg, BaseReg, NumBytes, true,
                             TII, MRI, MIFlags);
    return;
  }

  if (DstNotEqBase) {
    if (isARMLowRegister(DestReg) && isARMLowRegister(BaseReg)) {
      // If both are low registers, emit DestReg = add BaseReg, max(Imm, 7)
      unsigned Chunk = (1 << 3) - 1;
      unsigned ThisVal = (Bytes > Chunk) ? Chunk : Bytes;
      Bytes -= ThisVal;
      const MCInstrDesc &MCID = TII.get(isSub ? ARM::tSUBi3 : ARM::tADDi3);
      const MachineInstrBuilder MIB =
        AddDefaultT1CC(BuildMI(MBB, MBBI, dl, MCID, DestReg)
                         .setMIFlags(MIFlags));
      AddDefaultPred(MIB.addReg(BaseReg, RegState::Kill).addImm(ThisVal));
    } else {
      AddDefaultPred(BuildMI(MBB, MBBI, dl, TII.get(ARM::tMOVr), DestReg)
        .addReg(BaseReg, RegState::Kill))
        .setMIFlags(MIFlags);
    }
    BaseReg = DestReg;
  }

  unsigned Chunk = ((1 << NumBits) - 1) * Scale;
  while (Bytes) {
    unsigned ThisVal = (Bytes > Chunk) ? Chunk : Bytes;
    Bytes -= ThisVal;
    ThisVal /= Scale;
    // Build the new tADD / tSUB.
    if (isTwoAddr) {
      MachineInstrBuilder MIB = BuildMI(MBB, MBBI, dl, TII.get(Opc), DestReg);
      if (NeedCC)
        MIB = AddDefaultT1CC(MIB);
      MIB.addReg(DestReg).addImm(ThisVal);
      MIB = AddDefaultPred(MIB);
      MIB.setMIFlags(MIFlags);
    } else {
      bool isKill = BaseReg != ARM::SP;
      MachineInstrBuilder MIB = BuildMI(MBB, MBBI, dl, TII.get(Opc), DestReg);
      if (NeedCC)
        MIB = AddDefaultT1CC(MIB);
      MIB.addReg(BaseReg, getKillRegState(isKill)).addImm(ThisVal);
      MIB = AddDefaultPred(MIB);
      MIB.setMIFlags(MIFlags);

      BaseReg = DestReg;
      if (Opc == ARM::tADDrSPi) {
        // r4 = add sp, imm
        // r4 = add r4, imm
        // ...
        NumBits = 8;
        Scale = 1;
        Chunk = ((1 << NumBits) - 1) * Scale;
        Opc = isSub ? ARM::tSUBi8 : ARM::tADDi8;
        NeedCC = isTwoAddr = true;
      }
    }
  }

  if (ExtraOpc) {
    const MCInstrDesc &MCID = TII.get(ExtraOpc);
    AddDefaultPred(AddDefaultT1CC(BuildMI(MBB, MBBI, dl, MCID, DestReg))
                   .addReg(DestReg, RegState::Kill)
                   .addImm(((unsigned)NumBytes) & 3)
                   .setMIFlags(MIFlags));
  }
}

static void emitSPUpdate(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator &MBBI,
                         const TargetInstrInfo &TII, DebugLoc dl,
                         const Thumb1RegisterInfo &MRI,
                         int NumBytes) {
  emitThumbRegPlusImmediate(MBB, MBBI, dl, ARM::SP, ARM::SP, NumBytes, TII,
                            MRI);
}

void Thumb1RegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();

  if (!TFI->hasReservedCallFrame(MF)) {
    // If we have alloca, convert as follows:
    // ADJCALLSTACKDOWN -> sub, sp, sp, amount
    // ADJCALLSTACKUP   -> add, sp, sp, amount
    MachineInstr *Old = I;
    DebugLoc dl = Old->getDebugLoc();
    unsigned Amount = Old->getOperand(0).getImm();
    if (Amount != 0) {
      // We need to keep the stack aligned properly.  To do this, we round the
      // amount of space needed for the outgoing arguments up to the next
      // alignment boundary.
      unsigned Align = TFI->getStackAlignment();
      Amount = (Amount+Align-1)/Align*Align;

      // Replace the pseudo instruction with a new instruction...
      unsigned Opc = Old->getOpcode();
      if (Opc == ARM::ADJCALLSTACKDOWN || Opc == ARM::tADJCALLSTACKDOWN) {
        emitSPUpdate(MBB, I, TII, dl, *this, -Amount);
      } else {
        assert(Opc == ARM::ADJCALLSTACKUP || Opc == ARM::tADJCALLSTACKUP);
        emitSPUpdate(MBB, I, TII, dl, *this, Amount);
      }
    }
  }
  MBB.erase(I);
}

/// emitThumbConstant - Emit a series of instructions to materialize a
/// constant.
static void emitThumbConstant(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator &MBBI,
                              unsigned DestReg, int Imm,
                              const TargetInstrInfo &TII,
                              const Thumb1RegisterInfo& MRI,
                              DebugLoc dl) {
  bool isSub = Imm < 0;
  if (isSub) Imm = -Imm;

  int Chunk = (1 << 8) - 1;
  int ThisVal = (Imm > Chunk) ? Chunk : Imm;
  Imm -= ThisVal;
  AddDefaultPred(AddDefaultT1CC(BuildMI(MBB, MBBI, dl, TII.get(ARM::tMOVi8),
                                        DestReg))
                 .addImm(ThisVal));
  if (Imm > 0)
    emitThumbRegPlusImmediate(MBB, MBBI, dl, DestReg, DestReg, Imm, TII, MRI);
  if (isSub) {
    const MCInstrDesc &MCID = TII.get(ARM::tRSB);
    AddDefaultPred(AddDefaultT1CC(BuildMI(MBB, MBBI, dl, MCID, DestReg))
                   .addReg(DestReg, RegState::Kill));
  }
}

static void removeOperands(MachineInstr &MI, unsigned i) {
  unsigned Op = i;
  for (unsigned e = MI.getNumOperands(); i != e; ++i)
    MI.RemoveOperand(Op);
}

/// convertToNonSPOpcode - Change the opcode to the non-SP version, because
/// we're replacing the frame index with a non-SP register.
static unsigned convertToNonSPOpcode(unsigned Opcode) {
  switch (Opcode) {
  case ARM::tLDRspi:
    return ARM::tLDRi;

  case ARM::tSTRspi:
    return ARM::tSTRi;
  }

  return Opcode;
}

bool Thumb1RegisterInfo::
rewriteFrameIndex(MachineBasicBlock::iterator II, unsigned FrameRegIdx,
                  unsigned FrameReg, int &Offset,
                  const ARMBaseInstrInfo &TII) const {
  MachineInstr &MI = *II;
  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc dl = MI.getDebugLoc();
  unsigned Opcode = MI.getOpcode();
  const MCInstrDesc &Desc = MI.getDesc();
  unsigned AddrMode = (Desc.TSFlags & ARMII::AddrModeMask);

  if (Opcode == ARM::tADDrSPi) {
    Offset += MI.getOperand(FrameRegIdx+1).getImm();

    // Can't use tADDrSPi if it's based off the frame pointer.
    unsigned NumBits = 0;
    unsigned Scale = 1;
    if (FrameReg != ARM::SP) {
      Opcode = ARM::tADDi3;
      NumBits = 3;
    } else {
      NumBits = 8;
      Scale = 4;
      assert((Offset & 3) == 0 &&
             "Thumb add/sub sp, #imm immediate must be multiple of 4!");
    }

    unsigned PredReg;
    if (Offset == 0 && getInstrPredicate(&MI, PredReg) == ARMCC::AL) {
      // Turn it into a move.
      MI.setDesc(TII.get(ARM::tMOVr));
      MI.getOperand(FrameRegIdx).ChangeToRegister(FrameReg, false);
      // Remove offset
      MI.RemoveOperand(FrameRegIdx+1);
      MachineInstrBuilder MIB(&MI);
      return true;
    }

    // Common case: small offset, fits into instruction.
    unsigned Mask = (1 << NumBits) - 1;
    if (((Offset / Scale) & ~Mask) == 0) {
      // Replace the FrameIndex with sp / fp
      if (Opcode == ARM::tADDi3) {
        MI.setDesc(TII.get(Opcode));
        removeOperands(MI, FrameRegIdx);
        MachineInstrBuilder MIB(&MI);
        AddDefaultPred(AddDefaultT1CC(MIB).addReg(FrameReg)
                       .addImm(Offset / Scale));
      } else {
        MI.getOperand(FrameRegIdx).ChangeToRegister(FrameReg, false);
        MI.getOperand(FrameRegIdx+1).ChangeToImmediate(Offset / Scale);
      }
      return true;
    }

    unsigned DestReg = MI.getOperand(0).getReg();
    unsigned Bytes = (Offset > 0) ? Offset : -Offset;
    unsigned NumMIs = calcNumMI(Opcode, 0, Bytes, NumBits, Scale);
    // MI would expand into a large number of instructions. Don't try to
    // simplify the immediate.
    if (NumMIs > 2) {
      emitThumbRegPlusImmediate(MBB, II, dl, DestReg, FrameReg, Offset, TII,
                                *this);
      MBB.erase(II);
      return true;
    }

    if (Offset > 0) {
      // Translate r0 = add sp, imm to
      // r0 = add sp, 255*4
      // r0 = add r0, (imm - 255*4)
      if (Opcode == ARM::tADDi3) {
        MI.setDesc(TII.get(Opcode));
        removeOperands(MI, FrameRegIdx);
        MachineInstrBuilder MIB(&MI);
        AddDefaultPred(AddDefaultT1CC(MIB).addReg(FrameReg).addImm(Mask));
      } else {
        MI.getOperand(FrameRegIdx).ChangeToRegister(FrameReg, false);
        MI.getOperand(FrameRegIdx+1).ChangeToImmediate(Mask);
      }
      Offset = (Offset - Mask * Scale);
      MachineBasicBlock::iterator NII = llvm::next(II);
      emitThumbRegPlusImmediate(MBB, NII, dl, DestReg, DestReg, Offset, TII,
                                *this);
    } else {
      // Translate r0 = add sp, -imm to
      // r0 = -imm (this is then translated into a series of instructons)
      // r0 = add r0, sp
      emitThumbConstant(MBB, II, DestReg, Offset, TII, *this, dl);

      MI.setDesc(TII.get(ARM::tADDhirr));
      MI.getOperand(FrameRegIdx).ChangeToRegister(DestReg, false, false, true);
      MI.getOperand(FrameRegIdx+1).ChangeToRegister(FrameReg, false);
    }
    return true;
  } else {
    if (AddrMode != ARMII::AddrModeT1_s)
      llvm_unreachable("Unsupported addressing mode!");

    unsigned ImmIdx = FrameRegIdx + 1;
    int InstrOffs = MI.getOperand(ImmIdx).getImm();
    unsigned NumBits = (FrameReg == ARM::SP) ? 8 : 5;
    unsigned Scale = 4;

    Offset += InstrOffs * Scale;
    assert((Offset & (Scale - 1)) == 0 && "Can't encode this offset!");

    // Common case: small offset, fits into instruction.
    MachineOperand &ImmOp = MI.getOperand(ImmIdx);
    int ImmedOffset = Offset / Scale;
    unsigned Mask = (1 << NumBits) - 1;

    if ((unsigned)Offset <= Mask * Scale) {
      // Replace the FrameIndex with the frame register (e.g., sp).
      MI.getOperand(FrameRegIdx).ChangeToRegister(FrameReg, false);
      ImmOp.ChangeToImmediate(ImmedOffset);

      // If we're using a register where sp was stored, convert the instruction
      // to the non-SP version.
      unsigned NewOpc = convertToNonSPOpcode(Opcode);
      if (NewOpc != Opcode && FrameReg != ARM::SP)
        MI.setDesc(TII.get(NewOpc));

      return true;
    }

    NumBits = 5;
    Mask = (1 << NumBits) - 1;

    // If this is a thumb spill / restore, we will be using a constpool load to
    // materialize the offset.
    if (Opcode == ARM::tLDRspi || Opcode == ARM::tSTRspi) {
      ImmOp.ChangeToImmediate(0);
    } else {
      // Otherwise, it didn't fit. Pull in what we can to simplify the immed.
      ImmedOffset = ImmedOffset & Mask;
      ImmOp.ChangeToImmediate(ImmedOffset);
      Offset &= ~(Mask * Scale);
    }
  }

  return Offset == 0;
}

void
Thumb1RegisterInfo::resolveFrameIndex(MachineBasicBlock::iterator I,
                                      unsigned BaseReg, int64_t Offset) const {
  MachineInstr &MI = *I;
  int Off = Offset; // ARM doesn't need the general 64-bit offsets
  unsigned i = 0;

  while (!MI.getOperand(i).isFI()) {
    ++i;
    assert(i < MI.getNumOperands() && "Instr doesn't have FrameIndex operand!");
  }
  bool Done = rewriteFrameIndex(MI, i, BaseReg, Off, TII);
  assert (Done && "Unable to resolve frame index!");
  (void)Done;
}

/// saveScavengerRegister - Spill the register so it can be used by the
/// register scavenger. Return true.
bool
Thumb1RegisterInfo::saveScavengerRegister(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator I,
                                          MachineBasicBlock::iterator &UseMI,
                                          const TargetRegisterClass *RC,
                                          unsigned Reg) const {
  // Thumb1 can't use the emergency spill slot on the stack because
  // ldr/str immediate offsets must be positive, and if we're referencing
  // off the frame pointer (if, for example, there are alloca() calls in
  // the function, the offset will be negative. Use R12 instead since that's
  // a call clobbered register that we know won't be used in Thumb1 mode.
  DebugLoc DL;
  AddDefaultPred(BuildMI(MBB, I, DL, TII.get(ARM::tMOVr))
    .addReg(ARM::R12, RegState::Define)
    .addReg(Reg, RegState::Kill));

  // The UseMI is where we would like to restore the register. If there's
  // interference with R12 before then, however, we'll need to restore it
  // before that instead and adjust the UseMI.
  bool done = false;
  for (MachineBasicBlock::iterator II = I; !done && II != UseMI ; ++II) {
    if (II->isDebugValue())
      continue;
    // If this instruction affects R12, adjust our restore point.
    for (unsigned i = 0, e = II->getNumOperands(); i != e; ++i) {
      const MachineOperand &MO = II->getOperand(i);
      if (MO.isRegMask() && MO.clobbersPhysReg(ARM::R12)) {
        UseMI = II;
        done = true;
        break;
      }
      if (!MO.isReg() || MO.isUndef() || !MO.getReg() ||
          TargetRegisterInfo::isVirtualRegister(MO.getReg()))
        continue;
      if (MO.getReg() == ARM::R12) {
        UseMI = II;
        done = true;
        break;
      }
    }
  }
  // Restore the register from R12
  AddDefaultPred(BuildMI(MBB, UseMI, DL, TII.get(ARM::tMOVr)).
    addReg(Reg, RegState::Define).addReg(ARM::R12, RegState::Kill));

  return true;
}

void
Thumb1RegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                        int SPAdj, RegScavenger *RS) const {
  unsigned VReg = 0;
  unsigned i = 0;
  MachineInstr &MI = *II;
  MachineBasicBlock &MBB = *MI.getParent();
  MachineFunction &MF = *MBB.getParent();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  DebugLoc dl = MI.getDebugLoc();

  while (!MI.getOperand(i).isFI()) {
    ++i;
    assert(i < MI.getNumOperands() && "Instr doesn't have FrameIndex operand!");
  }

  unsigned FrameReg = ARM::SP;
  int FrameIndex = MI.getOperand(i).getIndex();
  int Offset = MF.getFrameInfo()->getObjectOffset(FrameIndex) +
               MF.getFrameInfo()->getStackSize() + SPAdj;

  if (AFI->isGPRCalleeSavedArea1Frame(FrameIndex))
    Offset -= AFI->getGPRCalleeSavedArea1Offset();
  else if (AFI->isGPRCalleeSavedArea2Frame(FrameIndex))
    Offset -= AFI->getGPRCalleeSavedArea2Offset();
  else if (MF.getFrameInfo()->hasVarSizedObjects()) {
    assert(SPAdj == 0 && MF.getTarget().getFrameLowering()->hasFP(MF) &&
           "Unexpected");
    // There are alloca()'s in this function, must reference off the frame
    // pointer or base pointer instead.
    if (!hasBasePointer(MF)) {
      FrameReg = getFrameRegister(MF);
      Offset -= AFI->getFramePtrSpillOffset();
    } else
      FrameReg = BasePtr;
  }

  // PEI::scavengeFrameVirtualRegs() cannot accurately track SPAdj because the
  // call frame setup/destroy instructions have already been eliminated.  That
  // means the stack pointer cannot be used to access the emergency spill slot
  // when !hasReservedCallFrame().
#ifndef NDEBUG
  if (RS && FrameReg == ARM::SP && FrameIndex == RS->getScavengingFrameIndex()){
    assert(MF.getTarget().getFrameLowering()->hasReservedCallFrame(MF) &&
           "Cannot use SP to access the emergency spill slot in "
           "functions without a reserved call frame");
    assert(!MF.getFrameInfo()->hasVarSizedObjects() &&
           "Cannot use SP to access the emergency spill slot in "
           "functions with variable sized frame objects");
  }
#endif // NDEBUG

  // Special handling of dbg_value instructions.
  if (MI.isDebugValue()) {
    MI.getOperand(i).  ChangeToRegister(FrameReg, false /*isDef*/);
    MI.getOperand(i+1).ChangeToImmediate(Offset);
    return;
  }

  // Modify MI as necessary to handle as much of 'Offset' as possible
  assert(AFI->isThumbFunction() &&
         "This eliminateFrameIndex only supports Thumb1!");
  if (rewriteFrameIndex(MI, i, FrameReg, Offset, TII))
    return;

  // If we get here, the immediate doesn't fit into the instruction.  We folded
  // as much as possible above, handle the rest, providing a register that is
  // SP+LargeImm.
  assert(Offset && "This code isn't needed if offset already handled!");

  unsigned Opcode = MI.getOpcode();

  // Remove predicate first.
  int PIdx = MI.findFirstPredOperandIdx();
  if (PIdx != -1)
    removeOperands(MI, PIdx);

  if (MI.mayLoad()) {
    // Use the destination register to materialize sp + offset.
    unsigned TmpReg = MI.getOperand(0).getReg();
    bool UseRR = false;
    if (Opcode == ARM::tLDRspi) {
      if (FrameReg == ARM::SP)
        emitThumbRegPlusImmInReg(MBB, II, dl, TmpReg, FrameReg,
                                 Offset, false, TII, *this);
      else {
        emitLoadConstPool(MBB, II, dl, TmpReg, 0, Offset);
        UseRR = true;
      }
    } else {
      emitThumbRegPlusImmediate(MBB, II, dl, TmpReg, FrameReg, Offset, TII,
                                *this);
    }

    MI.setDesc(TII.get(UseRR ? ARM::tLDRr : ARM::tLDRi));
    MI.getOperand(i).ChangeToRegister(TmpReg, false, false, true);
    if (UseRR)
      // Use [reg, reg] addrmode. Replace the immediate operand w/ the frame
      // register. The offset is already handled in the vreg value.
      MI.getOperand(i+1).ChangeToRegister(FrameReg, false, false, false);
  } else if (MI.mayStore()) {
      VReg = MF.getRegInfo().createVirtualRegister(&ARM::tGPRRegClass);
      bool UseRR = false;

      if (Opcode == ARM::tSTRspi) {
        if (FrameReg == ARM::SP)
          emitThumbRegPlusImmInReg(MBB, II, dl, VReg, FrameReg,
                                   Offset, false, TII, *this);
        else {
          emitLoadConstPool(MBB, II, dl, VReg, 0, Offset);
          UseRR = true;
        }
      } else
        emitThumbRegPlusImmediate(MBB, II, dl, VReg, FrameReg, Offset, TII,
                                  *this);
      MI.setDesc(TII.get(UseRR ? ARM::tSTRr : ARM::tSTRi));
      MI.getOperand(i).ChangeToRegister(VReg, false, false, true);
      if (UseRR)
        // Use [reg, reg] addrmode. Replace the immediate operand w/ the frame
        // register. The offset is already handled in the vreg value.
        MI.getOperand(i+1).ChangeToRegister(FrameReg, false, false, false);
  } else {
    llvm_unreachable("Unexpected opcode!");
  }

  // Add predicate back if it's needed.
  if (MI.isPredicable()) {
    MachineInstrBuilder MIB(&MI);
    AddDefaultPred(MIB);
  }
}
