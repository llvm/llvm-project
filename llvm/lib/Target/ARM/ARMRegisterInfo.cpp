//===- ARMRegisterInfo.cpp - ARM Register Information -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the ARM implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "ARMAddressingModes.h"
#include "ARMInstrInfo.h"
#include "ARMMachineFunctionInfo.h"
#include "ARMRegisterInfo.h"
#include "ARMSubtarget.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include <algorithm>
using namespace llvm;

static cl::opt<bool> ThumbRegScavenging("enable-thumb-reg-scavenging",
                               cl::Hidden,
                               cl::desc("Enable register scavenging on Thumb"));

unsigned ARMRegisterInfo::getRegisterNumbering(unsigned RegEnum) {
  using namespace ARM;
  switch (RegEnum) {
  case R0:  case S0:  case D0:  return 0;
  case R1:  case S1:  case D1:  return 1;
  case R2:  case S2:  case D2:  return 2;
  case R3:  case S3:  case D3:  return 3;
  case R4:  case S4:  case D4:  return 4;
  case R5:  case S5:  case D5:  return 5;
  case R6:  case S6:  case D6:  return 6;
  case R7:  case S7:  case D7:  return 7;
  case R8:  case S8:  case D8:  return 8;
  case R9:  case S9:  case D9:  return 9;
  case R10: case S10: case D10: return 10;
  case R11: case S11: case D11: return 11;
  case R12: case S12: case D12: return 12;
  case SP:  case S13: case D13: return 13;
  case LR:  case S14: case D14: return 14;
  case PC:  case S15: case D15: return 15;
  case S16: return 16;
  case S17: return 17;
  case S18: return 18;
  case S19: return 19;
  case S20: return 20;
  case S21: return 21;
  case S22: return 22;
  case S23: return 23;
  case S24: return 24;
  case S25: return 25;
  case S26: return 26;
  case S27: return 27;
  case S28: return 28;
  case S29: return 29;
  case S30: return 30;
  case S31: return 31;
  default:
    assert(0 && "Unknown ARM register!");
    abort();
  }
}

unsigned ARMRegisterInfo::getRegisterNumbering(unsigned RegEnum,
                                               bool &isSPVFP) {
  isSPVFP = false;

  using namespace ARM;
  switch (RegEnum) {
  default:
    assert(0 && "Unknown ARM register!");
    abort();
  case R0:  case D0:  return 0;
  case R1:  case D1:  return 1;
  case R2:  case D2:  return 2;
  case R3:  case D3:  return 3;
  case R4:  case D4:  return 4;
  case R5:  case D5:  return 5;
  case R6:  case D6:  return 6;
  case R7:  case D7:  return 7;
  case R8:  case D8:  return 8;
  case R9:  case D9:  return 9;
  case R10: case D10: return 10;
  case R11: case D11: return 11;
  case R12: case D12: return 12;
  case SP:  case D13: return 13;
  case LR:  case D14: return 14;
  case PC:  case D15: return 15;

  case S0: case S1: case S2: case S3:
  case S4: case S5: case S6: case S7: 
  case S8: case S9: case S10: case S11: 
  case S12: case S13: case S14: case S15: 
  case S16: case S17: case S18: case S19: 
  case S20: case S21: case S22: case S23: 
  case S24: case S25: case S26: case S27: 
  case S28: case S29: case S30: case S31:  {
    isSPVFP = true;
    switch (RegEnum) {
    default: return 0; // Avoid compile time warning.
    case S0: return 0;
    case S1: return 1;
    case S2: return 2;
    case S3: return 3;
    case S4: return 4;
    case S5: return 5;
    case S6: return 6;
    case S7: return 7;
    case S8: return 8;
    case S9: return 9;
    case S10: return 10;
    case S11: return 11;
    case S12: return 12;
    case S13: return 13;
    case S14: return 14;
    case S15: return 15;
    case S16: return 16;
    case S17: return 17;
    case S18: return 18;
    case S19: return 19;
    case S20: return 20;
    case S21: return 21;
    case S22: return 22;
    case S23: return 23;
    case S24: return 24;
    case S25: return 25;
    case S26: return 26;
    case S27: return 27;
    case S28: return 28;
    case S29: return 29;
    case S30: return 30;
    case S31: return 31;
    }
  }
  }
}

ARMRegisterInfo::ARMRegisterInfo(const TargetInstrInfo &tii,
                                 const ARMSubtarget &sti)
  : ARMGenRegisterInfo(ARM::ADJCALLSTACKDOWN, ARM::ADJCALLSTACKUP),
    TII(tii), STI(sti),
    FramePtr((STI.useThumbBacktraces() || STI.isThumb()) ? ARM::R7 : ARM::R11) {
}

static inline
const MachineInstrBuilder &AddDefaultPred(const MachineInstrBuilder &MIB) {
  return MIB.addImm((int64_t)ARMCC::AL).addReg(0);
}

static inline
const MachineInstrBuilder &AddDefaultCC(const MachineInstrBuilder &MIB) {
  return MIB.addReg(0);
}

/// emitLoadConstPool - Emits a load from constpool to materialize the
/// specified immediate.
void ARMRegisterInfo::emitLoadConstPool(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator &MBBI,
                                        unsigned DestReg, int Val,
                                        unsigned Pred, unsigned PredReg,
                                        const TargetInstrInfo *TII,
                                        bool isThumb) const {
  MachineFunction &MF = *MBB.getParent();
  MachineConstantPool *ConstantPool = MF.getConstantPool();
  Constant *C = ConstantInt::get(Type::Int32Ty, Val);
  unsigned Idx = ConstantPool->getConstantPoolIndex(C, 2);
  if (isThumb)
    BuildMI(MBB, MBBI, TII->get(ARM::tLDRcp),DestReg).addConstantPoolIndex(Idx);
  else
    BuildMI(MBB, MBBI, TII->get(ARM::LDRcp), DestReg).addConstantPoolIndex(Idx)
      .addReg(0).addImm(0).addImm(Pred).addReg(PredReg);
}

/// isLowRegister - Returns true if the register is low register r0-r7.
///
bool ARMRegisterInfo::isLowRegister(unsigned Reg) const {
  using namespace ARM;
  switch (Reg) {
  case R0:  case R1:  case R2:  case R3:
  case R4:  case R5:  case R6:  case R7:
    return true;
  default:
    return false;
  }
}

const unsigned*
ARMRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  static const unsigned CalleeSavedRegs[] = {
    ARM::LR, ARM::R11, ARM::R10, ARM::R9, ARM::R8,
    ARM::R7, ARM::R6,  ARM::R5,  ARM::R4,

    ARM::D15, ARM::D14, ARM::D13, ARM::D12,
    ARM::D11, ARM::D10, ARM::D9,  ARM::D8,
    0
  };

  static const unsigned DarwinCalleeSavedRegs[] = {
    ARM::LR,  ARM::R7,  ARM::R6, ARM::R5, ARM::R4,
    ARM::R11, ARM::R10, ARM::R9, ARM::R8,

    ARM::D15, ARM::D14, ARM::D13, ARM::D12,
    ARM::D11, ARM::D10, ARM::D9,  ARM::D8,
    0
  };
  return STI.isTargetDarwin() ? DarwinCalleeSavedRegs : CalleeSavedRegs;
}

const TargetRegisterClass* const *
ARMRegisterInfo::getCalleeSavedRegClasses(const MachineFunction *MF) const {
  static const TargetRegisterClass * const CalleeSavedRegClasses[] = {
    &ARM::GPRRegClass, &ARM::GPRRegClass, &ARM::GPRRegClass,
    &ARM::GPRRegClass, &ARM::GPRRegClass, &ARM::GPRRegClass,
    &ARM::GPRRegClass, &ARM::GPRRegClass, &ARM::GPRRegClass,

    &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass,
    &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass,
    0
  };
  return CalleeSavedRegClasses;
}

BitVector ARMRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  // FIXME: avoid re-calculating this everytime.
  BitVector Reserved(getNumRegs());
  Reserved.set(ARM::SP);
  Reserved.set(ARM::PC);
  if (STI.isTargetDarwin() || hasFP(MF))
    Reserved.set(FramePtr);
  // Some targets reserve R9.
  if (STI.isR9Reserved())
    Reserved.set(ARM::R9);
  return Reserved;
}

bool
ARMRegisterInfo::isReservedReg(const MachineFunction &MF, unsigned Reg) const {
  switch (Reg) {
  default: break;
  case ARM::SP:
  case ARM::PC:
    return true;
  case ARM::R7:
  case ARM::R11:
    if (FramePtr == Reg && (STI.isTargetDarwin() || hasFP(MF)))
      return true;
    break;
  case ARM::R9:
    return STI.isR9Reserved();
  }

  return false;
}

bool
ARMRegisterInfo::requiresRegisterScavenging(const MachineFunction &MF) const {
  const ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  return ThumbRegScavenging || !AFI->isThumbFunction();
}

/// hasFP - Return true if the specified function should have a dedicated frame
/// pointer register.  This is true if the function has variable sized allocas
/// or if frame pointer elimination is disabled.
///
bool ARMRegisterInfo::hasFP(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  return NoFramePointerElim || MFI->hasVarSizedObjects();
}

// hasReservedCallFrame - Under normal circumstances, when a frame pointer is
// not required, we reserve argument space for call sites in the function
// immediately on entry to the current function. This eliminates the need for
// add/sub sp brackets around call sites. Returns true if the call frame is
// included as part of the stack frame.
bool ARMRegisterInfo::hasReservedCallFrame(MachineFunction &MF) const {
  const MachineFrameInfo *FFI = MF.getFrameInfo();
  unsigned CFSize = FFI->getMaxCallFrameSize();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  // It's not always a good idea to include the call frame as part of the
  // stack frame. ARM (especially Thumb) has small immediate offset to
  // address the stack frame. So a large call frame can cause poor codegen
  // and may even makes it impossible to scavenge a register.
  if (AFI->isThumbFunction()) {
    if (CFSize >= ((1 << 8) - 1) * 4 / 2) // Half of imm8 * 4
      return false;
  } else {
    if (CFSize >= ((1 << 12) - 1) / 2)  // Half of imm12
      return false;
  }
  return !MF.getFrameInfo()->hasVarSizedObjects();
}

/// emitARMRegPlusImmediate - Emits a series of instructions to materialize
/// a destreg = basereg + immediate in ARM code.
static
void emitARMRegPlusImmediate(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator &MBBI,
                             unsigned DestReg, unsigned BaseReg, int NumBytes,
                             ARMCC::CondCodes Pred, unsigned PredReg,
                             const TargetInstrInfo &TII) {
  bool isSub = NumBytes < 0;
  if (isSub) NumBytes = -NumBytes;

  while (NumBytes) {
    unsigned RotAmt = ARM_AM::getSOImmValRotate(NumBytes);
    unsigned ThisVal = NumBytes & ARM_AM::rotr32(0xFF, RotAmt);
    assert(ThisVal && "Didn't extract field correctly");
    
    // We will handle these bits from offset, clear them.
    NumBytes &= ~ThisVal;
    
    // Get the properly encoded SOImmVal field.
    int SOImmVal = ARM_AM::getSOImmVal(ThisVal);
    assert(SOImmVal != -1 && "Bit extraction didn't work?");
    
    // Build the new ADD / SUB.
    BuildMI(MBB, MBBI, TII.get(isSub ? ARM::SUBri : ARM::ADDri), DestReg)
      .addReg(BaseReg, false, false, true).addImm(SOImmVal)
      .addImm((unsigned)Pred).addReg(PredReg).addReg(0);
    BaseReg = DestReg;
  }
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

/// emitThumbRegPlusImmInReg - Emits a series of instructions to materialize
/// a destreg = basereg + immediate in Thumb code. Materialize the immediate
/// in a register using mov / mvn sequences or load the immediate from a
/// constpool entry.
static
void emitThumbRegPlusImmInReg(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator &MBBI,
                              unsigned DestReg, unsigned BaseReg,
                              int NumBytes, bool CanChangeCC,
                              const TargetInstrInfo &TII,
                              const ARMRegisterInfo& MRI) {
    bool isHigh = !MRI.isLowRegister(DestReg) ||
                  (BaseReg != 0 && !MRI.isLowRegister(BaseReg));
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
      LdReg = ARM::R3;
      BuildMI(MBB, MBBI, TII.get(ARM::tMOVr), ARM::R12)
        .addReg(ARM::R3, false, false, true);
    }

    if (NumBytes <= 255 && NumBytes >= 0)
      BuildMI(MBB, MBBI, TII.get(ARM::tMOVi8), LdReg).addImm(NumBytes);
    else if (NumBytes < 0 && NumBytes >= -255) {
      BuildMI(MBB, MBBI, TII.get(ARM::tMOVi8), LdReg).addImm(NumBytes);
      BuildMI(MBB, MBBI, TII.get(ARM::tNEG), LdReg)
        .addReg(LdReg, false, false, true);
    } else
      MRI.emitLoadConstPool(MBB, MBBI, LdReg, NumBytes, ARMCC::AL, 0,&TII,true);

    // Emit add / sub.
    int Opc = (isSub) ? ARM::tSUBrr : (isHigh ? ARM::tADDhirr : ARM::tADDrr);
    const MachineInstrBuilder MIB = BuildMI(MBB, MBBI, TII.get(Opc), DestReg);
    if (DestReg == ARM::SP || isSub)
      MIB.addReg(BaseReg).addReg(LdReg, false, false, true);
    else
      MIB.addReg(LdReg).addReg(BaseReg, false, false, true);
    if (DestReg == ARM::SP)
      BuildMI(MBB, MBBI, TII.get(ARM::tMOVr), ARM::R3)
        .addReg(ARM::R12, false, false, true);
}

/// emitThumbRegPlusImmediate - Emits a series of instructions to materialize
/// a destreg = basereg + immediate in Thumb code.
static
void emitThumbRegPlusImmediate(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator &MBBI,
                               unsigned DestReg, unsigned BaseReg,
                               int NumBytes, const TargetInstrInfo &TII,
                               const ARMRegisterInfo& MRI) {
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
    Opc = isSub ? ARM::tSUBi8 : ARM::tADDi8;
    isTwoAddr = true;
  }

  unsigned NumMIs = calcNumMI(Opc, ExtraOpc, Bytes, NumBits, Scale);
  unsigned Threshold = (DestReg == ARM::SP) ? 3 : 2;
  if (NumMIs > Threshold) {
    // This will expand into too many instructions. Load the immediate from a
    // constpool entry.
    emitThumbRegPlusImmInReg(MBB, MBBI, DestReg, BaseReg, NumBytes, true, TII, MRI);
    return;
  }

  if (DstNotEqBase) {
    if (MRI.isLowRegister(DestReg) && MRI.isLowRegister(BaseReg)) {
      // If both are low registers, emit DestReg = add BaseReg, max(Imm, 7)
      unsigned Chunk = (1 << 3) - 1;
      unsigned ThisVal = (Bytes > Chunk) ? Chunk : Bytes;
      Bytes -= ThisVal;
      BuildMI(MBB, MBBI, TII.get(isSub ? ARM::tSUBi3 : ARM::tADDi3), DestReg)
        .addReg(BaseReg, false, false, true).addImm(ThisVal);
    } else {
      BuildMI(MBB, MBBI, TII.get(ARM::tMOVr), DestReg)
        .addReg(BaseReg, false, false, true);
    }
    BaseReg = DestReg;
  }

  unsigned Chunk = ((1 << NumBits) - 1) * Scale;
  while (Bytes) {
    unsigned ThisVal = (Bytes > Chunk) ? Chunk : Bytes;
    Bytes -= ThisVal;
    ThisVal /= Scale;
    // Build the new tADD / tSUB.
    if (isTwoAddr)
      BuildMI(MBB, MBBI, TII.get(Opc), DestReg).addReg(DestReg).addImm(ThisVal);
    else {
      bool isKill = BaseReg != ARM::SP;
      BuildMI(MBB, MBBI, TII.get(Opc), DestReg)
        .addReg(BaseReg, false, false, isKill).addImm(ThisVal);
      BaseReg = DestReg;

      if (Opc == ARM::tADDrSPi) {
        // r4 = add sp, imm
        // r4 = add r4, imm
        // ...
        NumBits = 8;
        Scale = 1;
        Chunk = ((1 << NumBits) - 1) * Scale;
        Opc = isSub ? ARM::tSUBi8 : ARM::tADDi8;
        isTwoAddr = true;
      }
    }
  }

  if (ExtraOpc)
    BuildMI(MBB, MBBI, TII.get(ExtraOpc), DestReg)
      .addReg(DestReg, false, false, true)
      .addImm(((unsigned)NumBytes) & 3);
}

static
void emitSPUpdate(MachineBasicBlock &MBB, MachineBasicBlock::iterator &MBBI,
                  int NumBytes, ARMCC::CondCodes Pred, unsigned PredReg,
                  bool isThumb, const TargetInstrInfo &TII, 
                  const ARMRegisterInfo& MRI) {
  if (isThumb)
    emitThumbRegPlusImmediate(MBB, MBBI, ARM::SP, ARM::SP, NumBytes, TII, MRI);
  else
    emitARMRegPlusImmediate(MBB, MBBI, ARM::SP, ARM::SP, NumBytes,
                            Pred, PredReg, TII);
}

void ARMRegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  if (!hasReservedCallFrame(MF)) {
    // If we have alloca, convert as follows:
    // ADJCALLSTACKDOWN -> sub, sp, sp, amount
    // ADJCALLSTACKUP   -> add, sp, sp, amount
    MachineInstr *Old = I;
    unsigned Amount = Old->getOperand(0).getImm();
    if (Amount != 0) {
      ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
      // We need to keep the stack aligned properly.  To do this, we round the
      // amount of space needed for the outgoing arguments up to the next
      // alignment boundary.
      unsigned Align = MF.getTarget().getFrameInfo()->getStackAlignment();
      Amount = (Amount+Align-1)/Align*Align;

      // Replace the pseudo instruction with a new instruction...
      unsigned Opc = Old->getOpcode();
      bool isThumb = AFI->isThumbFunction();
      ARMCC::CondCodes Pred = isThumb
        ? ARMCC::AL : (ARMCC::CondCodes)Old->getOperand(1).getImm();
      if (Opc == ARM::ADJCALLSTACKDOWN || Opc == ARM::tADJCALLSTACKDOWN) {
        // Note: PredReg is operand 2 for ADJCALLSTACKDOWN.
        unsigned PredReg = isThumb ? 0 : Old->getOperand(2).getReg();
        emitSPUpdate(MBB, I, -Amount, Pred, PredReg, isThumb, TII, *this);
      } else {
        // Note: PredReg is operand 3 for ADJCALLSTACKUP.
        unsigned PredReg = isThumb ? 0 : Old->getOperand(3).getReg();
        assert(Opc == ARM::ADJCALLSTACKUP || Opc == ARM::tADJCALLSTACKUP);
        emitSPUpdate(MBB, I, Amount, Pred, PredReg, isThumb, TII, *this);
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
                              const ARMRegisterInfo& MRI) {
  bool isSub = Imm < 0;
  if (isSub) Imm = -Imm;

  int Chunk = (1 << 8) - 1;
  int ThisVal = (Imm > Chunk) ? Chunk : Imm;
  Imm -= ThisVal;
  BuildMI(MBB, MBBI, TII.get(ARM::tMOVi8), DestReg).addImm(ThisVal);
  if (Imm > 0) 
    emitThumbRegPlusImmediate(MBB, MBBI, DestReg, DestReg, Imm, TII, MRI);
  if (isSub)
    BuildMI(MBB, MBBI, TII.get(ARM::tNEG), DestReg)
      .addReg(DestReg, false, false, true);
}

/// findScratchRegister - Find a 'free' ARM register. If register scavenger
/// is not being used, R12 is available. Otherwise, try for a call-clobbered
/// register first and then a spilled callee-saved register if that fails.
static
unsigned findScratchRegister(RegScavenger *RS, const TargetRegisterClass *RC,
                             ARMFunctionInfo *AFI) {
  unsigned Reg = RS ? RS->FindUnusedReg(RC, true) : (unsigned) ARM::R12;
  if (Reg == 0)
    // Try a already spilled CS register.
    Reg = RS->FindUnusedReg(RC, AFI->getSpilledCSRegisters());

  return Reg;
}

void ARMRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                          int SPAdj, RegScavenger *RS) const{
  unsigned i = 0;
  MachineInstr &MI = *II;
  MachineBasicBlock &MBB = *MI.getParent();
  MachineFunction &MF = *MBB.getParent();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  bool isThumb = AFI->isThumbFunction();

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
  else if (AFI->isDPRCalleeSavedAreaFrame(FrameIndex))
    Offset -= AFI->getDPRCalleeSavedAreaOffset();
  else if (hasFP(MF)) {
    assert(SPAdj == 0 && "Unexpected");
    // There is alloca()'s in this function, must reference off the frame
    // pointer instead.
    FrameReg = getFrameRegister(MF);
    Offset -= AFI->getFramePtrSpillOffset();
  }

  unsigned Opcode = MI.getOpcode();
  const TargetInstrDesc &Desc = MI.getDesc();
  unsigned AddrMode = (Desc.TSFlags & ARMII::AddrModeMask);
  bool isSub = false;

  if (Opcode == ARM::ADDri) {
    Offset += MI.getOperand(i+1).getImm();
    if (Offset == 0) {
      // Turn it into a move.
      MI.setDesc(TII.get(ARM::MOVr));
      MI.getOperand(i).ChangeToRegister(FrameReg, false);
      MI.RemoveOperand(i+1);
      return;
    } else if (Offset < 0) {
      Offset = -Offset;
      isSub = true;
      MI.setDesc(TII.get(ARM::SUBri));
    }

    // Common case: small offset, fits into instruction.
    int ImmedOffset = ARM_AM::getSOImmVal(Offset);
    if (ImmedOffset != -1) {
      // Replace the FrameIndex with sp / fp
      MI.getOperand(i).ChangeToRegister(FrameReg, false);
      MI.getOperand(i+1).ChangeToImmediate(ImmedOffset);
      return;
    }
    
    // Otherwise, we fallback to common code below to form the imm offset with
    // a sequence of ADDri instructions.  First though, pull as much of the imm
    // into this ADDri as possible.
    unsigned RotAmt = ARM_AM::getSOImmValRotate(Offset);
    unsigned ThisImmVal = Offset & ARM_AM::rotr32(0xFF, RotAmt);
    
    // We will handle these bits from offset, clear them.
    Offset &= ~ThisImmVal;
    
    // Get the properly encoded SOImmVal field.
    int ThisSOImmVal = ARM_AM::getSOImmVal(ThisImmVal);
    assert(ThisSOImmVal != -1 && "Bit extraction didn't work?");    
    MI.getOperand(i+1).ChangeToImmediate(ThisSOImmVal);
  } else if (Opcode == ARM::tADDrSPi) {
    Offset += MI.getOperand(i+1).getImm();

    // Can't use tADDrSPi if it's based off the frame pointer.
    unsigned NumBits = 0;
    unsigned Scale = 1;
    if (FrameReg != ARM::SP) {
      Opcode = ARM::tADDi3;
      MI.setDesc(TII.get(ARM::tADDi3));
      NumBits = 3;
    } else {
      NumBits = 8;
      Scale = 4;
      assert((Offset & 3) == 0 &&
             "Thumb add/sub sp, #imm immediate must be multiple of 4!");
    }

    if (Offset == 0) {
      // Turn it into a move.
      MI.setDesc(TII.get(ARM::tMOVr));
      MI.getOperand(i).ChangeToRegister(FrameReg, false);
      MI.RemoveOperand(i+1);
      return;
    }

    // Common case: small offset, fits into instruction.
    unsigned Mask = (1 << NumBits) - 1;
    if (((Offset / Scale) & ~Mask) == 0) {
      // Replace the FrameIndex with sp / fp
      MI.getOperand(i).ChangeToRegister(FrameReg, false);
      MI.getOperand(i+1).ChangeToImmediate(Offset / Scale);
      return;
    }

    unsigned DestReg = MI.getOperand(0).getReg();
    unsigned Bytes = (Offset > 0) ? Offset : -Offset;
    unsigned NumMIs = calcNumMI(Opcode, 0, Bytes, NumBits, Scale);
    // MI would expand into a large number of instructions. Don't try to
    // simplify the immediate.
    if (NumMIs > 2) {
      emitThumbRegPlusImmediate(MBB, II, DestReg, FrameReg, Offset, TII, *this);
      MBB.erase(II);
      return;
    }

    if (Offset > 0) {
      // Translate r0 = add sp, imm to
      // r0 = add sp, 255*4
      // r0 = add r0, (imm - 255*4)
      MI.getOperand(i).ChangeToRegister(FrameReg, false);
      MI.getOperand(i+1).ChangeToImmediate(Mask);
      Offset = (Offset - Mask * Scale);
      MachineBasicBlock::iterator NII = next(II);
      emitThumbRegPlusImmediate(MBB, NII, DestReg, DestReg, Offset, TII, *this);
    } else {
      // Translate r0 = add sp, -imm to
      // r0 = -imm (this is then translated into a series of instructons)
      // r0 = add r0, sp
      emitThumbConstant(MBB, II, DestReg, Offset, TII, *this);
      MI.setDesc(TII.get(ARM::tADDhirr));
      MI.getOperand(i).ChangeToRegister(DestReg, false, false, true);
      MI.getOperand(i+1).ChangeToRegister(FrameReg, false);
    }
    return;
  } else {
    unsigned ImmIdx = 0;
    int InstrOffs = 0;
    unsigned NumBits = 0;
    unsigned Scale = 1;
    switch (AddrMode) {
    case ARMII::AddrMode2: {
      ImmIdx = i+2;
      InstrOffs = ARM_AM::getAM2Offset(MI.getOperand(ImmIdx).getImm());
      if (ARM_AM::getAM2Op(MI.getOperand(ImmIdx).getImm()) == ARM_AM::sub)
        InstrOffs *= -1;
      NumBits = 12;
      break;
    }
    case ARMII::AddrMode3: {
      ImmIdx = i+2;
      InstrOffs = ARM_AM::getAM3Offset(MI.getOperand(ImmIdx).getImm());
      if (ARM_AM::getAM3Op(MI.getOperand(ImmIdx).getImm()) == ARM_AM::sub)
        InstrOffs *= -1;
      NumBits = 8;
      break;
    }
    case ARMII::AddrMode5: {
      ImmIdx = i+1;
      InstrOffs = ARM_AM::getAM5Offset(MI.getOperand(ImmIdx).getImm());
      if (ARM_AM::getAM5Op(MI.getOperand(ImmIdx).getImm()) == ARM_AM::sub)
        InstrOffs *= -1;
      NumBits = 8;
      Scale = 4;
      break;
    }
    case ARMII::AddrModeTs: {
      ImmIdx = i+1;
      InstrOffs = MI.getOperand(ImmIdx).getImm();
      NumBits = (FrameReg == ARM::SP) ? 8 : 5;
      Scale = 4;
      break;
    }
    default:
      assert(0 && "Unsupported addressing mode!");
      abort();
      break;
    }

    Offset += InstrOffs * Scale;
    assert((Offset & (Scale-1)) == 0 && "Can't encode this offset!");
    if (Offset < 0 && !isThumb) {
      Offset = -Offset;
      isSub = true;
    }

    // Common case: small offset, fits into instruction.
    MachineOperand &ImmOp = MI.getOperand(ImmIdx);
    int ImmedOffset = Offset / Scale;
    unsigned Mask = (1 << NumBits) - 1;
    if ((unsigned)Offset <= Mask * Scale) {
      // Replace the FrameIndex with sp
      MI.getOperand(i).ChangeToRegister(FrameReg, false);
      if (isSub)
        ImmedOffset |= 1 << NumBits;
      ImmOp.ChangeToImmediate(ImmedOffset);
      return;
    }

    bool isThumSpillRestore = Opcode == ARM::tRestore || Opcode == ARM::tSpill;
    if (AddrMode == ARMII::AddrModeTs) {
      // Thumb tLDRspi, tSTRspi. These will change to instructions that use
      // a different base register.
      NumBits = 5;
      Mask = (1 << NumBits) - 1;
    }
    // If this is a thumb spill / restore, we will be using a constpool load to
    // materialize the offset.
    if (AddrMode == ARMII::AddrModeTs && isThumSpillRestore)
      ImmOp.ChangeToImmediate(0);
    else {
      // Otherwise, it didn't fit. Pull in what we can to simplify the immed.
      ImmedOffset = ImmedOffset & Mask;
      if (isSub)
        ImmedOffset |= 1 << NumBits;
      ImmOp.ChangeToImmediate(ImmedOffset);
      Offset &= ~(Mask*Scale);
    }
  }
  
  // If we get here, the immediate doesn't fit into the instruction.  We folded
  // as much as possible above, handle the rest, providing a register that is
  // SP+LargeImm.
  assert(Offset && "This code isn't needed if offset already handled!");

  if (isThumb) {
    if (Desc.mayLoad()) {
      // Use the destination register to materialize sp + offset.
      unsigned TmpReg = MI.getOperand(0).getReg();
      bool UseRR = false;
      if (Opcode == ARM::tRestore) {
        if (FrameReg == ARM::SP)
          emitThumbRegPlusImmInReg(MBB, II, TmpReg, FrameReg,
                                   Offset, false, TII, *this);
        else {
          emitLoadConstPool(MBB, II, TmpReg, Offset, ARMCC::AL, 0, &TII, true);
          UseRR = true;
        }
      } else
        emitThumbRegPlusImmediate(MBB, II, TmpReg, FrameReg, Offset, TII, *this);
      MI.setDesc(TII.get(ARM::tLDR));
      MI.getOperand(i).ChangeToRegister(TmpReg, false, false, true);
      if (UseRR)
        // Use [reg, reg] addrmode.
        MI.addOperand(MachineOperand::CreateReg(FrameReg, false));
      else  // tLDR has an extra register operand.
        MI.addOperand(MachineOperand::CreateReg(0, false));
    } else if (Desc.mayStore()) {
      // FIXME! This is horrific!!! We need register scavenging.
      // Our temporary workaround has marked r3 unavailable. Of course, r3 is
      // also a ABI register so it's possible that is is the register that is
      // being storing here. If that's the case, we do the following:
      // r12 = r2
      // Use r2 to materialize sp + offset
      // str r3, r2
      // r2 = r12
      unsigned ValReg = MI.getOperand(0).getReg();
      unsigned TmpReg = ARM::R3;
      bool UseRR = false;
      if (ValReg == ARM::R3) {
        BuildMI(MBB, II, TII.get(ARM::tMOVr), ARM::R12)
          .addReg(ARM::R2, false, false, true);
        TmpReg = ARM::R2;
      }
      if (TmpReg == ARM::R3 && AFI->isR3LiveIn())
        BuildMI(MBB, II, TII.get(ARM::tMOVr), ARM::R12)
          .addReg(ARM::R3, false, false, true);
      if (Opcode == ARM::tSpill) {
        if (FrameReg == ARM::SP)
          emitThumbRegPlusImmInReg(MBB, II, TmpReg, FrameReg,
                                   Offset, false, TII, *this);
        else {
          emitLoadConstPool(MBB, II, TmpReg, Offset, ARMCC::AL, 0, &TII, true);
          UseRR = true;
        }
      } else
        emitThumbRegPlusImmediate(MBB, II, TmpReg, FrameReg, Offset, TII, *this);
      MI.setDesc(TII.get(ARM::tSTR));
      MI.getOperand(i).ChangeToRegister(TmpReg, false, false, true);
      if (UseRR)  // Use [reg, reg] addrmode.
        MI.addOperand(MachineOperand::CreateReg(FrameReg, false));
      else // tSTR has an extra register operand.
        MI.addOperand(MachineOperand::CreateReg(0, false));

      MachineBasicBlock::iterator NII = next(II);
      if (ValReg == ARM::R3)
        BuildMI(MBB, NII, TII.get(ARM::tMOVr), ARM::R2)
          .addReg(ARM::R12, false, false, true);
      if (TmpReg == ARM::R3 && AFI->isR3LiveIn())
        BuildMI(MBB, NII, TII.get(ARM::tMOVr), ARM::R3)
          .addReg(ARM::R12, false, false, true);
    } else
      assert(false && "Unexpected opcode!");
  } else {
    // Insert a set of r12 with the full address: r12 = sp + offset
    // If the offset we have is too large to fit into the instruction, we need
    // to form it with a series of ADDri's.  Do this by taking 8-bit chunks
    // out of 'Offset'.
    unsigned ScratchReg = findScratchRegister(RS, &ARM::GPRRegClass, AFI);
    if (ScratchReg == 0)
      // No register is "free". Scavenge a register.
      ScratchReg = RS->scavengeRegister(&ARM::GPRRegClass, II, SPAdj);
    int PIdx = MI.findFirstPredOperandIdx();
    ARMCC::CondCodes Pred = (PIdx == -1)
      ? ARMCC::AL : (ARMCC::CondCodes)MI.getOperand(PIdx).getImm();
    unsigned PredReg = (PIdx == -1) ? 0 : MI.getOperand(PIdx+1).getReg();
    emitARMRegPlusImmediate(MBB, II, ScratchReg, FrameReg,
                            isSub ? -Offset : Offset, Pred, PredReg, TII);
    MI.getOperand(i).ChangeToRegister(ScratchReg, false, false, true);
  }
}

static unsigned estimateStackSize(MachineFunction &MF, MachineFrameInfo *MFI) {
  const MachineFrameInfo *FFI = MF.getFrameInfo();
  int Offset = 0;
  for (int i = FFI->getObjectIndexBegin(); i != 0; ++i) {
    int FixedOff = -FFI->getObjectOffset(i);
    if (FixedOff > Offset) Offset = FixedOff;
  }
  for (unsigned i = 0, e = FFI->getObjectIndexEnd(); i != e; ++i) {
    if (FFI->isDeadObjectIndex(i))
      continue;
    Offset += FFI->getObjectSize(i);
    unsigned Align = FFI->getObjectAlignment(i);
    // Adjust to alignment boundary
    Offset = (Offset+Align-1)/Align*Align;
  }
  return (unsigned)Offset;
}

void
ARMRegisterInfo::processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                                      RegScavenger *RS) const {
  // This tells PEI to spill the FP as if it is any other callee-save register
  // to take advantage the eliminateFrameIndex machinery. This also ensures it
  // is spilled in the order specified by getCalleeSavedRegs() to make it easier
  // to combine multiple loads / stores.
  bool CanEliminateFrame = true;
  bool CS1Spilled = false;
  bool LRSpilled = false;
  unsigned NumGPRSpills = 0;
  SmallVector<unsigned, 4> UnspilledCS1GPRs;
  SmallVector<unsigned, 4> UnspilledCS2GPRs;
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();

  // Don't spill FP if the frame can be eliminated. This is determined
  // by scanning the callee-save registers to see if any is used.
  const unsigned *CSRegs = getCalleeSavedRegs();
  const TargetRegisterClass* const *CSRegClasses = getCalleeSavedRegClasses();
  for (unsigned i = 0; CSRegs[i]; ++i) {
    unsigned Reg = CSRegs[i];
    bool Spilled = false;
    if (MF.getRegInfo().isPhysRegUsed(Reg)) {
      AFI->setCSRegisterIsSpilled(Reg);
      Spilled = true;
      CanEliminateFrame = false;
    } else {
      // Check alias registers too.
      for (const unsigned *Aliases = getAliasSet(Reg); *Aliases; ++Aliases) {
        if (MF.getRegInfo().isPhysRegUsed(*Aliases)) {
          Spilled = true;
          CanEliminateFrame = false;
        }
      }
    }

    if (CSRegClasses[i] == &ARM::GPRRegClass) {
      if (Spilled) {
        NumGPRSpills++;

        if (!STI.isTargetDarwin()) {
          if (Reg == ARM::LR)
            LRSpilled = true;
          CS1Spilled = true;
          continue;
        }

        // Keep track if LR and any of R4, R5, R6, and R7 is spilled.
        switch (Reg) {
        case ARM::LR:
          LRSpilled = true;
          // Fallthrough
        case ARM::R4:
        case ARM::R5:
        case ARM::R6:
        case ARM::R7:
          CS1Spilled = true;
          break;
        default:
          break;
        }
      } else { 
        if (!STI.isTargetDarwin()) {
          UnspilledCS1GPRs.push_back(Reg);
          continue;
        }

        switch (Reg) {
        case ARM::R4:
        case ARM::R5:
        case ARM::R6:
        case ARM::R7:
        case ARM::LR:
          UnspilledCS1GPRs.push_back(Reg);
          break;
        default:
          UnspilledCS2GPRs.push_back(Reg);
          break;
        }
      }
    }
  }

  bool ForceLRSpill = false;
  if (!LRSpilled && AFI->isThumbFunction()) {
    unsigned FnSize = TII.GetFunctionSizeInBytes(MF);
    // Force LR to be spilled if the Thumb function size is > 2048. This enables
    // use of BL to implement far jump. If it turns out that it's not needed
    // then the branch fix up path will undo it.
    if (FnSize >= (1 << 11)) {
      CanEliminateFrame = false;
      ForceLRSpill = true;
    }
  }

  bool ExtraCSSpill = false;
  if (!CanEliminateFrame || hasFP(MF)) {
    AFI->setHasStackFrame(true);

    // If LR is not spilled, but at least one of R4, R5, R6, and R7 is spilled.
    // Spill LR as well so we can fold BX_RET to the registers restore (LDM).
    if (!LRSpilled && CS1Spilled) {
      MF.getRegInfo().setPhysRegUsed(ARM::LR);
      AFI->setCSRegisterIsSpilled(ARM::LR);
      NumGPRSpills++;
      UnspilledCS1GPRs.erase(std::find(UnspilledCS1GPRs.begin(),
                                    UnspilledCS1GPRs.end(), (unsigned)ARM::LR));
      ForceLRSpill = false;
      ExtraCSSpill = true;
    }

    // Darwin ABI requires FP to point to the stack slot that contains the
    // previous FP.
    if (STI.isTargetDarwin() || hasFP(MF)) {
      MF.getRegInfo().setPhysRegUsed(FramePtr);
      NumGPRSpills++;
    }

    // If stack and double are 8-byte aligned and we are spilling an odd number
    // of GPRs. Spill one extra callee save GPR so we won't have to pad between
    // the integer and double callee save areas.
    unsigned TargetAlign = MF.getTarget().getFrameInfo()->getStackAlignment();
    if (TargetAlign == 8 && (NumGPRSpills & 1)) {
      if (CS1Spilled && !UnspilledCS1GPRs.empty()) {
        for (unsigned i = 0, e = UnspilledCS1GPRs.size(); i != e; ++i) {
          unsigned Reg = UnspilledCS1GPRs[i];
          // Don't spiil high register if the function is thumb
          if (!AFI->isThumbFunction() || isLowRegister(Reg) || Reg == ARM::LR) {
            MF.getRegInfo().setPhysRegUsed(Reg);
            AFI->setCSRegisterIsSpilled(Reg);
            if (!isReservedReg(MF, Reg))
              ExtraCSSpill = true;
            break;
          }
        }
      } else if (!UnspilledCS2GPRs.empty() &&
                 !AFI->isThumbFunction()) {
        unsigned Reg = UnspilledCS2GPRs.front();
        MF.getRegInfo().setPhysRegUsed(Reg);
        AFI->setCSRegisterIsSpilled(Reg);
        if (!isReservedReg(MF, Reg))
          ExtraCSSpill = true;
      }
    }

    // Estimate if we might need to scavenge a register at some point in order
    // to materialize a stack offset. If so, either spill one additiona
    // callee-saved register or reserve a special spill slot to facilitate
    // register scavenging.
    if (RS && !ExtraCSSpill && !AFI->isThumbFunction()) {
      MachineFrameInfo  *MFI = MF.getFrameInfo();
      unsigned Size = estimateStackSize(MF, MFI);
      unsigned Limit = (1 << 12) - 1;
      for (MachineFunction::iterator BB = MF.begin(),E = MF.end();BB != E; ++BB)
        for (MachineBasicBlock::iterator I= BB->begin(); I != BB->end(); ++I) {
          for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
            if (I->getOperand(i).isFI()) {
              unsigned Opcode = I->getOpcode();
              const TargetInstrDesc &Desc = TII.get(Opcode);
              unsigned AddrMode = (Desc.TSFlags & ARMII::AddrModeMask);
              if (AddrMode == ARMII::AddrMode3) {
                Limit = (1 << 8) - 1;
                goto DoneEstimating;
              } else if (AddrMode == ARMII::AddrMode5) {
                unsigned ThisLimit = ((1 << 8) - 1) * 4;
                if (ThisLimit < Limit)
                  Limit = ThisLimit;
              }
            }
        }
    DoneEstimating:
      if (Size >= Limit) {
        // If any non-reserved CS register isn't spilled, just spill one or two
        // extra. That should take care of it!
        unsigned NumExtras = TargetAlign / 4;
        SmallVector<unsigned, 2> Extras;
        while (NumExtras && !UnspilledCS1GPRs.empty()) {
          unsigned Reg = UnspilledCS1GPRs.back();
          UnspilledCS1GPRs.pop_back();
          if (!isReservedReg(MF, Reg)) {
            Extras.push_back(Reg);
            NumExtras--;
          }
        }
        while (NumExtras && !UnspilledCS2GPRs.empty()) {
          unsigned Reg = UnspilledCS2GPRs.back();
          UnspilledCS2GPRs.pop_back();
          if (!isReservedReg(MF, Reg)) {
            Extras.push_back(Reg);
            NumExtras--;
          }
        }
        if (Extras.size() && NumExtras == 0) {
          for (unsigned i = 0, e = Extras.size(); i != e; ++i) {
            MF.getRegInfo().setPhysRegUsed(Extras[i]);
            AFI->setCSRegisterIsSpilled(Extras[i]);
          }
        } else {
          // Reserve a slot closest to SP or frame pointer.
          const TargetRegisterClass *RC = &ARM::GPRRegClass;
          RS->setScavengingFrameIndex(MFI->CreateStackObject(RC->getSize(),
                                                           RC->getAlignment()));
        }
      }
    }
  }

  if (ForceLRSpill) {
    MF.getRegInfo().setPhysRegUsed(ARM::LR);
    AFI->setCSRegisterIsSpilled(ARM::LR);
    AFI->setLRIsSpilledForFarJump(true);
  }
}

/// Move iterator pass the next bunch of callee save load / store ops for
/// the particular spill area (1: integer area 1, 2: integer area 2,
/// 3: fp area, 0: don't care).
static void movePastCSLoadStoreOps(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator &MBBI,
                                   int Opc, unsigned Area,
                                   const ARMSubtarget &STI) {
  while (MBBI != MBB.end() &&
         MBBI->getOpcode() == Opc && MBBI->getOperand(1).isFI()) {
    if (Area != 0) {
      bool Done = false;
      unsigned Category = 0;
      switch (MBBI->getOperand(0).getReg()) {
      case ARM::R4:  case ARM::R5:  case ARM::R6: case ARM::R7:
      case ARM::LR:
        Category = 1;
        break;
      case ARM::R8:  case ARM::R9:  case ARM::R10: case ARM::R11:
        Category = STI.isTargetDarwin() ? 2 : 1;
        break;
      case ARM::D8:  case ARM::D9:  case ARM::D10: case ARM::D11:
      case ARM::D12: case ARM::D13: case ARM::D14: case ARM::D15:
        Category = 3;
        break;
      default:
        Done = true;
        break;
      }
      if (Done || Category != Area)
        break;
    }

    ++MBBI;
  }
}

void ARMRegisterInfo::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front();
  MachineBasicBlock::iterator MBBI = MBB.begin();
  MachineFrameInfo  *MFI = MF.getFrameInfo();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  bool isThumb = AFI->isThumbFunction();
  unsigned VARegSaveSize = AFI->getVarArgsRegSaveSize();
  unsigned NumBytes = MFI->getStackSize();
  const std::vector<CalleeSavedInfo> &CSI = MFI->getCalleeSavedInfo();

  if (isThumb) {
    // Check if R3 is live in. It might have to be used as a scratch register.
    for (MachineRegisterInfo::livein_iterator I =MF.getRegInfo().livein_begin(),
         E = MF.getRegInfo().livein_end(); I != E; ++I) {
      if (I->first == ARM::R3) {
        AFI->setR3IsLiveIn(true);
        break;
      }
    }

    // Thumb add/sub sp, imm8 instructions implicitly multiply the offset by 4.
    NumBytes = (NumBytes + 3) & ~3;
    MFI->setStackSize(NumBytes);
  }

  // Determine the sizes of each callee-save spill areas and record which frame
  // belongs to which callee-save spill areas.
  unsigned GPRCS1Size = 0, GPRCS2Size = 0, DPRCSSize = 0;
  int FramePtrSpillFI = 0;

  if (VARegSaveSize)
    emitSPUpdate(MBB, MBBI, -VARegSaveSize, ARMCC::AL, 0, isThumb, TII, *this);

  if (!AFI->hasStackFrame()) {
    if (NumBytes != 0)
      emitSPUpdate(MBB, MBBI, -NumBytes, ARMCC::AL, 0, isThumb, TII, *this);
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
      if (STI.isTargetDarwin()) {
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

  if (!isThumb) {
    // Build the new SUBri to adjust SP for integer callee-save spill area 1.
    emitSPUpdate(MBB, MBBI, -GPRCS1Size, ARMCC::AL, 0, isThumb, TII, *this);
    movePastCSLoadStoreOps(MBB, MBBI, ARM::STR, 1, STI);
  } else if (MBBI != MBB.end() && MBBI->getOpcode() == ARM::tPUSH)
    ++MBBI;

  // Darwin ABI requires FP to point to the stack slot that contains the
  // previous FP.
  if (STI.isTargetDarwin() || hasFP(MF)) {
    MachineInstrBuilder MIB =
      BuildMI(MBB, MBBI, TII.get(isThumb ? ARM::tADDrSPi : ARM::ADDri),FramePtr)
      .addFrameIndex(FramePtrSpillFI).addImm(0);
    if (!isThumb) AddDefaultCC(AddDefaultPred(MIB));
  }

  if (!isThumb) {
    // Build the new SUBri to adjust SP for integer callee-save spill area 2.
    emitSPUpdate(MBB, MBBI, -GPRCS2Size, ARMCC::AL, 0, false, TII, *this);

    // Build the new SUBri to adjust SP for FP callee-save spill area.
    movePastCSLoadStoreOps(MBB, MBBI, ARM::STR, 2, STI);
    emitSPUpdate(MBB, MBBI, -DPRCSSize, ARMCC::AL, 0, false, TII, *this);
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
  if (NumBytes) {
    // Insert it after all the callee-save spills.
    if (!isThumb)
      movePastCSLoadStoreOps(MBB, MBBI, ARM::FSTD, 3, STI);
    emitSPUpdate(MBB, MBBI, -NumBytes, ARMCC::AL, 0, isThumb, TII, *this);
  }

  if(STI.isTargetELF() && hasFP(MF)) {
    MFI->setOffsetAdjustment(MFI->getOffsetAdjustment() -
                             AFI->getFramePtrSpillOffset());
  }

  AFI->setGPRCalleeSavedArea1Size(GPRCS1Size);
  AFI->setGPRCalleeSavedArea2Size(GPRCS2Size);
  AFI->setDPRCalleeSavedAreaSize(DPRCSSize);
}

static bool isCalleeSavedRegister(unsigned Reg, const unsigned *CSRegs) {
  for (unsigned i = 0; CSRegs[i]; ++i)
    if (Reg == CSRegs[i])
      return true;
  return false;
}

static bool isCSRestore(MachineInstr *MI, const unsigned *CSRegs) {
  return ((MI->getOpcode() == ARM::FLDD ||
           MI->getOpcode() == ARM::LDR  ||
           MI->getOpcode() == ARM::tRestore) &&
          MI->getOperand(1).isFI() &&
          isCalleeSavedRegister(MI->getOperand(0).getReg(), CSRegs));
}

void ARMRegisterInfo::emitEpilogue(MachineFunction &MF,
                                   MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = prior(MBB.end());
  assert((MBBI->getOpcode() == ARM::BX_RET ||
          MBBI->getOpcode() == ARM::tBX_RET ||
          MBBI->getOpcode() == ARM::tPOP_RET) &&
         "Can only insert epilog into returning blocks");

  MachineFrameInfo *MFI = MF.getFrameInfo();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  bool isThumb = AFI->isThumbFunction();
  unsigned VARegSaveSize = AFI->getVarArgsRegSaveSize();
  int NumBytes = (int)MFI->getStackSize();
  if (!AFI->hasStackFrame()) {
    if (NumBytes != 0)
      emitSPUpdate(MBB, MBBI, NumBytes, ARMCC::AL, 0, isThumb, TII, *this);
  } else {
    // Unwind MBBI to point to first LDR / FLDD.
    const unsigned *CSRegs = getCalleeSavedRegs();
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
    if (isThumb) {
      if (hasFP(MF)) {
        NumBytes = AFI->getFramePtrSpillOffset() - NumBytes;
        // Reset SP based on frame pointer only if the stack frame extends beyond
        // frame pointer stack slot or target is ELF and the function has FP.
        if (NumBytes)
          emitThumbRegPlusImmediate(MBB, MBBI, ARM::SP, FramePtr, -NumBytes,
                                    TII, *this);
        else
          BuildMI(MBB, MBBI, TII.get(ARM::tMOVr), ARM::SP).addReg(FramePtr);
      } else {
        if (MBBI->getOpcode() == ARM::tBX_RET &&
            &MBB.front() != MBBI &&
            prior(MBBI)->getOpcode() == ARM::tPOP) {
          MachineBasicBlock::iterator PMBBI = prior(MBBI);
          emitSPUpdate(MBB, PMBBI, NumBytes, ARMCC::AL, 0, isThumb, TII, *this);
        } else
          emitSPUpdate(MBB, MBBI, NumBytes, ARMCC::AL, 0, isThumb, TII, *this);
      }
    } else {
      // Darwin ABI requires FP to point to the stack slot that contains the
      // previous FP.
      if ((STI.isTargetDarwin() && NumBytes) || hasFP(MF)) {
        NumBytes = AFI->getFramePtrSpillOffset() - NumBytes;
        // Reset SP based on frame pointer only if the stack frame extends beyond
        // frame pointer stack slot or target is ELF and the function has FP.
        if (AFI->getGPRCalleeSavedArea2Size() ||
            AFI->getDPRCalleeSavedAreaSize()  ||
            AFI->getDPRCalleeSavedAreaOffset()||
            hasFP(MF)) {
          if (NumBytes)
            BuildMI(MBB, MBBI, TII.get(ARM::SUBri), ARM::SP).addReg(FramePtr)
              .addImm(NumBytes)
              .addImm((unsigned)ARMCC::AL).addReg(0).addReg(0);
          else
            BuildMI(MBB, MBBI, TII.get(ARM::MOVr), ARM::SP).addReg(FramePtr)
              .addImm((unsigned)ARMCC::AL).addReg(0).addReg(0);
        }
      } else if (NumBytes) {
        emitSPUpdate(MBB, MBBI, NumBytes, ARMCC::AL, 0, false, TII, *this);
      }

      // Move SP to start of integer callee save spill area 2.
      movePastCSLoadStoreOps(MBB, MBBI, ARM::FLDD, 3, STI);
      emitSPUpdate(MBB, MBBI, AFI->getDPRCalleeSavedAreaSize(), ARMCC::AL, 0,
                   false, TII, *this);

      // Move SP to start of integer callee save spill area 1.
      movePastCSLoadStoreOps(MBB, MBBI, ARM::LDR, 2, STI);
      emitSPUpdate(MBB, MBBI, AFI->getGPRCalleeSavedArea2Size(), ARMCC::AL, 0,
                   false, TII, *this);

      // Move SP to SP upon entry to the function.
      movePastCSLoadStoreOps(MBB, MBBI, ARM::LDR, 1, STI);
      emitSPUpdate(MBB, MBBI, AFI->getGPRCalleeSavedArea1Size(), ARMCC::AL, 0,
                   false, TII, *this);
    }
  }

  if (VARegSaveSize) {
    if (isThumb)
      // Epilogue for vararg functions: pop LR to R3 and branch off it.
      // FIXME: Verify this is still ok when R3 is no longer being reserved.
      BuildMI(MBB, MBBI, TII.get(ARM::tPOP)).addReg(ARM::R3);

    emitSPUpdate(MBB, MBBI, VARegSaveSize, ARMCC::AL, 0, isThumb, TII, *this);

    if (isThumb) {
      BuildMI(MBB, MBBI, TII.get(ARM::tBX_RET_vararg)).addReg(ARM::R3);
      MBB.erase(MBBI);
    }
  }
}

unsigned ARMRegisterInfo::getRARegister() const {
  return ARM::LR;
}

unsigned ARMRegisterInfo::getFrameRegister(MachineFunction &MF) const {
  if (STI.isTargetDarwin() || hasFP(MF))
    return (STI.useThumbBacktraces() || STI.isThumb()) ? ARM::R7 : ARM::R11;
  else
    return ARM::SP;
}

unsigned ARMRegisterInfo::getEHExceptionRegister() const {
  assert(0 && "What is the exception register");
  return 0;
}

unsigned ARMRegisterInfo::getEHHandlerRegister() const {
  assert(0 && "What is the exception handler register");
  return 0;
}

int ARMRegisterInfo::getDwarfRegNum(unsigned RegNum, bool isEH) const {
	return ARMGenRegisterInfo::getDwarfRegNumFull(RegNum, 0);
}

#include "ARMGenRegisterInfo.inc"
