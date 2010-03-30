//===- ARMBaseRegisterInfo.cpp - ARM Register Information -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the base ARM implementation of TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "ARMAddressingModes.h"
#include "ARMBaseInstrInfo.h"
#include "ARMBaseRegisterInfo.h"
#include "ARMInstrInfo.h"
#include "ARMMachineFunctionInfo.h"
#include "ARMSubtarget.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/LLVMContext.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

static cl::opt<bool>
ReuseFrameIndexVals("arm-reuse-frame-index-vals", cl::Hidden, cl::init(true),
          cl::desc("Reuse repeated frame index values"));

unsigned ARMBaseRegisterInfo::getRegisterNumbering(unsigned RegEnum,
                                                   bool *isSPVFP) {
  if (isSPVFP)
    *isSPVFP = false;

  using namespace ARM;
  switch (RegEnum) {
  default:
    llvm_unreachable("Unknown ARM register!");
  case R0:  case D0:  case Q0:  return 0;
  case R1:  case D1:  case Q1:  return 1;
  case R2:  case D2:  case Q2:  return 2;
  case R3:  case D3:  case Q3:  return 3;
  case R4:  case D4:  case Q4:  return 4;
  case R5:  case D5:  case Q5:  return 5;
  case R6:  case D6:  case Q6:  return 6;
  case R7:  case D7:  case Q7:  return 7;
  case R8:  case D8:  case Q8:  return 8;
  case R9:  case D9:  case Q9:  return 9;
  case R10: case D10: case Q10: return 10;
  case R11: case D11: case Q11: return 11;
  case R12: case D12: case Q12: return 12;
  case SP:  case D13: case Q13: return 13;
  case LR:  case D14: case Q14: return 14;
  case PC:  case D15: case Q15: return 15;

  case D16: return 16;
  case D17: return 17;
  case D18: return 18;
  case D19: return 19;
  case D20: return 20;
  case D21: return 21;
  case D22: return 22;
  case D23: return 23;
  case D24: return 24;
  case D25: return 25;
  case D26: return 26;
  case D27: return 27;
  case D28: return 28;
  case D29: return 29;
  case D30: return 30;
  case D31: return 31;

  case S0: case S1: case S2: case S3:
  case S4: case S5: case S6: case S7:
  case S8: case S9: case S10: case S11:
  case S12: case S13: case S14: case S15:
  case S16: case S17: case S18: case S19:
  case S20: case S21: case S22: case S23:
  case S24: case S25: case S26: case S27:
  case S28: case S29: case S30: case S31: {
    if (isSPVFP)
      *isSPVFP = true;
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

ARMBaseRegisterInfo::ARMBaseRegisterInfo(const ARMBaseInstrInfo &tii,
                                         const ARMSubtarget &sti)
  : ARMGenRegisterInfo(ARM::ADJCALLSTACKDOWN, ARM::ADJCALLSTACKUP),
    TII(tii), STI(sti),
    FramePtr((STI.isTargetDarwin() || STI.isThumb()) ? ARM::R7 : ARM::R11) {
}

const unsigned*
ARMBaseRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  static const unsigned CalleeSavedRegs[] = {
    ARM::LR, ARM::R11, ARM::R10, ARM::R9, ARM::R8,
    ARM::R7, ARM::R6,  ARM::R5,  ARM::R4,

    ARM::D15, ARM::D14, ARM::D13, ARM::D12,
    ARM::D11, ARM::D10, ARM::D9,  ARM::D8,
    0
  };

  static const unsigned DarwinCalleeSavedRegs[] = {
    // Darwin ABI deviates from ARM standard ABI. R9 is not a callee-saved
    // register.
    ARM::LR,  ARM::R7,  ARM::R6, ARM::R5, ARM::R4,
    ARM::R11, ARM::R10, ARM::R8,

    ARM::D15, ARM::D14, ARM::D13, ARM::D12,
    ARM::D11, ARM::D10, ARM::D9,  ARM::D8,
    0
  };
  return STI.isTargetDarwin() ? DarwinCalleeSavedRegs : CalleeSavedRegs;
}

const TargetRegisterClass* const *
ARMBaseRegisterInfo::getCalleeSavedRegClasses(const MachineFunction *MF) const {
  static const TargetRegisterClass * const CalleeSavedRegClasses[] = {
    &ARM::GPRRegClass, &ARM::GPRRegClass, &ARM::GPRRegClass,
    &ARM::GPRRegClass, &ARM::GPRRegClass, &ARM::GPRRegClass,
    &ARM::GPRRegClass, &ARM::GPRRegClass, &ARM::GPRRegClass,

    &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass,
    &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass,
    0
  };

  static const TargetRegisterClass * const ThumbCalleeSavedRegClasses[] = {
    &ARM::GPRRegClass, &ARM::GPRRegClass, &ARM::GPRRegClass,
    &ARM::GPRRegClass, &ARM::GPRRegClass, &ARM::tGPRRegClass,
    &ARM::tGPRRegClass,&ARM::tGPRRegClass,&ARM::tGPRRegClass,

    &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass,
    &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass,
    0
  };

  static const TargetRegisterClass * const DarwinCalleeSavedRegClasses[] = {
    &ARM::GPRRegClass, &ARM::GPRRegClass, &ARM::GPRRegClass,
    &ARM::GPRRegClass, &ARM::GPRRegClass, &ARM::GPRRegClass,
    &ARM::GPRRegClass, &ARM::GPRRegClass,

    &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass,
    &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass,
    0
  };

  static const TargetRegisterClass * const DarwinThumbCalleeSavedRegClasses[] ={
    &ARM::GPRRegClass,  &ARM::tGPRRegClass, &ARM::tGPRRegClass,
    &ARM::tGPRRegClass, &ARM::tGPRRegClass, &ARM::GPRRegClass,
    &ARM::GPRRegClass,  &ARM::GPRRegClass,

    &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass,
    &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass,
    0
  };

  if (STI.isThumb1Only()) {
    return STI.isTargetDarwin()
      ? DarwinThumbCalleeSavedRegClasses : ThumbCalleeSavedRegClasses;
  }
  return STI.isTargetDarwin()
    ? DarwinCalleeSavedRegClasses : CalleeSavedRegClasses;
}

BitVector ARMBaseRegisterInfo::
getReservedRegs(const MachineFunction &MF) const {
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

bool ARMBaseRegisterInfo::isReservedReg(const MachineFunction &MF,
                                        unsigned Reg) const {
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

const TargetRegisterClass *
ARMBaseRegisterInfo::getMatchingSuperRegClass(const TargetRegisterClass *A,
                                              const TargetRegisterClass *B,
                                              unsigned SubIdx) const {
  switch (SubIdx) {
  default: return 0;
  case 1:
  case 2:
  case 3:
  case 4:
    // S sub-registers.
    if (A->getSize() == 8) {
      if (B == &ARM::SPR_8RegClass)
        return &ARM::DPR_8RegClass;
      assert(B == &ARM::SPRRegClass && "Expecting SPR register class!");
      if (A == &ARM::DPR_8RegClass)
        return A;
      return &ARM::DPR_VFP2RegClass;
    }

    assert(A->getSize() == 16 && "Expecting a Q register class!");
    if (B == &ARM::SPR_8RegClass)
      return &ARM::QPR_8RegClass;
    return &ARM::QPR_VFP2RegClass;
  case 5:
  case 6:
    // D sub-registers.
    if (B == &ARM::DPR_VFP2RegClass)
      return &ARM::QPR_VFP2RegClass;
    if (B == &ARM::DPR_8RegClass)
      return &ARM::QPR_8RegClass;
    return A;
  }
  return 0;
}

const TargetRegisterClass *
ARMBaseRegisterInfo::getPointerRegClass(unsigned Kind) const {
  return ARM::GPRRegisterClass;
}

/// getAllocationOrder - Returns the register allocation order for a specified
/// register class in the form of a pair of TargetRegisterClass iterators.
std::pair<TargetRegisterClass::iterator,TargetRegisterClass::iterator>
ARMBaseRegisterInfo::getAllocationOrder(const TargetRegisterClass *RC,
                                        unsigned HintType, unsigned HintReg,
                                        const MachineFunction &MF) const {
  // Alternative register allocation orders when favoring even / odd registers
  // of register pairs.

  // No FP, R9 is available.
  static const unsigned GPREven1[] = {
    ARM::R0, ARM::R2, ARM::R4, ARM::R6, ARM::R8, ARM::R10,
    ARM::R1, ARM::R3, ARM::R12,ARM::LR, ARM::R5, ARM::R7,
    ARM::R9, ARM::R11
  };
  static const unsigned GPROdd1[] = {
    ARM::R1, ARM::R3, ARM::R5, ARM::R7, ARM::R9, ARM::R11,
    ARM::R0, ARM::R2, ARM::R12,ARM::LR, ARM::R4, ARM::R6,
    ARM::R8, ARM::R10
  };

  // FP is R7, R9 is available.
  static const unsigned GPREven2[] = {
    ARM::R0, ARM::R2, ARM::R4,          ARM::R8, ARM::R10,
    ARM::R1, ARM::R3, ARM::R12,ARM::LR, ARM::R5, ARM::R6,
    ARM::R9, ARM::R11
  };
  static const unsigned GPROdd2[] = {
    ARM::R1, ARM::R3, ARM::R5,          ARM::R9, ARM::R11,
    ARM::R0, ARM::R2, ARM::R12,ARM::LR, ARM::R4, ARM::R6,
    ARM::R8, ARM::R10
  };

  // FP is R11, R9 is available.
  static const unsigned GPREven3[] = {
    ARM::R0, ARM::R2, ARM::R4, ARM::R6, ARM::R8,
    ARM::R1, ARM::R3, ARM::R10,ARM::R12,ARM::LR, ARM::R5, ARM::R7,
    ARM::R9
  };
  static const unsigned GPROdd3[] = {
    ARM::R1, ARM::R3, ARM::R5, ARM::R6, ARM::R9,
    ARM::R0, ARM::R2, ARM::R10,ARM::R12,ARM::LR, ARM::R4, ARM::R7,
    ARM::R8
  };

  // No FP, R9 is not available.
  static const unsigned GPREven4[] = {
    ARM::R0, ARM::R2, ARM::R4, ARM::R6,          ARM::R10,
    ARM::R1, ARM::R3, ARM::R12,ARM::LR, ARM::R5, ARM::R7, ARM::R8,
    ARM::R11
  };
  static const unsigned GPROdd4[] = {
    ARM::R1, ARM::R3, ARM::R5, ARM::R7,          ARM::R11,
    ARM::R0, ARM::R2, ARM::R12,ARM::LR, ARM::R4, ARM::R6, ARM::R8,
    ARM::R10
  };

  // FP is R7, R9 is not available.
  static const unsigned GPREven5[] = {
    ARM::R0, ARM::R2, ARM::R4,                   ARM::R10,
    ARM::R1, ARM::R3, ARM::R12,ARM::LR, ARM::R5, ARM::R6, ARM::R8,
    ARM::R11
  };
  static const unsigned GPROdd5[] = {
    ARM::R1, ARM::R3, ARM::R5,                   ARM::R11,
    ARM::R0, ARM::R2, ARM::R12,ARM::LR, ARM::R4, ARM::R6, ARM::R8,
    ARM::R10
  };

  // FP is R11, R9 is not available.
  static const unsigned GPREven6[] = {
    ARM::R0, ARM::R2, ARM::R4, ARM::R6,
    ARM::R1, ARM::R3, ARM::R10,ARM::R12,ARM::LR, ARM::R5, ARM::R7, ARM::R8
  };
  static const unsigned GPROdd6[] = {
    ARM::R1, ARM::R3, ARM::R5, ARM::R7,
    ARM::R0, ARM::R2, ARM::R10,ARM::R12,ARM::LR, ARM::R4, ARM::R6, ARM::R8
  };


  if (HintType == ARMRI::RegPairEven) {
    if (isPhysicalRegister(HintReg) && getRegisterPairEven(HintReg, MF) == 0)
      // It's no longer possible to fulfill this hint. Return the default
      // allocation order.
      return std::make_pair(RC->allocation_order_begin(MF),
                            RC->allocation_order_end(MF));

    if (!STI.isTargetDarwin() && !hasFP(MF)) {
      if (!STI.isR9Reserved())
        return std::make_pair(GPREven1,
                              GPREven1 + (sizeof(GPREven1)/sizeof(unsigned)));
      else
        return std::make_pair(GPREven4,
                              GPREven4 + (sizeof(GPREven4)/sizeof(unsigned)));
    } else if (FramePtr == ARM::R7) {
      if (!STI.isR9Reserved())
        return std::make_pair(GPREven2,
                              GPREven2 + (sizeof(GPREven2)/sizeof(unsigned)));
      else
        return std::make_pair(GPREven5,
                              GPREven5 + (sizeof(GPREven5)/sizeof(unsigned)));
    } else { // FramePtr == ARM::R11
      if (!STI.isR9Reserved())
        return std::make_pair(GPREven3,
                              GPREven3 + (sizeof(GPREven3)/sizeof(unsigned)));
      else
        return std::make_pair(GPREven6,
                              GPREven6 + (sizeof(GPREven6)/sizeof(unsigned)));
    }
  } else if (HintType == ARMRI::RegPairOdd) {
    if (isPhysicalRegister(HintReg) && getRegisterPairOdd(HintReg, MF) == 0)
      // It's no longer possible to fulfill this hint. Return the default
      // allocation order.
      return std::make_pair(RC->allocation_order_begin(MF),
                            RC->allocation_order_end(MF));

    if (!STI.isTargetDarwin() && !hasFP(MF)) {
      if (!STI.isR9Reserved())
        return std::make_pair(GPROdd1,
                              GPROdd1 + (sizeof(GPROdd1)/sizeof(unsigned)));
      else
        return std::make_pair(GPROdd4,
                              GPROdd4 + (sizeof(GPROdd4)/sizeof(unsigned)));
    } else if (FramePtr == ARM::R7) {
      if (!STI.isR9Reserved())
        return std::make_pair(GPROdd2,
                              GPROdd2 + (sizeof(GPROdd2)/sizeof(unsigned)));
      else
        return std::make_pair(GPROdd5,
                              GPROdd5 + (sizeof(GPROdd5)/sizeof(unsigned)));
    } else { // FramePtr == ARM::R11
      if (!STI.isR9Reserved())
        return std::make_pair(GPROdd3,
                              GPROdd3 + (sizeof(GPROdd3)/sizeof(unsigned)));
      else
        return std::make_pair(GPROdd6,
                              GPROdd6 + (sizeof(GPROdd6)/sizeof(unsigned)));
    }
  }
  return std::make_pair(RC->allocation_order_begin(MF),
                        RC->allocation_order_end(MF));
}

/// ResolveRegAllocHint - Resolves the specified register allocation hint
/// to a physical register. Returns the physical register if it is successful.
unsigned
ARMBaseRegisterInfo::ResolveRegAllocHint(unsigned Type, unsigned Reg,
                                         const MachineFunction &MF) const {
  if (Reg == 0 || !isPhysicalRegister(Reg))
    return 0;
  if (Type == 0)
    return Reg;
  else if (Type == (unsigned)ARMRI::RegPairOdd)
    // Odd register.
    return getRegisterPairOdd(Reg, MF);
  else if (Type == (unsigned)ARMRI::RegPairEven)
    // Even register.
    return getRegisterPairEven(Reg, MF);
  return 0;
}

void
ARMBaseRegisterInfo::UpdateRegAllocHint(unsigned Reg, unsigned NewReg,
                                        MachineFunction &MF) const {
  MachineRegisterInfo *MRI = &MF.getRegInfo();
  std::pair<unsigned, unsigned> Hint = MRI->getRegAllocationHint(Reg);
  if ((Hint.first == (unsigned)ARMRI::RegPairOdd ||
       Hint.first == (unsigned)ARMRI::RegPairEven) &&
      Hint.second && TargetRegisterInfo::isVirtualRegister(Hint.second)) {
    // If 'Reg' is one of the even / odd register pair and it's now changed
    // (e.g. coalesced) into a different register. The other register of the
    // pair allocation hint must be updated to reflect the relationship
    // change.
    unsigned OtherReg = Hint.second;
    Hint = MRI->getRegAllocationHint(OtherReg);
    if (Hint.second == Reg)
      // Make sure the pair has not already divorced.
      MRI->setRegAllocationHint(OtherReg, Hint.first, NewReg);
  }
}

/// hasFP - Return true if the specified function should have a dedicated frame
/// pointer register.  This is true if the function has variable sized allocas
/// or if frame pointer elimination is disabled.
///
bool ARMBaseRegisterInfo::hasFP(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  return ((NoFramePointerElim && MFI->hasCalls())||
          needsStackRealignment(MF) ||
          MFI->hasVarSizedObjects() ||
          MFI->isFrameAddressTaken());
}

bool ARMBaseRegisterInfo::canRealignStack(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  const ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  return (RealignStack &&
          !AFI->isThumb1OnlyFunction() &&
          !MFI->hasVarSizedObjects());
}

bool ARMBaseRegisterInfo::
needsStackRealignment(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  const ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  unsigned StackAlign = MF.getTarget().getFrameInfo()->getStackAlignment();
  return (RealignStack &&
          !AFI->isThumb1OnlyFunction() &&
          (MFI->getMaxAlignment() > StackAlign) &&
          !MFI->hasVarSizedObjects());
}

bool ARMBaseRegisterInfo::
cannotEliminateFrame(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  if (NoFramePointerElim && MFI->hasCalls())
    return true;
  return MFI->hasVarSizedObjects() || MFI->isFrameAddressTaken()
    || needsStackRealignment(MF);
}

/// estimateStackSize - Estimate and return the size of the frame.
static unsigned estimateStackSize(MachineFunction &MF) {
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

/// estimateRSStackSizeLimit - Look at each instruction that references stack
/// frames and return the stack size limit beyond which some of these
/// instructions will require a scratch register during their expansion later.
unsigned
ARMBaseRegisterInfo::estimateRSStackSizeLimit(MachineFunction &MF) const {
  unsigned Limit = (1 << 12) - 1;
  for (MachineFunction::iterator BB = MF.begin(),E = MF.end(); BB != E; ++BB) {
    for (MachineBasicBlock::iterator I = BB->begin(), E = BB->end();
         I != E; ++I) {
      for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
        if (!I->getOperand(i).isFI()) continue;

        const TargetInstrDesc &Desc = TII.get(I->getOpcode());
        unsigned AddrMode = (Desc.TSFlags & ARMII::AddrModeMask);
        if (AddrMode == ARMII::AddrMode3 ||
            AddrMode == ARMII::AddrModeT2_i8)
          return (1 << 8) - 1;

        if (AddrMode == ARMII::AddrMode5 ||
            AddrMode == ARMII::AddrModeT2_i8s4)
          Limit = std::min(Limit, ((1U << 8) - 1) * 4);

        if (AddrMode == ARMII::AddrModeT2_i12 && hasFP(MF))
          // When the stack offset is negative, we will end up using
          // the i8 instructions instead.
          return (1 << 8) - 1;

        if (AddrMode == ARMII::AddrMode6)
          return 0;
        break; // At most one FI per instruction
      }
    }
  }

  return Limit;
}

void
ARMBaseRegisterInfo::processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
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

  // Spill R4 if Thumb2 function requires stack realignment - it will be used as
  // scratch register.
  // FIXME: It will be better just to find spare register here.
  if (needsStackRealignment(MF) &&
      AFI->isThumb2Function())
    MF.getRegInfo().setPhysRegUsed(ARM::R4);

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

    if (CSRegClasses[i] == ARM::GPRRegisterClass ||
        CSRegClasses[i] == ARM::tGPRRegisterClass) {
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
  if (!LRSpilled && AFI->isThumb1OnlyFunction()) {
    unsigned FnSize = TII.GetFunctionSizeInBytes(MF);
    // Force LR to be spilled if the Thumb function size is > 2048. This enables
    // use of BL to implement far jump. If it turns out that it's not needed
    // then the branch fix up path will undo it.
    if (FnSize >= (1 << 11)) {
      CanEliminateFrame = false;
      ForceLRSpill = true;
    }
  }

  // If any of the stack slot references may be out of range of an immediate
  // offset, make sure a register (or a spill slot) is available for the
  // register scavenger. Note that if we're indexing off the frame pointer, the
  // effective stack size is 4 bytes larger since the FP points to the stack
  // slot of the previous FP.
  bool BigStack = RS &&
    estimateStackSize(MF) + (hasFP(MF) ? 4 : 0) >= estimateRSStackSizeLimit(MF);

  bool ExtraCSSpill = false;
  if (BigStack || !CanEliminateFrame || cannotEliminateFrame(MF)) {
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
          // Don't spill high register if the function is thumb1
          if (!AFI->isThumb1OnlyFunction() ||
              isARMLowRegister(Reg) || Reg == ARM::LR) {
            MF.getRegInfo().setPhysRegUsed(Reg);
            AFI->setCSRegisterIsSpilled(Reg);
            if (!isReservedReg(MF, Reg))
              ExtraCSSpill = true;
            break;
          }
        }
      } else if (!UnspilledCS2GPRs.empty() &&
                 !AFI->isThumb1OnlyFunction()) {
        unsigned Reg = UnspilledCS2GPRs.front();
        MF.getRegInfo().setPhysRegUsed(Reg);
        AFI->setCSRegisterIsSpilled(Reg);
        if (!isReservedReg(MF, Reg))
          ExtraCSSpill = true;
      }
    }

    // Estimate if we might need to scavenge a register at some point in order
    // to materialize a stack offset. If so, either spill one additional
    // callee-saved register or reserve a special spill slot to facilitate
    // register scavenging. Thumb1 needs a spill slot for stack pointer
    // adjustments also, even when the frame itself is small.
    if (BigStack && !ExtraCSSpill) {
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
      // For non-Thumb1 functions, also check for hi-reg CS registers
      if (!AFI->isThumb1OnlyFunction()) {
        while (NumExtras && !UnspilledCS2GPRs.empty()) {
          unsigned Reg = UnspilledCS2GPRs.back();
          UnspilledCS2GPRs.pop_back();
          if (!isReservedReg(MF, Reg)) {
            Extras.push_back(Reg);
            NumExtras--;
          }
        }
      }
      if (Extras.size() && NumExtras == 0) {
        for (unsigned i = 0, e = Extras.size(); i != e; ++i) {
          MF.getRegInfo().setPhysRegUsed(Extras[i]);
          AFI->setCSRegisterIsSpilled(Extras[i]);
        }
      } else if (!AFI->isThumb1OnlyFunction()) {
        // note: Thumb1 functions spill to R12, not the stack.  Reserve a slot
        // closest to SP or frame pointer.
        const TargetRegisterClass *RC = ARM::GPRRegisterClass;
        MachineFrameInfo *MFI = MF.getFrameInfo();
        RS->setScavengingFrameIndex(MFI->CreateStackObject(RC->getSize(),
                                                           RC->getAlignment(),
                                                           false));
      }
    }
  }

  if (ForceLRSpill) {
    MF.getRegInfo().setPhysRegUsed(ARM::LR);
    AFI->setCSRegisterIsSpilled(ARM::LR);
    AFI->setLRIsSpilledForFarJump(true);
  }
}

unsigned ARMBaseRegisterInfo::getRARegister() const {
  return ARM::LR;
}

unsigned 
ARMBaseRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  if (STI.isTargetDarwin() || hasFP(MF))
    return FramePtr;
  return ARM::SP;
}

int
ARMBaseRegisterInfo::getFrameIndexReference(const MachineFunction &MF, int FI,
                                            unsigned &FrameReg) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  const ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  int Offset = MFI->getObjectOffset(FI) + MFI->getStackSize();
  bool isFixed = MFI->isFixedObjectIndex(FI);

  FrameReg = ARM::SP;
  if (AFI->isGPRCalleeSavedArea1Frame(FI))
    Offset -= AFI->getGPRCalleeSavedArea1Offset();
  else if (AFI->isGPRCalleeSavedArea2Frame(FI))
    Offset -= AFI->getGPRCalleeSavedArea2Offset();
  else if (AFI->isDPRCalleeSavedAreaFrame(FI))
    Offset -= AFI->getDPRCalleeSavedAreaOffset();
  else if (needsStackRealignment(MF)) {
    // When dynamically realigning the stack, use the frame pointer for
    // parameters, and the stack pointer for locals.
    assert (hasFP(MF) && "dynamic stack realignment without a FP!");
    if (isFixed) {
      FrameReg = getFrameRegister(MF);
      Offset -= AFI->getFramePtrSpillOffset();
    }
  } else if (hasFP(MF) && AFI->hasStackFrame()) {
    if (isFixed || MFI->hasVarSizedObjects()) {
      // Use frame pointer to reference fixed objects unless this is a
      // frameless function.
      FrameReg = getFrameRegister(MF);
      Offset -= AFI->getFramePtrSpillOffset();
    } else if (AFI->isThumb2Function()) {
      // In Thumb2 mode, the negative offset is very limited.
      int FPOffset = Offset - AFI->getFramePtrSpillOffset();
      if (FPOffset >= -255 && FPOffset < 0) {
        FrameReg = getFrameRegister(MF);
        Offset = FPOffset;
      }
    }
  }
  return Offset;
}


int
ARMBaseRegisterInfo::getFrameIndexOffset(const MachineFunction &MF,
                                         int FI) const {
  unsigned FrameReg;
  return getFrameIndexReference(MF, FI, FrameReg);
}

unsigned ARMBaseRegisterInfo::getEHExceptionRegister() const {
  llvm_unreachable("What is the exception register");
  return 0;
}

unsigned ARMBaseRegisterInfo::getEHHandlerRegister() const {
  llvm_unreachable("What is the exception handler register");
  return 0;
}

int ARMBaseRegisterInfo::getDwarfRegNum(unsigned RegNum, bool isEH) const {
  return ARMGenRegisterInfo::getDwarfRegNumFull(RegNum, 0);
}

unsigned ARMBaseRegisterInfo::getRegisterPairEven(unsigned Reg,
                                              const MachineFunction &MF) const {
  switch (Reg) {
  default: break;
  // Return 0 if either register of the pair is a special register.
  // So no R12, etc.
  case ARM::R1:
    return ARM::R0;
  case ARM::R3:
    return ARM::R2;
  case ARM::R5:
    return ARM::R4;
  case ARM::R7:
    return isReservedReg(MF, ARM::R7)  ? 0 : ARM::R6;
  case ARM::R9:
    return isReservedReg(MF, ARM::R9)  ? 0 :ARM::R8;
  case ARM::R11:
    return isReservedReg(MF, ARM::R11) ? 0 : ARM::R10;

  case ARM::S1:
    return ARM::S0;
  case ARM::S3:
    return ARM::S2;
  case ARM::S5:
    return ARM::S4;
  case ARM::S7:
    return ARM::S6;
  case ARM::S9:
    return ARM::S8;
  case ARM::S11:
    return ARM::S10;
  case ARM::S13:
    return ARM::S12;
  case ARM::S15:
    return ARM::S14;
  case ARM::S17:
    return ARM::S16;
  case ARM::S19:
    return ARM::S18;
  case ARM::S21:
    return ARM::S20;
  case ARM::S23:
    return ARM::S22;
  case ARM::S25:
    return ARM::S24;
  case ARM::S27:
    return ARM::S26;
  case ARM::S29:
    return ARM::S28;
  case ARM::S31:
    return ARM::S30;

  case ARM::D1:
    return ARM::D0;
  case ARM::D3:
    return ARM::D2;
  case ARM::D5:
    return ARM::D4;
  case ARM::D7:
    return ARM::D6;
  case ARM::D9:
    return ARM::D8;
  case ARM::D11:
    return ARM::D10;
  case ARM::D13:
    return ARM::D12;
  case ARM::D15:
    return ARM::D14;
  case ARM::D17:
    return ARM::D16;
  case ARM::D19:
    return ARM::D18;
  case ARM::D21:
    return ARM::D20;
  case ARM::D23:
    return ARM::D22;
  case ARM::D25:
    return ARM::D24;
  case ARM::D27:
    return ARM::D26;
  case ARM::D29:
    return ARM::D28;
  case ARM::D31:
    return ARM::D30;
  }

  return 0;
}

unsigned ARMBaseRegisterInfo::getRegisterPairOdd(unsigned Reg,
                                             const MachineFunction &MF) const {
  switch (Reg) {
  default: break;
  // Return 0 if either register of the pair is a special register.
  // So no R12, etc.
  case ARM::R0:
    return ARM::R1;
  case ARM::R2:
    return ARM::R3;
  case ARM::R4:
    return ARM::R5;
  case ARM::R6:
    return isReservedReg(MF, ARM::R7)  ? 0 : ARM::R7;
  case ARM::R8:
    return isReservedReg(MF, ARM::R9)  ? 0 :ARM::R9;
  case ARM::R10:
    return isReservedReg(MF, ARM::R11) ? 0 : ARM::R11;

  case ARM::S0:
    return ARM::S1;
  case ARM::S2:
    return ARM::S3;
  case ARM::S4:
    return ARM::S5;
  case ARM::S6:
    return ARM::S7;
  case ARM::S8:
    return ARM::S9;
  case ARM::S10:
    return ARM::S11;
  case ARM::S12:
    return ARM::S13;
  case ARM::S14:
    return ARM::S15;
  case ARM::S16:
    return ARM::S17;
  case ARM::S18:
    return ARM::S19;
  case ARM::S20:
    return ARM::S21;
  case ARM::S22:
    return ARM::S23;
  case ARM::S24:
    return ARM::S25;
  case ARM::S26:
    return ARM::S27;
  case ARM::S28:
    return ARM::S29;
  case ARM::S30:
    return ARM::S31;

  case ARM::D0:
    return ARM::D1;
  case ARM::D2:
    return ARM::D3;
  case ARM::D4:
    return ARM::D5;
  case ARM::D6:
    return ARM::D7;
  case ARM::D8:
    return ARM::D9;
  case ARM::D10:
    return ARM::D11;
  case ARM::D12:
    return ARM::D13;
  case ARM::D14:
    return ARM::D15;
  case ARM::D16:
    return ARM::D17;
  case ARM::D18:
    return ARM::D19;
  case ARM::D20:
    return ARM::D21;
  case ARM::D22:
    return ARM::D23;
  case ARM::D24:
    return ARM::D25;
  case ARM::D26:
    return ARM::D27;
  case ARM::D28:
    return ARM::D29;
  case ARM::D30:
    return ARM::D31;
  }

  return 0;
}

/// emitLoadConstPool - Emits a load from constpool to materialize the
/// specified immediate.
void ARMBaseRegisterInfo::
emitLoadConstPool(MachineBasicBlock &MBB,
                  MachineBasicBlock::iterator &MBBI,
                  DebugLoc dl,
                  unsigned DestReg, unsigned SubIdx, int Val,
                  ARMCC::CondCodes Pred,
                  unsigned PredReg) const {
  MachineFunction &MF = *MBB.getParent();
  MachineConstantPool *ConstantPool = MF.getConstantPool();
  Constant *C =
        ConstantInt::get(Type::getInt32Ty(MF.getFunction()->getContext()), Val);
  unsigned Idx = ConstantPool->getConstantPoolIndex(C, 4);

  BuildMI(MBB, MBBI, dl, TII.get(ARM::LDRcp))
    .addReg(DestReg, getDefRegState(true), SubIdx)
    .addConstantPoolIndex(Idx)
    .addReg(0).addImm(0).addImm(Pred).addReg(PredReg);
}

bool ARMBaseRegisterInfo::
requiresRegisterScavenging(const MachineFunction &MF) const {
  return true;
}

bool ARMBaseRegisterInfo::
requiresFrameIndexScavenging(const MachineFunction &MF) const {
  return true;
}

// hasReservedCallFrame - Under normal circumstances, when a frame pointer is
// not required, we reserve argument space for call sites in the function
// immediately on entry to the current function. This eliminates the need for
// add/sub sp brackets around call sites. Returns true if the call frame is
// included as part of the stack frame.
bool ARMBaseRegisterInfo::
hasReservedCallFrame(MachineFunction &MF) const {
  const MachineFrameInfo *FFI = MF.getFrameInfo();
  unsigned CFSize = FFI->getMaxCallFrameSize();
  // It's not always a good idea to include the call frame as part of the
  // stack frame. ARM (especially Thumb) has small immediate offset to
  // address the stack frame. So a large call frame can cause poor codegen
  // and may even makes it impossible to scavenge a register.
  if (CFSize >= ((1 << 12) - 1) / 2)  // Half of imm12
    return false;

  return !MF.getFrameInfo()->hasVarSizedObjects();
}

// canSimplifyCallFramePseudos - If there is a reserved call frame, the
// call frame pseudos can be simplified. Unlike most targets, having a FP
// is not sufficient here since we still may reference some objects via SP
// even when FP is available in Thumb2 mode.
bool ARMBaseRegisterInfo::
canSimplifyCallFramePseudos(MachineFunction &MF) const {
  return hasReservedCallFrame(MF) || MF.getFrameInfo()->hasVarSizedObjects();
}

static void
emitSPUpdate(bool isARM,
             MachineBasicBlock &MBB, MachineBasicBlock::iterator &MBBI,
             DebugLoc dl, const ARMBaseInstrInfo &TII,
             int NumBytes,
             ARMCC::CondCodes Pred = ARMCC::AL, unsigned PredReg = 0) {
  if (isARM)
    emitARMRegPlusImmediate(MBB, MBBI, dl, ARM::SP, ARM::SP, NumBytes,
                            Pred, PredReg, TII);
  else
    emitT2RegPlusImmediate(MBB, MBBI, dl, ARM::SP, ARM::SP, NumBytes,
                           Pred, PredReg, TII);
}


void ARMBaseRegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  if (!hasReservedCallFrame(MF)) {
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
      unsigned Align = MF.getTarget().getFrameInfo()->getStackAlignment();
      Amount = (Amount+Align-1)/Align*Align;

      ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
      assert(!AFI->isThumb1OnlyFunction() &&
             "This eliminateCallFramePseudoInstr does not support Thumb1!");
      bool isARM = !AFI->isThumbFunction();

      // Replace the pseudo instruction with a new instruction...
      unsigned Opc = Old->getOpcode();
      int PIdx = Old->findFirstPredOperandIdx();
      ARMCC::CondCodes Pred = (PIdx == -1)
        ? ARMCC::AL : (ARMCC::CondCodes)Old->getOperand(PIdx).getImm();
      if (Opc == ARM::ADJCALLSTACKDOWN || Opc == ARM::tADJCALLSTACKDOWN) {
        // Note: PredReg is operand 2 for ADJCALLSTACKDOWN.
        unsigned PredReg = Old->getOperand(2).getReg();
        emitSPUpdate(isARM, MBB, I, dl, TII, -Amount, Pred, PredReg);
      } else {
        // Note: PredReg is operand 3 for ADJCALLSTACKUP.
        unsigned PredReg = Old->getOperand(3).getReg();
        assert(Opc == ARM::ADJCALLSTACKUP || Opc == ARM::tADJCALLSTACKUP);
        emitSPUpdate(isARM, MBB, I, dl, TII, Amount, Pred, PredReg);
      }
    }
  }
  MBB.erase(I);
}

unsigned
ARMBaseRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                         int SPAdj, int *Value,
                                         RegScavenger *RS) const {
  unsigned i = 0;
  MachineInstr &MI = *II;
  MachineBasicBlock &MBB = *MI.getParent();
  MachineFunction &MF = *MBB.getParent();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  assert(!AFI->isThumb1OnlyFunction() &&
         "This eliminateFrameIndex does not support Thumb1!");

  while (!MI.getOperand(i).isFI()) {
    ++i;
    assert(i < MI.getNumOperands() && "Instr doesn't have FrameIndex operand!");
  }

  int FrameIndex = MI.getOperand(i).getIndex();
  unsigned FrameReg;

  int Offset = getFrameIndexReference(MF, FrameIndex, FrameReg);
  if (FrameReg != ARM::SP)
    SPAdj = 0;
  Offset += SPAdj;

  // Modify MI as necessary to handle as much of 'Offset' as possible
  bool Done = false;
  if (!AFI->isThumbFunction())
    Done = rewriteARMFrameIndex(MI, i, FrameReg, Offset, TII);
  else {
    assert(AFI->isThumb2Function());
    Done = rewriteT2FrameIndex(MI, i, FrameReg, Offset, TII);
  }
  if (Done)
    return 0;

  // If we get here, the immediate doesn't fit into the instruction.  We folded
  // as much as possible above, handle the rest, providing a register that is
  // SP+LargeImm.
  assert((Offset ||
          (MI.getDesc().TSFlags & ARMII::AddrModeMask) == ARMII::AddrMode4 ||
          (MI.getDesc().TSFlags & ARMII::AddrModeMask) == ARMII::AddrMode6) &&
         "This code isn't needed if offset already handled!");

  unsigned ScratchReg = 0;
  int PIdx = MI.findFirstPredOperandIdx();
  ARMCC::CondCodes Pred = (PIdx == -1)
    ? ARMCC::AL : (ARMCC::CondCodes)MI.getOperand(PIdx).getImm();
  unsigned PredReg = (PIdx == -1) ? 0 : MI.getOperand(PIdx+1).getReg();
  if (Offset == 0)
    // Must be addrmode4/6.
    MI.getOperand(i).ChangeToRegister(FrameReg, false, false, false);
  else {
    ScratchReg = MF.getRegInfo().createVirtualRegister(ARM::GPRRegisterClass);
    if (Value) *Value = Offset;
    if (!AFI->isThumbFunction())
      emitARMRegPlusImmediate(MBB, II, MI.getDebugLoc(), ScratchReg, FrameReg,
                              Offset, Pred, PredReg, TII);
    else {
      assert(AFI->isThumb2Function());
      emitT2RegPlusImmediate(MBB, II, MI.getDebugLoc(), ScratchReg, FrameReg,
                             Offset, Pred, PredReg, TII);
    }
    MI.getOperand(i).ChangeToRegister(ScratchReg, false, false, true);
    if (!ReuseFrameIndexVals)
      ScratchReg = 0;
  }
  return ScratchReg;
}

/// Move iterator past the next bunch of callee save load / store ops for
/// the particular spill area (1: integer area 1, 2: integer area 2,
/// 3: fp area, 0: don't care).
static void movePastCSLoadStoreOps(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator &MBBI,
                                   int Opc1, int Opc2, unsigned Area,
                                   const ARMSubtarget &STI) {
  while (MBBI != MBB.end() &&
         ((MBBI->getOpcode() == Opc1) || (MBBI->getOpcode() == Opc2)) &&
         MBBI->getOperand(1).isFI()) {
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

void ARMBaseRegisterInfo::
emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front();
  MachineBasicBlock::iterator MBBI = MBB.begin();
  MachineFrameInfo  *MFI = MF.getFrameInfo();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  assert(!AFI->isThumb1OnlyFunction() &&
         "This emitPrologue does not support Thumb1!");
  bool isARM = !AFI->isThumbFunction();
  unsigned VARegSaveSize = AFI->getVarArgsRegSaveSize();
  unsigned NumBytes = MFI->getStackSize();
  const std::vector<CalleeSavedInfo> &CSI = MFI->getCalleeSavedInfo();
  DebugLoc dl = (MBBI != MBB.end() ?
                 MBBI->getDebugLoc() : DebugLoc::getUnknownLoc());

  // Determine the sizes of each callee-save spill areas and record which frame
  // belongs to which callee-save spill areas.
  unsigned GPRCS1Size = 0, GPRCS2Size = 0, DPRCSSize = 0;
  int FramePtrSpillFI = 0;

  // Allocate the vararg register save area. This is not counted in NumBytes.
  if (VARegSaveSize)
    emitSPUpdate(isARM, MBB, MBBI, dl, TII, -VARegSaveSize);

  if (!AFI->hasStackFrame()) {
    if (NumBytes != 0)
      emitSPUpdate(isARM, MBB, MBBI, dl, TII, -NumBytes);
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

  // Build the new SUBri to adjust SP for integer callee-save spill area 1.
  emitSPUpdate(isARM, MBB, MBBI, dl, TII, -GPRCS1Size);
  movePastCSLoadStoreOps(MBB, MBBI, ARM::STR, ARM::t2STRi12, 1, STI);

  // Set FP to point to the stack slot that contains the previous FP.
  // For Darwin, FP is R7, which has now been stored in spill area 1.
  // Otherwise, if this is not Darwin, all the callee-saved registers go
  // into spill area 1, including the FP in R11.  In either case, it is
  // now safe to emit this assignment.
  if (STI.isTargetDarwin() || hasFP(MF)) {
    unsigned ADDriOpc = !AFI->isThumbFunction() ? ARM::ADDri : ARM::t2ADDri;
    MachineInstrBuilder MIB =
      BuildMI(MBB, MBBI, dl, TII.get(ADDriOpc), FramePtr)
      .addFrameIndex(FramePtrSpillFI).addImm(0);
    AddDefaultCC(AddDefaultPred(MIB));
  }

  // Build the new SUBri to adjust SP for integer callee-save spill area 2.
  emitSPUpdate(isARM, MBB, MBBI, dl, TII, -GPRCS2Size);

  // Build the new SUBri to adjust SP for FP callee-save spill area.
  movePastCSLoadStoreOps(MBB, MBBI, ARM::STR, ARM::t2STRi12, 2, STI);
  emitSPUpdate(isARM, MBB, MBBI, dl, TII, -DPRCSSize);

  // Determine starting offsets of spill areas.
  unsigned DPRCSOffset  = NumBytes - (GPRCS1Size + GPRCS2Size + DPRCSSize);
  unsigned GPRCS2Offset = DPRCSOffset + DPRCSSize;
  unsigned GPRCS1Offset = GPRCS2Offset + GPRCS2Size;
  if (STI.isTargetDarwin() || hasFP(MF))
    AFI->setFramePtrSpillOffset(MFI->getObjectOffset(FramePtrSpillFI) +
                                NumBytes);
  AFI->setGPRCalleeSavedArea1Offset(GPRCS1Offset);
  AFI->setGPRCalleeSavedArea2Offset(GPRCS2Offset);
  AFI->setDPRCalleeSavedAreaOffset(DPRCSOffset);

  movePastCSLoadStoreOps(MBB, MBBI, ARM::VSTRD, 0, 3, STI);
  NumBytes = DPRCSOffset;
  if (NumBytes) {
    // Adjust SP after all the callee-save spills.
    emitSPUpdate(isARM, MBB, MBBI, dl, TII, -NumBytes);
  }

  if (STI.isTargetELF() && hasFP(MF)) {
    MFI->setOffsetAdjustment(MFI->getOffsetAdjustment() -
                             AFI->getFramePtrSpillOffset());
  }

  AFI->setGPRCalleeSavedArea1Size(GPRCS1Size);
  AFI->setGPRCalleeSavedArea2Size(GPRCS2Size);
  AFI->setDPRCalleeSavedAreaSize(DPRCSSize);

  // If we need dynamic stack realignment, do it here.
  if (needsStackRealignment(MF)) {
    unsigned MaxAlign = MFI->getMaxAlignment();
    assert (!AFI->isThumb1OnlyFunction());
    if (!AFI->isThumbFunction()) {
      // Emit bic sp, sp, MaxAlign
      AddDefaultCC(AddDefaultPred(BuildMI(MBB, MBBI, dl,
                                          TII.get(ARM::BICri), ARM::SP)
                                  .addReg(ARM::SP, RegState::Kill)
                                  .addImm(MaxAlign-1)));
    } else {
      // We cannot use sp as source/dest register here, thus we're emitting the
      // following sequence:
      // mov r4, sp
      // bic r4, r4, MaxAlign
      // mov sp, r4
      // FIXME: It will be better just to find spare register here.
      BuildMI(MBB, MBBI, dl, TII.get(ARM::tMOVgpr2tgpr), ARM::R4)
        .addReg(ARM::SP, RegState::Kill);
      AddDefaultCC(AddDefaultPred(BuildMI(MBB, MBBI, dl,
                                          TII.get(ARM::t2BICri), ARM::R4)
                                  .addReg(ARM::R4, RegState::Kill)
                                  .addImm(MaxAlign-1)));
      BuildMI(MBB, MBBI, dl, TII.get(ARM::tMOVtgpr2gpr), ARM::SP)
        .addReg(ARM::R4, RegState::Kill);
    }
  }
}

static bool isCalleeSavedRegister(unsigned Reg, const unsigned *CSRegs) {
  for (unsigned i = 0; CSRegs[i]; ++i)
    if (Reg == CSRegs[i])
      return true;
  return false;
}

static bool isCSRestore(MachineInstr *MI,
                        const ARMBaseInstrInfo &TII,
                        const unsigned *CSRegs) {
  return ((MI->getOpcode() == (int)ARM::VLDRD ||
           MI->getOpcode() == (int)ARM::LDR ||
           MI->getOpcode() == (int)ARM::t2LDRi12) &&
          MI->getOperand(1).isFI() &&
          isCalleeSavedRegister(MI->getOperand(0).getReg(), CSRegs));
}

void ARMBaseRegisterInfo::
emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = prior(MBB.end());
  assert(MBBI->getDesc().isReturn() &&
         "Can only insert epilog into returning blocks");
  DebugLoc dl = MBBI->getDebugLoc();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  assert(!AFI->isThumb1OnlyFunction() &&
         "This emitEpilogue does not support Thumb1!");
  bool isARM = !AFI->isThumbFunction();

  unsigned VARegSaveSize = AFI->getVarArgsRegSaveSize();
  int NumBytes = (int)MFI->getStackSize();

  if (!AFI->hasStackFrame()) {
    if (NumBytes != 0)
      emitSPUpdate(isARM, MBB, MBBI, dl, TII, NumBytes);
  } else {
    // Unwind MBBI to point to first LDR / VLDRD.
    const unsigned *CSRegs = getCalleeSavedRegs();
    if (MBBI != MBB.begin()) {
      do
        --MBBI;
      while (MBBI != MBB.begin() && isCSRestore(MBBI, TII, CSRegs));
      if (!isCSRestore(MBBI, TII, CSRegs))
        ++MBBI;
    }

    // Move SP to start of FP callee save spill area.
    NumBytes -= (AFI->getGPRCalleeSavedArea1Size() +
                 AFI->getGPRCalleeSavedArea2Size() +
                 AFI->getDPRCalleeSavedAreaSize());

    // Darwin ABI requires FP to point to the stack slot that contains the
    // previous FP.
    bool HasFP = hasFP(MF);
    if ((STI.isTargetDarwin() && NumBytes) || HasFP) {
      NumBytes = AFI->getFramePtrSpillOffset() - NumBytes;
      // Reset SP based on frame pointer only if the stack frame extends beyond
      // frame pointer stack slot or target is ELF and the function has FP.
      if (HasFP ||
          AFI->getGPRCalleeSavedArea2Size() ||
          AFI->getDPRCalleeSavedAreaSize()  ||
          AFI->getDPRCalleeSavedAreaOffset()) {
        if (NumBytes) {
          if (isARM)
            emitARMRegPlusImmediate(MBB, MBBI, dl, ARM::SP, FramePtr, -NumBytes,
                                    ARMCC::AL, 0, TII);
          else
            emitT2RegPlusImmediate(MBB, MBBI, dl, ARM::SP, FramePtr, -NumBytes,
                                    ARMCC::AL, 0, TII);
        } else {
          // Thumb2 or ARM.
          if (isARM)
            BuildMI(MBB, MBBI, dl, TII.get(ARM::MOVr), ARM::SP)
              .addReg(FramePtr)
              .addImm((unsigned)ARMCC::AL).addReg(0).addReg(0);
          else
            BuildMI(MBB, MBBI, dl, TII.get(ARM::tMOVgpr2gpr), ARM::SP)
              .addReg(FramePtr);
        }
      }
    } else if (NumBytes)
      emitSPUpdate(isARM, MBB, MBBI, dl, TII, NumBytes);

    // Move SP to start of integer callee save spill area 2.
    movePastCSLoadStoreOps(MBB, MBBI, ARM::VLDRD, 0, 3, STI);
    emitSPUpdate(isARM, MBB, MBBI, dl, TII, AFI->getDPRCalleeSavedAreaSize());

    // Move SP to start of integer callee save spill area 1.
    movePastCSLoadStoreOps(MBB, MBBI, ARM::LDR, ARM::t2LDRi12, 2, STI);
    emitSPUpdate(isARM, MBB, MBBI, dl, TII, AFI->getGPRCalleeSavedArea2Size());

    // Move SP to SP upon entry to the function.
    movePastCSLoadStoreOps(MBB, MBBI, ARM::LDR, ARM::t2LDRi12, 1, STI);
    emitSPUpdate(isARM, MBB, MBBI, dl, TII, AFI->getGPRCalleeSavedArea1Size());
  }

  if (VARegSaveSize)
    emitSPUpdate(isARM, MBB, MBBI, dl, TII, VARegSaveSize);
}

#include "ARMGenRegisterInfo.inc"
