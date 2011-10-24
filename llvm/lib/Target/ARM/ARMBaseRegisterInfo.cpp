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
#include "ARMBaseInstrInfo.h"
#include "ARMBaseRegisterInfo.h"
#include "ARMFrameLowering.h"
#include "ARMInstrInfo.h"
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
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"

#define GET_REGINFO_TARGET_DESC
#include "ARMGenRegisterInfo.inc"

using namespace llvm;

static cl::opt<bool>
ForceAllBaseRegAlloc("arm-force-base-reg-alloc", cl::Hidden, cl::init(false),
          cl::desc("Force use of virtual base registers for stack load/store"));
static cl::opt<bool>
EnableLocalStackAlloc("enable-local-stack-alloc", cl::init(true), cl::Hidden,
          cl::desc("Enable pre-regalloc stack frame index allocation"));
static cl::opt<bool>
EnableBasePointer("arm-use-base-pointer", cl::Hidden, cl::init(true),
          cl::desc("Enable use of a base pointer for complex stack frames"));

ARMBaseRegisterInfo::ARMBaseRegisterInfo(const ARMBaseInstrInfo &tii,
                                         const ARMSubtarget &sti)
  : ARMGenRegisterInfo(ARM::LR), TII(tii), STI(sti),
    FramePtr((STI.isTargetDarwin() || STI.isThumb()) ? ARM::R7 : ARM::R11),
    BasePtr(ARM::R6) {
}

const unsigned*
ARMBaseRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  bool ghcCall = false;

  if (MF) {
    const Function *F = MF->getFunction();
    ghcCall = (F ? F->getCallingConv() == CallingConv::GHC : false);
  }

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

  static const unsigned GhcCalleeSavedRegs[] = {
    0
  };

  return ghcCall ? GhcCalleeSavedRegs :
         STI.isTargetDarwin() ? DarwinCalleeSavedRegs : CalleeSavedRegs;
}

BitVector ARMBaseRegisterInfo::
getReservedRegs(const MachineFunction &MF) const {
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();

  // FIXME: avoid re-calculating this every time.
  BitVector Reserved(getNumRegs());
  Reserved.set(ARM::SP);
  Reserved.set(ARM::PC);
  Reserved.set(ARM::FPSCR);
  if (TFI->hasFP(MF))
    Reserved.set(FramePtr);
  if (hasBasePointer(MF))
    Reserved.set(BasePtr);
  // Some targets reserve R9.
  if (STI.isR9Reserved())
    Reserved.set(ARM::R9);
  // Reserve D16-D31 if the subtarget doesn't support them.
  if (!STI.hasVFP3() || STI.hasD16()) {
    assert(ARM::D31 == ARM::D16 + 15);
    for (unsigned i = 0; i != 16; ++i)
      Reserved.set(ARM::D16 + i);
  }
  return Reserved;
}

bool ARMBaseRegisterInfo::isReservedReg(const MachineFunction &MF,
                                        unsigned Reg) const {
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();

  switch (Reg) {
  default: break;
  case ARM::SP:
  case ARM::PC:
    return true;
  case ARM::R6:
    if (hasBasePointer(MF))
      return true;
    break;
  case ARM::R7:
  case ARM::R11:
    if (FramePtr == Reg && TFI->hasFP(MF))
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
  case ARM::ssub_0:
  case ARM::ssub_1:
  case ARM::ssub_2:
  case ARM::ssub_3: {
    // S sub-registers.
    if (A->getSize() == 8) {
      if (B == &ARM::SPR_8RegClass)
        return &ARM::DPR_8RegClass;
      assert(B == &ARM::SPRRegClass && "Expecting SPR register class!");
      if (A == &ARM::DPR_8RegClass)
        return A;
      return &ARM::DPR_VFP2RegClass;
    }

    if (A->getSize() == 16) {
      if (B == &ARM::SPR_8RegClass)
        return &ARM::QPR_8RegClass;
      return &ARM::QPR_VFP2RegClass;
    }

    if (A->getSize() == 32) {
      if (B == &ARM::SPR_8RegClass)
        return 0;  // Do not allow coalescing!
      return &ARM::QQPR_VFP2RegClass;
    }

    assert(A->getSize() == 64 && "Expecting a QQQQ register class!");
    return 0;  // Do not allow coalescing!
  }
  case ARM::dsub_0:
  case ARM::dsub_1:
  case ARM::dsub_2:
  case ARM::dsub_3: {
    // D sub-registers.
    if (A->getSize() == 16) {
      if (B == &ARM::DPR_VFP2RegClass)
        return &ARM::QPR_VFP2RegClass;
      if (B == &ARM::DPR_8RegClass)
        return 0;  // Do not allow coalescing!
      return A;
    }

    if (A->getSize() == 32) {
      if (B == &ARM::DPR_VFP2RegClass)
        return &ARM::QQPR_VFP2RegClass;
      if (B == &ARM::DPR_8RegClass)
        return 0;  // Do not allow coalescing!
      return A;
    }

    assert(A->getSize() == 64 && "Expecting a QQQQ register class!");
    if (B != &ARM::DPRRegClass)
      return 0;  // Do not allow coalescing!
    return A;
  }
  case ARM::dsub_4:
  case ARM::dsub_5:
  case ARM::dsub_6:
  case ARM::dsub_7: {
    // D sub-registers of QQQQ registers.
    if (A->getSize() == 64 && B == &ARM::DPRRegClass)
      return A;
    return 0;  // Do not allow coalescing!
  }

  case ARM::qsub_0:
  case ARM::qsub_1: {
    // Q sub-registers.
    if (A->getSize() == 32) {
      if (B == &ARM::QPR_VFP2RegClass)
        return &ARM::QQPR_VFP2RegClass;
      if (B == &ARM::QPR_8RegClass)
        return 0;  // Do not allow coalescing!
      return A;
    }

    assert(A->getSize() == 64 && "Expecting a QQQQ register class!");
    if (B == &ARM::QPRRegClass)
      return A;
    return 0;  // Do not allow coalescing!
  }
  case ARM::qsub_2:
  case ARM::qsub_3: {
    // Q sub-registers of QQQQ registers.
    if (A->getSize() == 64 && B == &ARM::QPRRegClass)
      return A;
    return 0;  // Do not allow coalescing!
  }
  }
  return 0;
}

bool
ARMBaseRegisterInfo::canCombineSubRegIndices(const TargetRegisterClass *RC,
                                          SmallVectorImpl<unsigned> &SubIndices,
                                          unsigned &NewSubIdx) const {

  unsigned Size = RC->getSize() * 8;
  if (Size < 6)
    return 0;

  NewSubIdx = 0;  // Whole register.
  unsigned NumRegs = SubIndices.size();
  if (NumRegs == 8) {
    // 8 D registers -> 1 QQQQ register.
    return (Size == 512 &&
            SubIndices[0] == ARM::dsub_0 &&
            SubIndices[1] == ARM::dsub_1 &&
            SubIndices[2] == ARM::dsub_2 &&
            SubIndices[3] == ARM::dsub_3 &&
            SubIndices[4] == ARM::dsub_4 &&
            SubIndices[5] == ARM::dsub_5 &&
            SubIndices[6] == ARM::dsub_6 &&
            SubIndices[7] == ARM::dsub_7);
  } else if (NumRegs == 4) {
    if (SubIndices[0] == ARM::qsub_0) {
      // 4 Q registers -> 1 QQQQ register.
      return (Size == 512 &&
              SubIndices[1] == ARM::qsub_1 &&
              SubIndices[2] == ARM::qsub_2 &&
              SubIndices[3] == ARM::qsub_3);
    } else if (SubIndices[0] == ARM::dsub_0) {
      // 4 D registers -> 1 QQ register.
      if (Size >= 256 &&
          SubIndices[1] == ARM::dsub_1 &&
          SubIndices[2] == ARM::dsub_2 &&
          SubIndices[3] == ARM::dsub_3) {
        if (Size == 512)
          NewSubIdx = ARM::qqsub_0;
        return true;
      }
    } else if (SubIndices[0] == ARM::dsub_4) {
      // 4 D registers -> 1 QQ register (2nd).
      if (Size == 512 &&
          SubIndices[1] == ARM::dsub_5 &&
          SubIndices[2] == ARM::dsub_6 &&
          SubIndices[3] == ARM::dsub_7) {
        NewSubIdx = ARM::qqsub_1;
        return true;
      }
    } else if (SubIndices[0] == ARM::ssub_0) {
      // 4 S registers -> 1 Q register.
      if (Size >= 128 &&
          SubIndices[1] == ARM::ssub_1 &&
          SubIndices[2] == ARM::ssub_2 &&
          SubIndices[3] == ARM::ssub_3) {
        if (Size >= 256)
          NewSubIdx = ARM::qsub_0;
        return true;
      }
    }
  } else if (NumRegs == 2) {
    if (SubIndices[0] == ARM::qsub_0) {
      // 2 Q registers -> 1 QQ register.
      if (Size >= 256 && SubIndices[1] == ARM::qsub_1) {
        if (Size == 512)
          NewSubIdx = ARM::qqsub_0;
        return true;
      }
    } else if (SubIndices[0] == ARM::qsub_2) {
      // 2 Q registers -> 1 QQ register (2nd).
      if (Size == 512 && SubIndices[1] == ARM::qsub_3) {
        NewSubIdx = ARM::qqsub_1;
        return true;
      }
    } else if (SubIndices[0] == ARM::dsub_0) {
      // 2 D registers -> 1 Q register.
      if (Size >= 128 && SubIndices[1] == ARM::dsub_1) {
        if (Size >= 256)
          NewSubIdx = ARM::qsub_0;
        return true;
      }
    } else if (SubIndices[0] == ARM::dsub_2) {
      // 2 D registers -> 1 Q register (2nd).
      if (Size >= 256 && SubIndices[1] == ARM::dsub_3) {
        NewSubIdx = ARM::qsub_1;
        return true;
      }
    } else if (SubIndices[0] == ARM::dsub_4) {
      // 2 D registers -> 1 Q register (3rd).
      if (Size == 512 && SubIndices[1] == ARM::dsub_5) {
        NewSubIdx = ARM::qsub_2;
        return true;
      }
    } else if (SubIndices[0] == ARM::dsub_6) {
      // 2 D registers -> 1 Q register (3rd).
      if (Size == 512 && SubIndices[1] == ARM::dsub_7) {
        NewSubIdx = ARM::qsub_3;
        return true;
      }
    } else if (SubIndices[0] == ARM::ssub_0) {
      // 2 S registers -> 1 D register.
      if (SubIndices[1] == ARM::ssub_1) {
        if (Size >= 128)
          NewSubIdx = ARM::dsub_0;
        return true;
      }
    } else if (SubIndices[0] == ARM::ssub_2) {
      // 2 S registers -> 1 D register (2nd).
      if (Size >= 128 && SubIndices[1] == ARM::ssub_3) {
        NewSubIdx = ARM::dsub_1;
        return true;
      }
    }
  }
  return false;
}

const TargetRegisterClass*
ARMBaseRegisterInfo::getLargestLegalSuperClass(const TargetRegisterClass *RC)
                                                                         const {
  const TargetRegisterClass *Super = RC;
  TargetRegisterClass::sc_iterator I = RC->getSuperClasses();
  do {
    switch (Super->getID()) {
    case ARM::GPRRegClassID:
    case ARM::SPRRegClassID:
    case ARM::DPRRegClassID:
    case ARM::QPRRegClassID:
    case ARM::QQPRRegClassID:
    case ARM::QQQQPRRegClassID:
      return Super;
    }
    Super = *I++;
  } while (Super);
  return RC;
}

const TargetRegisterClass *
ARMBaseRegisterInfo::getPointerRegClass(unsigned Kind) const {
  return ARM::GPRRegisterClass;
}

const TargetRegisterClass *
ARMBaseRegisterInfo::getCrossCopyRegClass(const TargetRegisterClass *RC) const {
  if (RC == &ARM::CCRRegClass)
    return 0;  // Can't copy CCR registers.
  return RC;
}

unsigned
ARMBaseRegisterInfo::getRegPressureLimit(const TargetRegisterClass *RC,
                                         MachineFunction &MF) const {
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();

  switch (RC->getID()) {
  default:
    return 0;
  case ARM::tGPRRegClassID:
    return TFI->hasFP(MF) ? 4 : 5;
  case ARM::GPRRegClassID: {
    unsigned FP = TFI->hasFP(MF) ? 1 : 0;
    return 10 - FP - (STI.isR9Reserved() ? 1 : 0);
  }
  case ARM::SPRRegClassID:  // Currently not used as 'rep' register class.
  case ARM::DPRRegClassID:
    return 32 - 10;
  }
}

/// getRawAllocationOrder - Returns the register allocation order for a
/// specified register class with a target-dependent hint.
ArrayRef<unsigned>
ARMBaseRegisterInfo::getRawAllocationOrder(const TargetRegisterClass *RC,
                                           unsigned HintType, unsigned HintReg,
                                           const MachineFunction &MF) const {
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();
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

  // We only support even/odd hints for GPR and rGPR.
  if (RC != ARM::GPRRegisterClass && RC != ARM::rGPRRegisterClass)
    return RC->getRawAllocationOrder(MF);

  if (HintType == ARMRI::RegPairEven) {
    if (isPhysicalRegister(HintReg) && getRegisterPairEven(HintReg, MF) == 0)
      // It's no longer possible to fulfill this hint. Return the default
      // allocation order.
      return RC->getRawAllocationOrder(MF);

    if (!TFI->hasFP(MF)) {
      if (!STI.isR9Reserved())
        return makeArrayRef(GPREven1);
      else
        return makeArrayRef(GPREven4);
    } else if (FramePtr == ARM::R7) {
      if (!STI.isR9Reserved())
        return makeArrayRef(GPREven2);
      else
        return makeArrayRef(GPREven5);
    } else { // FramePtr == ARM::R11
      if (!STI.isR9Reserved())
        return makeArrayRef(GPREven3);
      else
        return makeArrayRef(GPREven6);
    }
  } else if (HintType == ARMRI::RegPairOdd) {
    if (isPhysicalRegister(HintReg) && getRegisterPairOdd(HintReg, MF) == 0)
      // It's no longer possible to fulfill this hint. Return the default
      // allocation order.
      return RC->getRawAllocationOrder(MF);

    if (!TFI->hasFP(MF)) {
      if (!STI.isR9Reserved())
        return makeArrayRef(GPROdd1);
      else
        return makeArrayRef(GPROdd4);
    } else if (FramePtr == ARM::R7) {
      if (!STI.isR9Reserved())
        return makeArrayRef(GPROdd2);
      else
        return makeArrayRef(GPROdd5);
    } else { // FramePtr == ARM::R11
      if (!STI.isR9Reserved())
        return makeArrayRef(GPROdd3);
      else
        return makeArrayRef(GPROdd6);
    }
  }
  return RC->getRawAllocationOrder(MF);
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
      TargetRegisterInfo::isVirtualRegister(Hint.second)) {
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

bool
ARMBaseRegisterInfo::avoidWriteAfterWrite(const TargetRegisterClass *RC) const {
  // CortexA9 has a Write-after-write hazard for NEON registers.
  if (!STI.isCortexA9())
    return false;

  switch (RC->getID()) {
  case ARM::DPRRegClassID:
  case ARM::DPR_8RegClassID:
  case ARM::DPR_VFP2RegClassID:
  case ARM::QPRRegClassID:
  case ARM::QPR_8RegClassID:
  case ARM::QPR_VFP2RegClassID:
  case ARM::SPRRegClassID:
  case ARM::SPR_8RegClassID:
    // Avoid reusing S, D, and Q registers.
    // Don't increase register pressure for QQ and QQQQ.
    return true;
  default:
    return false;
  }
}

bool ARMBaseRegisterInfo::hasBasePointer(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  const ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();

  if (!EnableBasePointer)
    return false;

  if (needsStackRealignment(MF) && MFI->hasVarSizedObjects())
    return true;

  // Thumb has trouble with negative offsets from the FP. Thumb2 has a limited
  // negative range for ldr/str (255), and thumb1 is positive offsets only.
  // It's going to be better to use the SP or Base Pointer instead. When there
  // are variable sized objects, we can't reference off of the SP, so we
  // reserve a Base Pointer.
  if (AFI->isThumbFunction() && MFI->hasVarSizedObjects()) {
    // Conservatively estimate whether the negative offset from the frame
    // pointer will be sufficient to reach. If a function has a smallish
    // frame, it's less likely to have lots of spills and callee saved
    // space, so it's all more likely to be within range of the frame pointer.
    // If it's wrong, the scavenger will still enable access to work, it just
    // won't be optimal.
    if (AFI->isThumb2Function() && MFI->getLocalFrameSize() < 128)
      return false;
    return true;
  }

  return false;
}

bool ARMBaseRegisterInfo::canRealignStack(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  const ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  // We can't realign the stack if:
  // 1. Dynamic stack realignment is explicitly disabled,
  // 2. This is a Thumb1 function (it's not useful, so we don't bother), or
  // 3. There are VLAs in the function and the base pointer is disabled.
  return (RealignStack && !AFI->isThumb1OnlyFunction() &&
          (!MFI->hasVarSizedObjects() || EnableBasePointer));
}

bool ARMBaseRegisterInfo::
needsStackRealignment(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  const Function *F = MF.getFunction();
  unsigned StackAlign = MF.getTarget().getFrameLowering()->getStackAlignment();
  bool requiresRealignment = ((MFI->getLocalFrameMaxAlign() > StackAlign) ||
                               F->hasFnAttr(Attribute::StackAlignment));

  return requiresRealignment && canRealignStack(MF);
}

bool ARMBaseRegisterInfo::
cannotEliminateFrame(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  if (DisableFramePointerElim(MF) && MFI->adjustsStack())
    return true;
  return MFI->hasVarSizedObjects() || MFI->isFrameAddressTaken()
    || needsStackRealignment(MF);
}

unsigned
ARMBaseRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();

  if (TFI->hasFP(MF))
    return FramePtr;
  return ARM::SP;
}

unsigned ARMBaseRegisterInfo::getEHExceptionRegister() const {
  llvm_unreachable("What is the exception register");
  return 0;
}

unsigned ARMBaseRegisterInfo::getEHHandlerRegister() const {
  llvm_unreachable("What is the exception handler register");
  return 0;
}

unsigned ARMBaseRegisterInfo::getRegisterPairEven(unsigned Reg,
                                              const MachineFunction &MF) const {
  switch (Reg) {
  default: break;
  // Return 0 if either register of the pair is a special register.
  // So no R12, etc.
  case ARM::R1: return ARM::R0;
  case ARM::R3: return ARM::R2;
  case ARM::R5: return ARM::R4;
  case ARM::R7:
    return (isReservedReg(MF, ARM::R7) || isReservedReg(MF, ARM::R6))
      ? 0 : ARM::R6;
  case ARM::R9: return isReservedReg(MF, ARM::R9)  ? 0 :ARM::R8;
  case ARM::R11: return isReservedReg(MF, ARM::R11) ? 0 : ARM::R10;

  case ARM::S1: return ARM::S0;
  case ARM::S3: return ARM::S2;
  case ARM::S5: return ARM::S4;
  case ARM::S7: return ARM::S6;
  case ARM::S9: return ARM::S8;
  case ARM::S11: return ARM::S10;
  case ARM::S13: return ARM::S12;
  case ARM::S15: return ARM::S14;
  case ARM::S17: return ARM::S16;
  case ARM::S19: return ARM::S18;
  case ARM::S21: return ARM::S20;
  case ARM::S23: return ARM::S22;
  case ARM::S25: return ARM::S24;
  case ARM::S27: return ARM::S26;
  case ARM::S29: return ARM::S28;
  case ARM::S31: return ARM::S30;

  case ARM::D1: return ARM::D0;
  case ARM::D3: return ARM::D2;
  case ARM::D5: return ARM::D4;
  case ARM::D7: return ARM::D6;
  case ARM::D9: return ARM::D8;
  case ARM::D11: return ARM::D10;
  case ARM::D13: return ARM::D12;
  case ARM::D15: return ARM::D14;
  case ARM::D17: return ARM::D16;
  case ARM::D19: return ARM::D18;
  case ARM::D21: return ARM::D20;
  case ARM::D23: return ARM::D22;
  case ARM::D25: return ARM::D24;
  case ARM::D27: return ARM::D26;
  case ARM::D29: return ARM::D28;
  case ARM::D31: return ARM::D30;
  }

  return 0;
}

unsigned ARMBaseRegisterInfo::getRegisterPairOdd(unsigned Reg,
                                             const MachineFunction &MF) const {
  switch (Reg) {
  default: break;
  // Return 0 if either register of the pair is a special register.
  // So no R12, etc.
  case ARM::R0: return ARM::R1;
  case ARM::R2: return ARM::R3;
  case ARM::R4: return ARM::R5;
  case ARM::R6:
    return (isReservedReg(MF, ARM::R7) || isReservedReg(MF, ARM::R6))
      ? 0 : ARM::R7;
  case ARM::R8: return isReservedReg(MF, ARM::R9)  ? 0 :ARM::R9;
  case ARM::R10: return isReservedReg(MF, ARM::R11) ? 0 : ARM::R11;

  case ARM::S0: return ARM::S1;
  case ARM::S2: return ARM::S3;
  case ARM::S4: return ARM::S5;
  case ARM::S6: return ARM::S7;
  case ARM::S8: return ARM::S9;
  case ARM::S10: return ARM::S11;
  case ARM::S12: return ARM::S13;
  case ARM::S14: return ARM::S15;
  case ARM::S16: return ARM::S17;
  case ARM::S18: return ARM::S19;
  case ARM::S20: return ARM::S21;
  case ARM::S22: return ARM::S23;
  case ARM::S24: return ARM::S25;
  case ARM::S26: return ARM::S27;
  case ARM::S28: return ARM::S29;
  case ARM::S30: return ARM::S31;

  case ARM::D0: return ARM::D1;
  case ARM::D2: return ARM::D3;
  case ARM::D4: return ARM::D5;
  case ARM::D6: return ARM::D7;
  case ARM::D8: return ARM::D9;
  case ARM::D10: return ARM::D11;
  case ARM::D12: return ARM::D13;
  case ARM::D14: return ARM::D15;
  case ARM::D16: return ARM::D17;
  case ARM::D18: return ARM::D19;
  case ARM::D20: return ARM::D21;
  case ARM::D22: return ARM::D23;
  case ARM::D24: return ARM::D25;
  case ARM::D26: return ARM::D27;
  case ARM::D28: return ARM::D29;
  case ARM::D30: return ARM::D31;
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
                  unsigned PredReg, unsigned MIFlags) const {
  MachineFunction &MF = *MBB.getParent();
  MachineConstantPool *ConstantPool = MF.getConstantPool();
  const Constant *C =
        ConstantInt::get(Type::getInt32Ty(MF.getFunction()->getContext()), Val);
  unsigned Idx = ConstantPool->getConstantPoolIndex(C, 4);

  BuildMI(MBB, MBBI, dl, TII.get(ARM::LDRcp))
    .addReg(DestReg, getDefRegState(true), SubIdx)
    .addConstantPoolIndex(Idx)
    .addImm(0).addImm(Pred).addReg(PredReg)
    .setMIFlags(MIFlags);
}

bool ARMBaseRegisterInfo::
requiresRegisterScavenging(const MachineFunction &MF) const {
  return true;
}

bool ARMBaseRegisterInfo::
requiresFrameIndexScavenging(const MachineFunction &MF) const {
  return true;
}

bool ARMBaseRegisterInfo::
requiresVirtualBaseRegisters(const MachineFunction &MF) const {
  return EnableLocalStackAlloc;
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

int64_t ARMBaseRegisterInfo::
getFrameIndexInstrOffset(const MachineInstr *MI, int Idx) const {
  const MCInstrDesc &Desc = MI->getDesc();
  unsigned AddrMode = (Desc.TSFlags & ARMII::AddrModeMask);
  int64_t InstrOffs = 0;;
  int Scale = 1;
  unsigned ImmIdx = 0;
  switch (AddrMode) {
  case ARMII::AddrModeT2_i8:
  case ARMII::AddrModeT2_i12:
  case ARMII::AddrMode_i12:
    InstrOffs = MI->getOperand(Idx+1).getImm();
    Scale = 1;
    break;
  case ARMII::AddrMode5: {
    // VFP address mode.
    const MachineOperand &OffOp = MI->getOperand(Idx+1);
    InstrOffs = ARM_AM::getAM5Offset(OffOp.getImm());
    if (ARM_AM::getAM5Op(OffOp.getImm()) == ARM_AM::sub)
      InstrOffs = -InstrOffs;
    Scale = 4;
    break;
  }
  case ARMII::AddrMode2: {
    ImmIdx = Idx+2;
    InstrOffs = ARM_AM::getAM2Offset(MI->getOperand(ImmIdx).getImm());
    if (ARM_AM::getAM2Op(MI->getOperand(ImmIdx).getImm()) == ARM_AM::sub)
      InstrOffs = -InstrOffs;
    break;
  }
  case ARMII::AddrMode3: {
    ImmIdx = Idx+2;
    InstrOffs = ARM_AM::getAM3Offset(MI->getOperand(ImmIdx).getImm());
    if (ARM_AM::getAM3Op(MI->getOperand(ImmIdx).getImm()) == ARM_AM::sub)
      InstrOffs = -InstrOffs;
    break;
  }
  case ARMII::AddrModeT1_s: {
    ImmIdx = Idx+1;
    InstrOffs = MI->getOperand(ImmIdx).getImm();
    Scale = 4;
    break;
  }
  default:
    llvm_unreachable("Unsupported addressing mode!");
    break;
  }

  return InstrOffs * Scale;
}

/// needsFrameBaseReg - Returns true if the instruction's frame index
/// reference would be better served by a base register other than FP
/// or SP. Used by LocalStackFrameAllocation to determine which frame index
/// references it should create new base registers for.
bool ARMBaseRegisterInfo::
needsFrameBaseReg(MachineInstr *MI, int64_t Offset) const {
  for (unsigned i = 0; !MI->getOperand(i).isFI(); ++i) {
    assert(i < MI->getNumOperands() &&"Instr doesn't have FrameIndex operand!");
  }

  // It's the load/store FI references that cause issues, as it can be difficult
  // to materialize the offset if it won't fit in the literal field. Estimate
  // based on the size of the local frame and some conservative assumptions
  // about the rest of the stack frame (note, this is pre-regalloc, so
  // we don't know everything for certain yet) whether this offset is likely
  // to be out of range of the immediate. Return true if so.

  // We only generate virtual base registers for loads and stores, so
  // return false for everything else.
  unsigned Opc = MI->getOpcode();
  switch (Opc) {
  case ARM::LDRi12: case ARM::LDRH: case ARM::LDRBi12:
  case ARM::STRi12: case ARM::STRH: case ARM::STRBi12:
  case ARM::t2LDRi12: case ARM::t2LDRi8:
  case ARM::t2STRi12: case ARM::t2STRi8:
  case ARM::VLDRS: case ARM::VLDRD:
  case ARM::VSTRS: case ARM::VSTRD:
  case ARM::tSTRspi: case ARM::tLDRspi:
    if (ForceAllBaseRegAlloc)
      return true;
    break;
  default:
    return false;
  }

  // Without a virtual base register, if the function has variable sized
  // objects, all fixed-size local references will be via the frame pointer,
  // Approximate the offset and see if it's legal for the instruction.
  // Note that the incoming offset is based on the SP value at function entry,
  // so it'll be negative.
  MachineFunction &MF = *MI->getParent()->getParent();
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();

  // Estimate an offset from the frame pointer.
  // Conservatively assume all callee-saved registers get pushed. R4-R6
  // will be earlier than the FP, so we ignore those.
  // R7, LR
  int64_t FPOffset = Offset - 8;
  // ARM and Thumb2 functions also need to consider R8-R11 and D8-D15
  if (!AFI->isThumbFunction() || !AFI->isThumb1OnlyFunction())
    FPOffset -= 80;
  // Estimate an offset from the stack pointer.
  // The incoming offset is relating to the SP at the start of the function,
  // but when we access the local it'll be relative to the SP after local
  // allocation, so adjust our SP-relative offset by that allocation size.
  Offset = -Offset;
  Offset += MFI->getLocalFrameSize();
  // Assume that we'll have at least some spill slots allocated.
  // FIXME: This is a total SWAG number. We should run some statistics
  //        and pick a real one.
  Offset += 128; // 128 bytes of spill slots

  // If there is a frame pointer, try using it.
  // The FP is only available if there is no dynamic realignment. We
  // don't know for sure yet whether we'll need that, so we guess based
  // on whether there are any local variables that would trigger it.
  unsigned StackAlign = TFI->getStackAlignment();
  if (TFI->hasFP(MF) &&
      !((MFI->getLocalFrameMaxAlign() > StackAlign) && canRealignStack(MF))) {
    if (isFrameOffsetLegal(MI, FPOffset))
      return false;
  }
  // If we can reference via the stack pointer, try that.
  // FIXME: This (and the code that resolves the references) can be improved
  //        to only disallow SP relative references in the live range of
  //        the VLA(s). In practice, it's unclear how much difference that
  //        would make, but it may be worth doing.
  if (!MFI->hasVarSizedObjects() && isFrameOffsetLegal(MI, Offset))
    return false;

  // The offset likely isn't legal, we want to allocate a virtual base register.
  return true;
}

/// materializeFrameBaseRegister - Insert defining instruction(s) for BaseReg to
/// be a pointer to FrameIdx at the beginning of the basic block.
void ARMBaseRegisterInfo::
materializeFrameBaseRegister(MachineBasicBlock *MBB,
                             unsigned BaseReg, int FrameIdx,
                             int64_t Offset) const {
  ARMFunctionInfo *AFI = MBB->getParent()->getInfo<ARMFunctionInfo>();
  unsigned ADDriOpc = !AFI->isThumbFunction() ? ARM::ADDri :
    (AFI->isThumb1OnlyFunction() ? ARM::tADDrSPi : ARM::t2ADDri);

  MachineBasicBlock::iterator Ins = MBB->begin();
  DebugLoc DL;                  // Defaults to "unknown"
  if (Ins != MBB->end())
    DL = Ins->getDebugLoc();

  const MCInstrDesc &MCID = TII.get(ADDriOpc);
  MachineRegisterInfo &MRI = MBB->getParent()->getRegInfo();
  MRI.constrainRegClass(BaseReg, TII.getRegClass(MCID, 0, this));

  MachineInstrBuilder MIB = AddDefaultPred(BuildMI(*MBB, Ins, DL, MCID, BaseReg)
    .addFrameIndex(FrameIdx).addImm(Offset));

  if (!AFI->isThumb1OnlyFunction())
    AddDefaultCC(MIB);
}

void
ARMBaseRegisterInfo::resolveFrameIndex(MachineBasicBlock::iterator I,
                                       unsigned BaseReg, int64_t Offset) const {
  MachineInstr &MI = *I;
  MachineBasicBlock &MBB = *MI.getParent();
  MachineFunction &MF = *MBB.getParent();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  int Off = Offset; // ARM doesn't need the general 64-bit offsets
  unsigned i = 0;

  assert(!AFI->isThumb1OnlyFunction() &&
         "This resolveFrameIndex does not support Thumb1!");

  while (!MI.getOperand(i).isFI()) {
    ++i;
    assert(i < MI.getNumOperands() && "Instr doesn't have FrameIndex operand!");
  }
  bool Done = false;
  if (!AFI->isThumbFunction())
    Done = rewriteARMFrameIndex(MI, i, BaseReg, Off, TII);
  else {
    assert(AFI->isThumb2Function());
    Done = rewriteT2FrameIndex(MI, i, BaseReg, Off, TII);
  }
  assert (Done && "Unable to resolve frame index!");
  (void)Done;
}

bool ARMBaseRegisterInfo::isFrameOffsetLegal(const MachineInstr *MI,
                                             int64_t Offset) const {
  const MCInstrDesc &Desc = MI->getDesc();
  unsigned AddrMode = (Desc.TSFlags & ARMII::AddrModeMask);
  unsigned i = 0;

  while (!MI->getOperand(i).isFI()) {
    ++i;
    assert(i < MI->getNumOperands() &&"Instr doesn't have FrameIndex operand!");
  }

  // AddrMode4 and AddrMode6 cannot handle any offset.
  if (AddrMode == ARMII::AddrMode4 || AddrMode == ARMII::AddrMode6)
    return Offset == 0;

  unsigned NumBits = 0;
  unsigned Scale = 1;
  bool isSigned = true;
  switch (AddrMode) {
  case ARMII::AddrModeT2_i8:
  case ARMII::AddrModeT2_i12:
    // i8 supports only negative, and i12 supports only positive, so
    // based on Offset sign, consider the appropriate instruction
    Scale = 1;
    if (Offset < 0) {
      NumBits = 8;
      Offset = -Offset;
    } else {
      NumBits = 12;
    }
    break;
  case ARMII::AddrMode5:
    // VFP address mode.
    NumBits = 8;
    Scale = 4;
    break;
  case ARMII::AddrMode_i12:
  case ARMII::AddrMode2:
    NumBits = 12;
    break;
  case ARMII::AddrMode3:
    NumBits = 8;
    break;
  case ARMII::AddrModeT1_s:
    NumBits = 5;
    Scale = 4;
    isSigned = false;
    break;
  default:
    llvm_unreachable("Unsupported addressing mode!");
    break;
  }

  Offset += getFrameIndexInstrOffset(MI, i);
  // Make sure the offset is encodable for instructions that scale the
  // immediate.
  if ((Offset & (Scale-1)) != 0)
    return false;

  if (isSigned && Offset < 0)
    Offset = -Offset;

  unsigned Mask = (1 << NumBits) - 1;
  if ((unsigned)Offset <= Mask * Scale)
    return true;

  return false;
}

void
ARMBaseRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                         int SPAdj, RegScavenger *RS) const {
  unsigned i = 0;
  MachineInstr &MI = *II;
  MachineBasicBlock &MBB = *MI.getParent();
  MachineFunction &MF = *MBB.getParent();
  const ARMFrameLowering *TFI =
    static_cast<const ARMFrameLowering*>(MF.getTarget().getFrameLowering());
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  assert(!AFI->isThumb1OnlyFunction() &&
         "This eliminateFrameIndex does not support Thumb1!");

  while (!MI.getOperand(i).isFI()) {
    ++i;
    assert(i < MI.getNumOperands() && "Instr doesn't have FrameIndex operand!");
  }

  int FrameIndex = MI.getOperand(i).getIndex();
  unsigned FrameReg;

  int Offset = TFI->ResolveFrameIndexReference(MF, FrameIndex, FrameReg, SPAdj);

  // Special handling of dbg_value instructions.
  if (MI.isDebugValue()) {
    MI.getOperand(i).  ChangeToRegister(FrameReg, false /*isDef*/);
    MI.getOperand(i+1).ChangeToImmediate(Offset);
    return;
  }

  // Modify MI as necessary to handle as much of 'Offset' as possible
  bool Done = false;
  if (!AFI->isThumbFunction())
    Done = rewriteARMFrameIndex(MI, i, FrameReg, Offset, TII);
  else {
    assert(AFI->isThumb2Function());
    Done = rewriteT2FrameIndex(MI, i, FrameReg, Offset, TII);
  }
  if (Done)
    return;

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
    if (!AFI->isThumbFunction())
      emitARMRegPlusImmediate(MBB, II, MI.getDebugLoc(), ScratchReg, FrameReg,
                              Offset, Pred, PredReg, TII);
    else {
      assert(AFI->isThumb2Function());
      emitT2RegPlusImmediate(MBB, II, MI.getDebugLoc(), ScratchReg, FrameReg,
                             Offset, Pred, PredReg, TII);
    }
    // Update the original instruction to use the scratch register.
    MI.getOperand(i).ChangeToRegister(ScratchReg, false, false, true);
  }
}
