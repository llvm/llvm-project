//===- X86RegisterInfo.cpp - X86 Register Information -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the X86 implementation of the TargetRegisterInfo class.
// This file is responsible for the frame pointer elimination optimization
// on X86.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86RegisterInfo.h"
#include "X86InstrBuilder.h"
#include "X86MachineFunctionInfo.h"
#include "X86Subtarget.h"
#include "X86TargetMachine.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Type.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
using namespace llvm;

X86RegisterInfo::X86RegisterInfo(X86TargetMachine &tm,
                                 const TargetInstrInfo &tii)
  : X86GenRegisterInfo(tm.getSubtarget<X86Subtarget>().is64Bit() ?
                         X86::ADJCALLSTACKDOWN64 :
                         X86::ADJCALLSTACKDOWN32,
                       tm.getSubtarget<X86Subtarget>().is64Bit() ?
                         X86::ADJCALLSTACKUP64 :
                         X86::ADJCALLSTACKUP32),
    TM(tm), TII(tii) {
  // Cache some information.
  const X86Subtarget *Subtarget = &TM.getSubtarget<X86Subtarget>();
  Is64Bit = Subtarget->is64Bit();
  IsWin64 = Subtarget->isTargetWin64();
  StackAlign = TM.getFrameInfo()->getStackAlignment();

  if (Is64Bit) {
    SlotSize = 8;
    StackPtr = X86::RSP;
    FramePtr = X86::RBP;
  } else {
    SlotSize = 4;
    StackPtr = X86::ESP;
    FramePtr = X86::EBP;
  }
}

/// getDwarfRegNum - This function maps LLVM register identifiers to the DWARF
/// specific numbering, used in debug info and exception tables.
int X86RegisterInfo::getDwarfRegNum(unsigned RegNo, bool isEH) const {
  const X86Subtarget *Subtarget = &TM.getSubtarget<X86Subtarget>();
  unsigned Flavour = DWARFFlavour::X86_64;

  if (!Subtarget->is64Bit()) {
    if (Subtarget->isTargetDarwin()) {
      if (isEH)
        Flavour = DWARFFlavour::X86_32_DarwinEH;
      else
        Flavour = DWARFFlavour::X86_32_Generic;
    } else if (Subtarget->isTargetCygMing()) {
      // Unsupported by now, just quick fallback
      Flavour = DWARFFlavour::X86_32_Generic;
    } else {
      Flavour = DWARFFlavour::X86_32_Generic;
    }
  }

  return X86GenRegisterInfo::getDwarfRegNumFull(RegNo, Flavour);
}

/// getX86RegNum - This function maps LLVM register identifiers to their X86
/// specific numbering, which is used in various places encoding instructions.
unsigned X86RegisterInfo::getX86RegNum(unsigned RegNo) {
  switch(RegNo) {
  case X86::RAX: case X86::EAX: case X86::AX: case X86::AL: return N86::EAX;
  case X86::RCX: case X86::ECX: case X86::CX: case X86::CL: return N86::ECX;
  case X86::RDX: case X86::EDX: case X86::DX: case X86::DL: return N86::EDX;
  case X86::RBX: case X86::EBX: case X86::BX: case X86::BL: return N86::EBX;
  case X86::RSP: case X86::ESP: case X86::SP: case X86::SPL: case X86::AH:
    return N86::ESP;
  case X86::RBP: case X86::EBP: case X86::BP: case X86::BPL: case X86::CH:
    return N86::EBP;
  case X86::RSI: case X86::ESI: case X86::SI: case X86::SIL: case X86::DH:
    return N86::ESI;
  case X86::RDI: case X86::EDI: case X86::DI: case X86::DIL: case X86::BH:
    return N86::EDI;

  case X86::R8:  case X86::R8D:  case X86::R8W:  case X86::R8B:
    return N86::EAX;
  case X86::R9:  case X86::R9D:  case X86::R9W:  case X86::R9B:
    return N86::ECX;
  case X86::R10: case X86::R10D: case X86::R10W: case X86::R10B:
    return N86::EDX;
  case X86::R11: case X86::R11D: case X86::R11W: case X86::R11B:
    return N86::EBX;
  case X86::R12: case X86::R12D: case X86::R12W: case X86::R12B:
    return N86::ESP;
  case X86::R13: case X86::R13D: case X86::R13W: case X86::R13B:
    return N86::EBP;
  case X86::R14: case X86::R14D: case X86::R14W: case X86::R14B:
    return N86::ESI;
  case X86::R15: case X86::R15D: case X86::R15W: case X86::R15B:
    return N86::EDI;

  case X86::ST0: case X86::ST1: case X86::ST2: case X86::ST3:
  case X86::ST4: case X86::ST5: case X86::ST6: case X86::ST7:
    return RegNo-X86::ST0;

  case X86::XMM0: case X86::XMM8: case X86::MM0:
    return 0;
  case X86::XMM1: case X86::XMM9: case X86::MM1:
    return 1;
  case X86::XMM2: case X86::XMM10: case X86::MM2:
    return 2;
  case X86::XMM3: case X86::XMM11: case X86::MM3:
    return 3;
  case X86::XMM4: case X86::XMM12: case X86::MM4:
    return 4;
  case X86::XMM5: case X86::XMM13: case X86::MM5:
    return 5;
  case X86::XMM6: case X86::XMM14: case X86::MM6:
    return 6;
  case X86::XMM7: case X86::XMM15: case X86::MM7:
    return 7;

  default:
    assert(isVirtualRegister(RegNo) && "Unknown physical register!");
    llvm_unreachable("Register allocator hasn't allocated reg correctly yet!");
    return 0;
  }
}

const TargetRegisterClass *
X86RegisterInfo::getMatchingSuperRegClass(const TargetRegisterClass *A,
                                          const TargetRegisterClass *B,
                                          unsigned SubIdx) const {
  switch (SubIdx) {
  default: return 0;
  case 1:
    // 8-bit
    if (B == &X86::GR8RegClass) {
      if (A->getSize() == 2 || A->getSize() == 4 || A->getSize() == 8)
        return A;
    } else if (B == &X86::GR8_ABCD_LRegClass || B == &X86::GR8_ABCD_HRegClass) {
      if (A == &X86::GR64RegClass || A == &X86::GR64_ABCDRegClass ||
          A == &X86::GR64_NOREXRegClass ||
          A == &X86::GR64_NOSPRegClass ||
          A == &X86::GR64_NOREX_NOSPRegClass)
        return &X86::GR64_ABCDRegClass;
      else if (A == &X86::GR32RegClass || A == &X86::GR32_ABCDRegClass ||
               A == &X86::GR32_NOREXRegClass ||
               A == &X86::GR32_NOSPRegClass)
        return &X86::GR32_ABCDRegClass;
      else if (A == &X86::GR16RegClass || A == &X86::GR16_ABCDRegClass ||
               A == &X86::GR16_NOREXRegClass)
        return &X86::GR16_ABCDRegClass;
    } else if (B == &X86::GR8_NOREXRegClass) {
      if (A == &X86::GR64RegClass || A == &X86::GR64_NOREXRegClass ||
          A == &X86::GR64_NOSPRegClass || A == &X86::GR64_NOREX_NOSPRegClass)
        return &X86::GR64_NOREXRegClass;
      else if (A == &X86::GR64_ABCDRegClass)
        return &X86::GR64_ABCDRegClass;
      else if (A == &X86::GR32RegClass || A == &X86::GR32_NOREXRegClass ||
               A == &X86::GR32_NOSPRegClass)
        return &X86::GR32_NOREXRegClass;
      else if (A == &X86::GR32_ABCDRegClass)
        return &X86::GR32_ABCDRegClass;
      else if (A == &X86::GR16RegClass || A == &X86::GR16_NOREXRegClass)
        return &X86::GR16_NOREXRegClass;
      else if (A == &X86::GR16_ABCDRegClass)
        return &X86::GR16_ABCDRegClass;
    }
    break;
  case 2:
    // 8-bit hi
    if (B == &X86::GR8_ABCD_HRegClass) {
      if (A == &X86::GR64RegClass || A == &X86::GR64_ABCDRegClass ||
          A == &X86::GR64_NOREXRegClass ||
          A == &X86::GR64_NOSPRegClass ||
          A == &X86::GR64_NOREX_NOSPRegClass)
        return &X86::GR64_ABCDRegClass;
      else if (A == &X86::GR32RegClass || A == &X86::GR32_ABCDRegClass ||
               A == &X86::GR32_NOREXRegClass || A == &X86::GR32_NOSPRegClass)
        return &X86::GR32_ABCDRegClass;
      else if (A == &X86::GR16RegClass || A == &X86::GR16_ABCDRegClass ||
               A == &X86::GR16_NOREXRegClass)
        return &X86::GR16_ABCDRegClass;
    }
    break;
  case 3:
    // 16-bit
    if (B == &X86::GR16RegClass) {
      if (A->getSize() == 4 || A->getSize() == 8)
        return A;
    } else if (B == &X86::GR16_ABCDRegClass) {
      if (A == &X86::GR64RegClass || A == &X86::GR64_ABCDRegClass ||
          A == &X86::GR64_NOREXRegClass ||
          A == &X86::GR64_NOSPRegClass ||
          A == &X86::GR64_NOREX_NOSPRegClass)
        return &X86::GR64_ABCDRegClass;
      else if (A == &X86::GR32RegClass || A == &X86::GR32_ABCDRegClass ||
               A == &X86::GR32_NOREXRegClass || A == &X86::GR32_NOSPRegClass)
        return &X86::GR32_ABCDRegClass;
    } else if (B == &X86::GR16_NOREXRegClass) {
      if (A == &X86::GR64RegClass || A == &X86::GR64_NOREXRegClass ||
          A == &X86::GR64_NOSPRegClass || A == &X86::GR64_NOREX_NOSPRegClass)
        return &X86::GR64_NOREXRegClass;
      else if (A == &X86::GR64_ABCDRegClass)
        return &X86::GR64_ABCDRegClass;
      else if (A == &X86::GR32RegClass || A == &X86::GR32_NOREXRegClass ||
               A == &X86::GR32_NOSPRegClass)
        return &X86::GR32_NOREXRegClass;
      else if (A == &X86::GR32_ABCDRegClass)
        return &X86::GR64_ABCDRegClass;
    }
    break;
  case 4:
    // 32-bit
    if (B == &X86::GR32RegClass || B == &X86::GR32_NOSPRegClass) {
      if (A->getSize() == 8)
        return A;
    } else if (B == &X86::GR32_ABCDRegClass) {
      if (A == &X86::GR64RegClass || A == &X86::GR64_ABCDRegClass ||
          A == &X86::GR64_NOREXRegClass ||
          A == &X86::GR64_NOSPRegClass ||
          A == &X86::GR64_NOREX_NOSPRegClass)
        return &X86::GR64_ABCDRegClass;
    } else if (B == &X86::GR32_NOREXRegClass) {
      if (A == &X86::GR64RegClass || A == &X86::GR64_NOREXRegClass ||
          A == &X86::GR64_NOSPRegClass || A == &X86::GR64_NOREX_NOSPRegClass)
        return &X86::GR64_NOREXRegClass;
      else if (A == &X86::GR64_ABCDRegClass)
        return &X86::GR64_ABCDRegClass;
    }
    break;
  }
  return 0;
}

const TargetRegisterClass *
X86RegisterInfo::getPointerRegClass(unsigned Kind) const {
  switch (Kind) {
  default: llvm_unreachable("Unexpected Kind in getPointerRegClass!");
  case 0: // Normal GPRs.
    if (TM.getSubtarget<X86Subtarget>().is64Bit())
      return &X86::GR64RegClass;
    return &X86::GR32RegClass;
  case 1: // Normal GRPs except the stack pointer (for encoding reasons).
    if (TM.getSubtarget<X86Subtarget>().is64Bit())
      return &X86::GR64_NOSPRegClass;
    return &X86::GR32_NOSPRegClass;
  }
}

const TargetRegisterClass *
X86RegisterInfo::getCrossCopyRegClass(const TargetRegisterClass *RC) const {
  if (RC == &X86::CCRRegClass) {
    if (Is64Bit)
      return &X86::GR64RegClass;
    else
      return &X86::GR32RegClass;
  }
  return NULL;
}

const unsigned *
X86RegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  bool callsEHReturn = false;

  if (MF) {
    const MachineFrameInfo *MFI = MF->getFrameInfo();
    const MachineModuleInfo *MMI = MFI->getMachineModuleInfo();
    callsEHReturn = (MMI ? MMI->callsEHReturn() : false);
  }

  static const unsigned CalleeSavedRegs32Bit[] = {
    X86::ESI, X86::EDI, X86::EBX, X86::EBP,  0
  };

  static const unsigned CalleeSavedRegs32EHRet[] = {
    X86::EAX, X86::EDX, X86::ESI, X86::EDI, X86::EBX, X86::EBP,  0
  };

  static const unsigned CalleeSavedRegs64Bit[] = {
    X86::RBX, X86::R12, X86::R13, X86::R14, X86::R15, X86::RBP, 0
  };

  static const unsigned CalleeSavedRegs64EHRet[] = {
    X86::RAX, X86::RDX, X86::RBX, X86::R12,
    X86::R13, X86::R14, X86::R15, X86::RBP, 0
  };

  static const unsigned CalleeSavedRegsWin64[] = {
    X86::RBX,   X86::RBP,   X86::RDI,   X86::RSI,
    X86::R12,   X86::R13,   X86::R14,   X86::R15,
    X86::XMM6,  X86::XMM7,  X86::XMM8,  X86::XMM9,
    X86::XMM10, X86::XMM11, X86::XMM12, X86::XMM13,
    X86::XMM14, X86::XMM15, 0
  };

  if (Is64Bit) {
    if (IsWin64)
      return CalleeSavedRegsWin64;
    else
      return (callsEHReturn ? CalleeSavedRegs64EHRet : CalleeSavedRegs64Bit);
  } else {
    return (callsEHReturn ? CalleeSavedRegs32EHRet : CalleeSavedRegs32Bit);
  }
}

const TargetRegisterClass* const*
X86RegisterInfo::getCalleeSavedRegClasses(const MachineFunction *MF) const {
  bool callsEHReturn = false;

  if (MF) {
    const MachineFrameInfo *MFI = MF->getFrameInfo();
    const MachineModuleInfo *MMI = MFI->getMachineModuleInfo();
    callsEHReturn = (MMI ? MMI->callsEHReturn() : false);
  }

  static const TargetRegisterClass * const CalleeSavedRegClasses32Bit[] = {
    &X86::GR32RegClass, &X86::GR32RegClass,
    &X86::GR32RegClass, &X86::GR32RegClass,  0
  };
  static const TargetRegisterClass * const CalleeSavedRegClasses32EHRet[] = {
    &X86::GR32RegClass, &X86::GR32RegClass,
    &X86::GR32RegClass, &X86::GR32RegClass,
    &X86::GR32RegClass, &X86::GR32RegClass,  0
  };
  static const TargetRegisterClass * const CalleeSavedRegClasses64Bit[] = {
    &X86::GR64RegClass, &X86::GR64RegClass,
    &X86::GR64RegClass, &X86::GR64RegClass,
    &X86::GR64RegClass, &X86::GR64RegClass, 0
  };
  static const TargetRegisterClass * const CalleeSavedRegClasses64EHRet[] = {
    &X86::GR64RegClass, &X86::GR64RegClass,
    &X86::GR64RegClass, &X86::GR64RegClass,
    &X86::GR64RegClass, &X86::GR64RegClass,
    &X86::GR64RegClass, &X86::GR64RegClass, 0
  };
  static const TargetRegisterClass * const CalleeSavedRegClassesWin64[] = {
    &X86::GR64RegClass,  &X86::GR64RegClass,
    &X86::GR64RegClass,  &X86::GR64RegClass,
    &X86::GR64RegClass,  &X86::GR64RegClass,
    &X86::GR64RegClass,  &X86::GR64RegClass,
    &X86::VR128RegClass, &X86::VR128RegClass,
    &X86::VR128RegClass, &X86::VR128RegClass,
    &X86::VR128RegClass, &X86::VR128RegClass,
    &X86::VR128RegClass, &X86::VR128RegClass,
    &X86::VR128RegClass, &X86::VR128RegClass, 0
  };

  if (Is64Bit) {
    if (IsWin64)
      return CalleeSavedRegClassesWin64;
    else
      return (callsEHReturn ?
              CalleeSavedRegClasses64EHRet : CalleeSavedRegClasses64Bit);
  } else {
    return (callsEHReturn ?
            CalleeSavedRegClasses32EHRet : CalleeSavedRegClasses32Bit);
  }
}

BitVector X86RegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  // Set the stack-pointer register and its aliases as reserved.
  Reserved.set(X86::RSP);
  Reserved.set(X86::ESP);
  Reserved.set(X86::SP);
  Reserved.set(X86::SPL);

  // Set the frame-pointer register and its aliases as reserved if needed.
  if (hasFP(MF)) {
    Reserved.set(X86::RBP);
    Reserved.set(X86::EBP);
    Reserved.set(X86::BP);
    Reserved.set(X86::BPL);
  }

  // Mark the x87 stack registers as reserved, since they don't behave normally
  // with respect to liveness. We don't fully model the effects of x87 stack
  // pushes and pops after stackification.
  Reserved.set(X86::ST0);
  Reserved.set(X86::ST1);
  Reserved.set(X86::ST2);
  Reserved.set(X86::ST3);
  Reserved.set(X86::ST4);
  Reserved.set(X86::ST5);
  Reserved.set(X86::ST6);
  Reserved.set(X86::ST7);
  return Reserved;
}

//===----------------------------------------------------------------------===//
// Stack Frame Processing methods
//===----------------------------------------------------------------------===//

static unsigned calculateMaxStackAlignment(const MachineFrameInfo *FFI) {
  unsigned MaxAlign = 0;

  for (int i = FFI->getObjectIndexBegin(),
         e = FFI->getObjectIndexEnd(); i != e; ++i) {
    if (FFI->isDeadObjectIndex(i))
      continue;

    unsigned Align = FFI->getObjectAlignment(i);
    MaxAlign = std::max(MaxAlign, Align);
  }

  return MaxAlign;
}

/// hasFP - Return true if the specified function should have a dedicated frame
/// pointer register.  This is true if the function has variable sized allocas
/// or if frame pointer elimination is disabled.
bool X86RegisterInfo::hasFP(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  const MachineModuleInfo *MMI = MFI->getMachineModuleInfo();

  return (NoFramePointerElim ||
          needsStackRealignment(MF) ||
          MFI->hasVarSizedObjects() ||
          MFI->isFrameAddressTaken() ||
          MF.getInfo<X86MachineFunctionInfo>()->getForceFramePointer() ||
          (MMI && MMI->callsUnwindInit()));
}

bool X86RegisterInfo::needsStackRealignment(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();

  // FIXME: Currently we don't support stack realignment for functions with
  //        variable-sized allocas
  return (RealignStack &&
          (MFI->getMaxAlignment() > StackAlign &&
           !MFI->hasVarSizedObjects()));
}

bool X86RegisterInfo::hasReservedCallFrame(MachineFunction &MF) const {
  return !MF.getFrameInfo()->hasVarSizedObjects();
}

bool X86RegisterInfo::hasReservedSpillSlot(MachineFunction &MF, unsigned Reg,
                                           int &FrameIdx) const {
  if (Reg == FramePtr && hasFP(MF)) {
    FrameIdx = MF.getFrameInfo()->getObjectIndexBegin();
    return true;
  }
  return false;
}

int
X86RegisterInfo::getFrameIndexOffset(MachineFunction &MF, int FI) const {
  const TargetFrameInfo &TFI = *MF.getTarget().getFrameInfo();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  int Offset = MFI->getObjectOffset(FI) - TFI.getOffsetOfLocalArea();
  uint64_t StackSize = MFI->getStackSize();

  if (needsStackRealignment(MF)) {
    if (FI < 0) {
      // Skip the saved EBP.
      Offset += SlotSize;
    } else {
      unsigned Align = MFI->getObjectAlignment(FI);
      assert( (-(Offset + StackSize)) % Align == 0);
      Align = 0;
      return Offset + StackSize;
    }
    // FIXME: Support tail calls
  } else {
    if (!hasFP(MF))
      return Offset + StackSize;

    // Skip the saved EBP.
    Offset += SlotSize;

    // Skip the RETADDR move area
    X86MachineFunctionInfo *X86FI = MF.getInfo<X86MachineFunctionInfo>();
    int TailCallReturnAddrDelta = X86FI->getTCReturnAddrDelta();
    if (TailCallReturnAddrDelta < 0)
      Offset -= TailCallReturnAddrDelta;
  }

  return Offset;
}

void X86RegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  if (!hasReservedCallFrame(MF)) {
    // If the stack pointer can be changed after prologue, turn the
    // adjcallstackup instruction into a 'sub ESP, <amt>' and the
    // adjcallstackdown instruction into 'add ESP, <amt>'
    // TODO: consider using push / pop instead of sub + store / add
    MachineInstr *Old = I;
    uint64_t Amount = Old->getOperand(0).getImm();
    if (Amount != 0) {
      // We need to keep the stack aligned properly.  To do this, we round the
      // amount of space needed for the outgoing arguments up to the next
      // alignment boundary.
      Amount = (Amount + StackAlign - 1) / StackAlign * StackAlign;

      MachineInstr *New = 0;
      if (Old->getOpcode() == getCallFrameSetupOpcode()) {
        New = BuildMI(MF, Old->getDebugLoc(),
                      TII.get(Is64Bit ? X86::SUB64ri32 : X86::SUB32ri),
                      StackPtr)
          .addReg(StackPtr)
          .addImm(Amount);
      } else {
        assert(Old->getOpcode() == getCallFrameDestroyOpcode());

        // Factor out the amount the callee already popped.
        uint64_t CalleeAmt = Old->getOperand(1).getImm();
        Amount -= CalleeAmt;
  
      if (Amount) {
          unsigned Opc = (Amount < 128) ?
            (Is64Bit ? X86::ADD64ri8 : X86::ADD32ri8) :
            (Is64Bit ? X86::ADD64ri32 : X86::ADD32ri);
          New = BuildMI(MF, Old->getDebugLoc(), TII.get(Opc), StackPtr)
            .addReg(StackPtr)
            .addImm(Amount);
        }
      }

      if (New) {
        // The EFLAGS implicit def is dead.
        New->getOperand(3).setIsDead();

        // Replace the pseudo instruction with a new instruction.
        MBB.insert(I, New);
      }
    }
  } else if (I->getOpcode() == getCallFrameDestroyOpcode()) {
    // If we are performing frame pointer elimination and if the callee pops
    // something off the stack pointer, add it back.  We do this until we have
    // more advanced stack pointer tracking ability.
    if (uint64_t CalleeAmt = I->getOperand(1).getImm()) {
      unsigned Opc = (CalleeAmt < 128) ?
        (Is64Bit ? X86::SUB64ri8 : X86::SUB32ri8) :
        (Is64Bit ? X86::SUB64ri32 : X86::SUB32ri);
      MachineInstr *Old = I;
      MachineInstr *New =
        BuildMI(MF, Old->getDebugLoc(), TII.get(Opc), 
                StackPtr)
          .addReg(StackPtr)
          .addImm(CalleeAmt);

      // The EFLAGS implicit def is dead.
      New->getOperand(3).setIsDead();
      MBB.insert(I, New);
    }
  }

  MBB.erase(I);
}

void X86RegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                          int SPAdj, RegScavenger *RS) const{
  assert(SPAdj == 0 && "Unexpected");

  unsigned i = 0;
  MachineInstr &MI = *II;
  MachineFunction &MF = *MI.getParent()->getParent();

  while (!MI.getOperand(i).isFI()) {
    ++i;
    assert(i < MI.getNumOperands() && "Instr doesn't have FrameIndex operand!");
  }

  int FrameIndex = MI.getOperand(i).getIndex();
  unsigned BasePtr;

  if (needsStackRealignment(MF))
    BasePtr = (FrameIndex < 0 ? FramePtr : StackPtr);
  else
    BasePtr = (hasFP(MF) ? FramePtr : StackPtr);

  // This must be part of a four operand memory reference.  Replace the
  // FrameIndex with base register with EBP.  Add an offset to the offset.
  MI.getOperand(i).ChangeToRegister(BasePtr, false);

  // Now add the frame object offset to the offset from EBP.
  if (MI.getOperand(i+3).isImm()) {
    // Offset is a 32-bit integer.
    int Offset = getFrameIndexOffset(MF, FrameIndex) +
      (int)(MI.getOperand(i + 3).getImm());
  
     MI.getOperand(i + 3).ChangeToImmediate(Offset);
  } else {
    // Offset is symbolic. This is extremely rare.
    uint64_t Offset = getFrameIndexOffset(MF, FrameIndex) +
                      (uint64_t)MI.getOperand(i+3).getOffset();
    MI.getOperand(i+3).setOffset(Offset);
  }
}

void
X86RegisterInfo::processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                                      RegScavenger *RS) const {
  MachineFrameInfo *MFI = MF.getFrameInfo();

  // Calculate and set max stack object alignment early, so we can decide
  // whether we will need stack realignment (and thus FP).
  unsigned MaxAlign = std::max(MFI->getMaxAlignment(),
                               calculateMaxStackAlignment(MFI));

  MFI->setMaxAlignment(MaxAlign);

  X86MachineFunctionInfo *X86FI = MF.getInfo<X86MachineFunctionInfo>();
  int32_t TailCallReturnAddrDelta = X86FI->getTCReturnAddrDelta();

  if (TailCallReturnAddrDelta < 0) {
    // create RETURNADDR area
    //   arg
    //   arg
    //   RETADDR
    //   { ...
    //     RETADDR area
    //     ...
    //   }
    //   [EBP]
    MFI->CreateFixedObject(-TailCallReturnAddrDelta,
                           (-1*SlotSize)+TailCallReturnAddrDelta);
  }

  if (hasFP(MF)) {
    assert((TailCallReturnAddrDelta <= 0) &&
           "The Delta should always be zero or negative");
    const TargetFrameInfo &TFI = *MF.getTarget().getFrameInfo();

    // Create a frame entry for the EBP register that must be saved.
    int FrameIdx = MFI->CreateFixedObject(SlotSize,
                                          -(int)SlotSize +
                                          TFI.getOffsetOfLocalArea() +
                                          TailCallReturnAddrDelta);
    assert(FrameIdx == MFI->getObjectIndexBegin() &&
           "Slot for EBP register must be last in order to be found!");
    FrameIdx = 0;
  }
}

/// emitSPUpdate - Emit a series of instructions to increment / decrement the
/// stack pointer by a constant value.
static
void emitSPUpdate(MachineBasicBlock &MBB, MachineBasicBlock::iterator &MBBI,
                  unsigned StackPtr, int64_t NumBytes, bool Is64Bit,
                  const TargetInstrInfo &TII) {
  bool isSub = NumBytes < 0;
  uint64_t Offset = isSub ? -NumBytes : NumBytes;
  unsigned Opc = isSub
    ? ((Offset < 128) ?
       (Is64Bit ? X86::SUB64ri8 : X86::SUB32ri8) :
       (Is64Bit ? X86::SUB64ri32 : X86::SUB32ri))
    : ((Offset < 128) ?
       (Is64Bit ? X86::ADD64ri8 : X86::ADD32ri8) :
       (Is64Bit ? X86::ADD64ri32 : X86::ADD32ri));
  uint64_t Chunk = (1LL << 31) - 1;
  DebugLoc DL = (MBBI != MBB.end() ? MBBI->getDebugLoc() :
                 DebugLoc::getUnknownLoc());

  while (Offset) {
    uint64_t ThisVal = (Offset > Chunk) ? Chunk : Offset;
    MachineInstr *MI =
      BuildMI(MBB, MBBI, DL, TII.get(Opc), StackPtr)
        .addReg(StackPtr)
        .addImm(ThisVal);
    MI->getOperand(3).setIsDead(); // The EFLAGS implicit def is dead.
    Offset -= ThisVal;
  }
}

/// mergeSPUpdatesUp - Merge two stack-manipulating instructions upper iterator.
static
void mergeSPUpdatesUp(MachineBasicBlock &MBB, MachineBasicBlock::iterator &MBBI,
                      unsigned StackPtr, uint64_t *NumBytes = NULL) {
  if (MBBI == MBB.begin()) return;

  MachineBasicBlock::iterator PI = prior(MBBI);
  unsigned Opc = PI->getOpcode();
  if ((Opc == X86::ADD64ri32 || Opc == X86::ADD64ri8 ||
       Opc == X86::ADD32ri || Opc == X86::ADD32ri8) &&
      PI->getOperand(0).getReg() == StackPtr) {
    if (NumBytes)
      *NumBytes += PI->getOperand(2).getImm();
    MBB.erase(PI);
  } else if ((Opc == X86::SUB64ri32 || Opc == X86::SUB64ri8 ||
              Opc == X86::SUB32ri || Opc == X86::SUB32ri8) &&
             PI->getOperand(0).getReg() == StackPtr) {
    if (NumBytes)
      *NumBytes -= PI->getOperand(2).getImm();
    MBB.erase(PI);
  }
}

/// mergeSPUpdatesUp - Merge two stack-manipulating instructions lower iterator.
static
void mergeSPUpdatesDown(MachineBasicBlock &MBB,
                        MachineBasicBlock::iterator &MBBI,
                        unsigned StackPtr, uint64_t *NumBytes = NULL) {
  // FIXME: THIS ISN'T RUN!!!
  return;

  if (MBBI == MBB.end()) return;

  MachineBasicBlock::iterator NI = next(MBBI);
  if (NI == MBB.end()) return;

  unsigned Opc = NI->getOpcode();
  if ((Opc == X86::ADD64ri32 || Opc == X86::ADD64ri8 ||
       Opc == X86::ADD32ri || Opc == X86::ADD32ri8) &&
      NI->getOperand(0).getReg() == StackPtr) {
    if (NumBytes)
      *NumBytes -= NI->getOperand(2).getImm();
    MBB.erase(NI);
    MBBI = NI;
  } else if ((Opc == X86::SUB64ri32 || Opc == X86::SUB64ri8 ||
              Opc == X86::SUB32ri || Opc == X86::SUB32ri8) &&
             NI->getOperand(0).getReg() == StackPtr) {
    if (NumBytes)
      *NumBytes += NI->getOperand(2).getImm();
    MBB.erase(NI);
    MBBI = NI;
  }
}

/// mergeSPUpdates - Checks the instruction before/after the passed
/// instruction. If it is an ADD/SUB instruction it is deleted argument and the
/// stack adjustment is returned as a positive value for ADD and a negative for
/// SUB.
static int mergeSPUpdates(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator &MBBI,
                           unsigned StackPtr,
                           bool doMergeWithPrevious) {
  if ((doMergeWithPrevious && MBBI == MBB.begin()) ||
      (!doMergeWithPrevious && MBBI == MBB.end()))
    return 0;

  MachineBasicBlock::iterator PI = doMergeWithPrevious ? prior(MBBI) : MBBI;
  MachineBasicBlock::iterator NI = doMergeWithPrevious ? 0 : next(MBBI);
  unsigned Opc = PI->getOpcode();
  int Offset = 0;

  if ((Opc == X86::ADD64ri32 || Opc == X86::ADD64ri8 ||
       Opc == X86::ADD32ri || Opc == X86::ADD32ri8) &&
      PI->getOperand(0).getReg() == StackPtr){
    Offset += PI->getOperand(2).getImm();
    MBB.erase(PI);
    if (!doMergeWithPrevious) MBBI = NI;
  } else if ((Opc == X86::SUB64ri32 || Opc == X86::SUB64ri8 ||
              Opc == X86::SUB32ri || Opc == X86::SUB32ri8) &&
             PI->getOperand(0).getReg() == StackPtr) {
    Offset -= PI->getOperand(2).getImm();
    MBB.erase(PI);
    if (!doMergeWithPrevious) MBBI = NI;
  }

  return Offset;
}

void X86RegisterInfo::emitCalleeSavedFrameMoves(MachineFunction &MF,
                                                unsigned LabelId,
                                                unsigned FramePtr) const {
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineModuleInfo *MMI = MFI->getMachineModuleInfo();
  if (!MMI) return;

  // Add callee saved registers to move list.
  const std::vector<CalleeSavedInfo> &CSI = MFI->getCalleeSavedInfo();
  if (CSI.empty()) return;

  std::vector<MachineMove> &Moves = MMI->getFrameMoves();
  const TargetData *TD = MF.getTarget().getTargetData();
  bool HasFP = hasFP(MF);

  // Calculate amount of bytes used for return address storing.
  int stackGrowth =
    (MF.getTarget().getFrameInfo()->getStackGrowthDirection() ==
     TargetFrameInfo::StackGrowsUp ?
     TD->getPointerSize() : -TD->getPointerSize());

  // FIXME: This is dirty hack. The code itself is pretty mess right now.
  // It should be rewritten from scratch and generalized sometimes.

  // Determine maximum offset (minumum due to stack growth).
  int64_t MaxOffset = 0;
  for (std::vector<CalleeSavedInfo>::const_iterator
         I = CSI.begin(), E = CSI.end(); I != E; ++I)
    MaxOffset = std::min(MaxOffset,
                         MFI->getObjectOffset(I->getFrameIdx()));

  // Calculate offsets.
  int64_t saveAreaOffset = (HasFP ? 3 : 2) * stackGrowth;
  for (std::vector<CalleeSavedInfo>::const_iterator
         I = CSI.begin(), E = CSI.end(); I != E; ++I) {
    int64_t Offset = MFI->getObjectOffset(I->getFrameIdx());
    unsigned Reg = I->getReg();
    Offset = MaxOffset - Offset + saveAreaOffset;

    // Don't output a new machine move if we're re-saving the frame
    // pointer. This happens when the PrologEpilogInserter has inserted an extra
    // "PUSH" of the frame pointer -- the "emitPrologue" method automatically
    // generates one when frame pointers are used. If we generate a "machine
    // move" for this extra "PUSH", the linker will lose track of the fact that
    // the frame pointer should have the value of the first "PUSH" when it's
    // trying to unwind.
    // 
    // FIXME: This looks inelegant. It's possibly correct, but it's covering up
    //        another bug. I.e., one where we generate a prolog like this:
    //
    //          pushl  %ebp
    //          movl   %esp, %ebp
    //          pushl  %ebp
    //          pushl  %esi
    //           ...
    //
    //        The immediate re-push of EBP is unnecessary. At the least, it's an
    //        optimization bug. EBP can be used as a scratch register in certain
    //        cases, but probably not when we have a frame pointer.
    if (HasFP && FramePtr == Reg)
      continue;

    MachineLocation CSDst(MachineLocation::VirtualFP, Offset);
    MachineLocation CSSrc(Reg);
    Moves.push_back(MachineMove(LabelId, CSDst, CSSrc));
  }
}

/// emitPrologue - Push callee-saved registers onto the stack, which
/// automatically adjust the stack pointer. Adjust the stack pointer to allocate
/// space for local variables. Also emit labels used by the exception handler to
/// generate the exception handling frames.
void X86RegisterInfo::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front(); // Prologue goes in entry BB.
  MachineBasicBlock::iterator MBBI = MBB.begin();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  const Function *Fn = MF.getFunction();
  const X86Subtarget *Subtarget = &MF.getTarget().getSubtarget<X86Subtarget>();
  MachineModuleInfo *MMI = MFI->getMachineModuleInfo();
  X86MachineFunctionInfo *X86FI = MF.getInfo<X86MachineFunctionInfo>();
  bool needsFrameMoves = (MMI && MMI->hasDebugInfo()) ||
                          !Fn->doesNotThrow() || UnwindTablesMandatory;
  uint64_t MaxAlign  = MFI->getMaxAlignment(); // Desired stack alignment.
  uint64_t StackSize = MFI->getStackSize();    // Number of bytes to allocate.
  bool HasFP = hasFP(MF);
  DebugLoc DL;

  // Add RETADDR move area to callee saved frame size.
  int TailCallReturnAddrDelta = X86FI->getTCReturnAddrDelta();
  if (TailCallReturnAddrDelta < 0)
    X86FI->setCalleeSavedFrameSize(
      X86FI->getCalleeSavedFrameSize() - TailCallReturnAddrDelta);

  // If this is x86-64 and the Red Zone is not disabled, if we are a leaf
  // function, and use up to 128 bytes of stack space, don't have a frame
  // pointer, calls, or dynamic alloca then we do not need to adjust the
  // stack pointer (we fit in the Red Zone).
  if (Is64Bit && !Fn->hasFnAttr(Attribute::NoRedZone) &&
      !needsStackRealignment(MF) &&
      !MFI->hasVarSizedObjects() &&                // No dynamic alloca.
      !MFI->hasCalls() &&                          // No calls.
      !Subtarget->isTargetWin64()) {               // Win64 has no Red Zone
    uint64_t MinSize = X86FI->getCalleeSavedFrameSize();
    if (HasFP) MinSize += SlotSize;
    StackSize = std::max(MinSize, StackSize > 128 ? StackSize - 128 : 0);
    MFI->setStackSize(StackSize);
  } else if (Subtarget->isTargetWin64()) {
    // We need to always allocate 32 bytes as register spill area.
    // FIXME: We might reuse these 32 bytes for leaf functions.
    StackSize += 32;
    MFI->setStackSize(StackSize);
  }

  // Insert stack pointer adjustment for later moving of return addr.  Only
  // applies to tail call optimized functions where the callee argument stack
  // size is bigger than the callers.
  if (TailCallReturnAddrDelta < 0) {
    MachineInstr *MI =
      BuildMI(MBB, MBBI, DL, TII.get(Is64Bit? X86::SUB64ri32 : X86::SUB32ri),
              StackPtr)
        .addReg(StackPtr)
        .addImm(-TailCallReturnAddrDelta);
    MI->getOperand(3).setIsDead(); // The EFLAGS implicit def is dead.
  }

  // Mapping for machine moves:
  //
  //   DST: VirtualFP AND
  //        SRC: VirtualFP              => DW_CFA_def_cfa_offset
  //        ELSE                        => DW_CFA_def_cfa
  //
  //   SRC: VirtualFP AND
  //        DST: Register               => DW_CFA_def_cfa_register
  //
  //   ELSE
  //        OFFSET < 0                  => DW_CFA_offset_extended_sf
  //        REG < 64                    => DW_CFA_offset + Reg
  //        ELSE                        => DW_CFA_offset_extended

  std::vector<MachineMove> &Moves = MMI->getFrameMoves();
  const TargetData *TD = MF.getTarget().getTargetData();
  uint64_t NumBytes = 0;
  int stackGrowth =
    (MF.getTarget().getFrameInfo()->getStackGrowthDirection() ==
     TargetFrameInfo::StackGrowsUp ?
       TD->getPointerSize() : -TD->getPointerSize());

  if (HasFP) {
    // Calculate required stack adjustment.
    uint64_t FrameSize = StackSize - SlotSize;
    if (needsStackRealignment(MF))
      FrameSize = (FrameSize + MaxAlign - 1) / MaxAlign * MaxAlign;

    NumBytes = FrameSize - X86FI->getCalleeSavedFrameSize();

    // Get the offset of the stack slot for the EBP register, which is
    // guaranteed to be the last slot by processFunctionBeforeFrameFinalized.
    // Update the frame offset adjustment.
    MFI->setOffsetAdjustment(-NumBytes);

    // Save EBP/RBP into the appropriate stack slot.
    BuildMI(MBB, MBBI, DL, TII.get(Is64Bit ? X86::PUSH64r : X86::PUSH32r))
      .addReg(FramePtr, RegState::Kill);

    if (needsFrameMoves) {
      // Mark the place where EBP/RBP was saved.
      unsigned FrameLabelId = MMI->NextLabelID();
      BuildMI(MBB, MBBI, DL, TII.get(X86::DBG_LABEL)).addImm(FrameLabelId);

      // Define the current CFA rule to use the provided offset.
      if (StackSize) {
        MachineLocation SPDst(MachineLocation::VirtualFP);
        MachineLocation SPSrc(MachineLocation::VirtualFP, 2 * stackGrowth);
        Moves.push_back(MachineMove(FrameLabelId, SPDst, SPSrc));
      } else {
        // FIXME: Verify & implement for FP
        MachineLocation SPDst(StackPtr);
        MachineLocation SPSrc(StackPtr, stackGrowth);
        Moves.push_back(MachineMove(FrameLabelId, SPDst, SPSrc));
      }

      // Change the rule for the FramePtr to be an "offset" rule.
      MachineLocation FPDst(MachineLocation::VirtualFP,
                            2 * stackGrowth);
      MachineLocation FPSrc(FramePtr);
      Moves.push_back(MachineMove(FrameLabelId, FPDst, FPSrc));
    }

    // Update EBP with the new base value...
    BuildMI(MBB, MBBI, DL,
            TII.get(Is64Bit ? X86::MOV64rr : X86::MOV32rr), FramePtr)
        .addReg(StackPtr);

    if (needsFrameMoves) {
      // Mark effective beginning of when frame pointer becomes valid.
      unsigned FrameLabelId = MMI->NextLabelID();
      BuildMI(MBB, MBBI, DL, TII.get(X86::DBG_LABEL)).addImm(FrameLabelId);

      // Define the current CFA to use the EBP/RBP register.
      MachineLocation FPDst(FramePtr);
      MachineLocation FPSrc(MachineLocation::VirtualFP);
      Moves.push_back(MachineMove(FrameLabelId, FPDst, FPSrc));
    }

    // Mark the FramePtr as live-in in every block except the entry.
    for (MachineFunction::iterator I = next(MF.begin()), E = MF.end();
         I != E; ++I)
      I->addLiveIn(FramePtr);

    // Realign stack
    if (needsStackRealignment(MF)) {
      MachineInstr *MI =
        BuildMI(MBB, MBBI, DL,
                TII.get(Is64Bit ? X86::AND64ri32 : X86::AND32ri),
                StackPtr).addReg(StackPtr).addImm(-MaxAlign);

      // The EFLAGS implicit def is dead.
      MI->getOperand(3).setIsDead();
    }
  } else {
    NumBytes = StackSize - X86FI->getCalleeSavedFrameSize();
  }

  // Skip the callee-saved push instructions.
  bool PushedRegs = false;
  int StackOffset = 2 * stackGrowth;

  while (MBBI != MBB.end() &&
         (MBBI->getOpcode() == X86::PUSH32r ||
          MBBI->getOpcode() == X86::PUSH64r)) {
    PushedRegs = true;
    ++MBBI;

    if (!HasFP && needsFrameMoves) {
      // Mark callee-saved push instruction.
      unsigned LabelId = MMI->NextLabelID();
      BuildMI(MBB, MBBI, DL, TII.get(X86::DBG_LABEL)).addImm(LabelId);

      // Define the current CFA rule to use the provided offset.
      unsigned Ptr = StackSize ?
        MachineLocation::VirtualFP : StackPtr;
      MachineLocation SPDst(Ptr);
      MachineLocation SPSrc(Ptr, StackOffset);
      Moves.push_back(MachineMove(LabelId, SPDst, SPSrc));
      StackOffset += stackGrowth;
    }
  }

  if (MBBI != MBB.end())
    DL = MBBI->getDebugLoc();

  // Adjust stack pointer: ESP -= numbytes.
  if (NumBytes >= 4096 && Subtarget->isTargetCygMing()) {
    // Check, whether EAX is livein for this function.
    bool isEAXAlive = false;
    for (MachineRegisterInfo::livein_iterator
           II = MF.getRegInfo().livein_begin(),
           EE = MF.getRegInfo().livein_end(); (II != EE) && !isEAXAlive; ++II) {
      unsigned Reg = II->first;
      isEAXAlive = (Reg == X86::EAX || Reg == X86::AX ||
                    Reg == X86::AH || Reg == X86::AL);
    }

    // Function prologue calls _alloca to probe the stack when allocating more
    // than 4k bytes in one go. Touching the stack at 4K increments is necessary
    // to ensure that the guard pages used by the OS virtual memory manager are
    // allocated in correct sequence.
    if (!isEAXAlive) {
      BuildMI(MBB, MBBI, DL, TII.get(X86::MOV32ri), X86::EAX)
        .addImm(NumBytes);
      BuildMI(MBB, MBBI, DL, TII.get(X86::CALLpcrel32))
        .addExternalSymbol("_alloca");
    } else {
      // Save EAX
      BuildMI(MBB, MBBI, DL, TII.get(X86::PUSH32r))
        .addReg(X86::EAX, RegState::Kill);

      // Allocate NumBytes-4 bytes on stack. We'll also use 4 already
      // allocated bytes for EAX.
      BuildMI(MBB, MBBI, DL, TII.get(X86::MOV32ri), X86::EAX)
        .addImm(NumBytes - 4);
      BuildMI(MBB, MBBI, DL, TII.get(X86::CALLpcrel32))
        .addExternalSymbol("_alloca");

      // Restore EAX
      MachineInstr *MI = addRegOffset(BuildMI(MF, DL, TII.get(X86::MOV32rm),
                                              X86::EAX),
                                      StackPtr, false, NumBytes - 4);
      MBB.insert(MBBI, MI);
    }
  } else if (NumBytes) {
    // If there is an SUB32ri of ESP immediately before this instruction, merge
    // the two. This can be the case when tail call elimination is enabled and
    // the callee has more arguments then the caller.
    NumBytes -= mergeSPUpdates(MBB, MBBI, StackPtr, true);

    // If there is an ADD32ri or SUB32ri of ESP immediately after this
    // instruction, merge the two instructions.
    mergeSPUpdatesDown(MBB, MBBI, StackPtr, &NumBytes);

    if (NumBytes)
      emitSPUpdate(MBB, MBBI, StackPtr, -(int64_t)NumBytes, Is64Bit, TII);
  }

  if ((NumBytes || PushedRegs) && needsFrameMoves) {
    // Mark end of stack pointer adjustment.
    unsigned LabelId = MMI->NextLabelID();
    BuildMI(MBB, MBBI, DL, TII.get(X86::DBG_LABEL)).addImm(LabelId);

    if (!HasFP && NumBytes) {
      // Define the current CFA rule to use the provided offset.
      if (StackSize) {
        MachineLocation SPDst(MachineLocation::VirtualFP);
        MachineLocation SPSrc(MachineLocation::VirtualFP,
                              -StackSize + stackGrowth);
        Moves.push_back(MachineMove(LabelId, SPDst, SPSrc));
      } else {
        // FIXME: Verify & implement for FP
        MachineLocation SPDst(StackPtr);
        MachineLocation SPSrc(StackPtr, stackGrowth);
        Moves.push_back(MachineMove(LabelId, SPDst, SPSrc));
      }
    }

    // Emit DWARF info specifying the offsets of the callee-saved registers.
    if (PushedRegs)
      emitCalleeSavedFrameMoves(MF, LabelId, HasFP ? FramePtr : StackPtr);
  }
}

void X86RegisterInfo::emitEpilogue(MachineFunction &MF,
                                   MachineBasicBlock &MBB) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  X86MachineFunctionInfo *X86FI = MF.getInfo<X86MachineFunctionInfo>();
  MachineBasicBlock::iterator MBBI = prior(MBB.end());
  unsigned RetOpcode = MBBI->getOpcode();
  DebugLoc DL = MBBI->getDebugLoc();

  switch (RetOpcode) {
  default:
    llvm_unreachable("Can only insert epilog into returning blocks");
  case X86::RET:
  case X86::RETI:
  case X86::TCRETURNdi:
  case X86::TCRETURNri:
  case X86::TCRETURNri64:
  case X86::TCRETURNdi64:
  case X86::EH_RETURN:
  case X86::EH_RETURN64:
  case X86::TAILJMPd:
  case X86::TAILJMPr:
  case X86::TAILJMPm:
    break;  // These are ok
  }

  // Get the number of bytes to allocate from the FrameInfo.
  uint64_t StackSize = MFI->getStackSize();
  uint64_t MaxAlign  = MFI->getMaxAlignment();
  unsigned CSSize = X86FI->getCalleeSavedFrameSize();
  uint64_t NumBytes = 0;

  if (hasFP(MF)) {
    // Calculate required stack adjustment.
    uint64_t FrameSize = StackSize - SlotSize;
    if (needsStackRealignment(MF))
      FrameSize = (FrameSize + MaxAlign - 1)/MaxAlign*MaxAlign;

    NumBytes = FrameSize - CSSize;

    // Pop EBP.
    BuildMI(MBB, MBBI, DL,
            TII.get(Is64Bit ? X86::POP64r : X86::POP32r), FramePtr);
  } else {
    NumBytes = StackSize - CSSize;
  }

  // Skip the callee-saved pop instructions.
  MachineBasicBlock::iterator LastCSPop = MBBI;
  while (MBBI != MBB.begin()) {
    MachineBasicBlock::iterator PI = prior(MBBI);
    unsigned Opc = PI->getOpcode();

    if (Opc != X86::POP32r && Opc != X86::POP64r &&
        !PI->getDesc().isTerminator())
      break;

    --MBBI;
  }

  DL = MBBI->getDebugLoc();

  // If there is an ADD32ri or SUB32ri of ESP immediately before this
  // instruction, merge the two instructions.
  if (NumBytes || MFI->hasVarSizedObjects())
    mergeSPUpdatesUp(MBB, MBBI, StackPtr, &NumBytes);

  // If dynamic alloca is used, then reset esp to point to the last callee-saved
  // slot before popping them off! Same applies for the case, when stack was
  // realigned.
  if (needsStackRealignment(MF)) {
    // We cannot use LEA here, because stack pointer was realigned. We need to
    // deallocate local frame back.
    if (CSSize) {
      emitSPUpdate(MBB, MBBI, StackPtr, NumBytes, Is64Bit, TII);
      MBBI = prior(LastCSPop);
    }

    BuildMI(MBB, MBBI, DL,
            TII.get(Is64Bit ? X86::MOV64rr : X86::MOV32rr),
            StackPtr).addReg(FramePtr);
  } else if (MFI->hasVarSizedObjects()) {
    if (CSSize) {
      unsigned Opc = Is64Bit ? X86::LEA64r : X86::LEA32r;
      MachineInstr *MI =
        addLeaRegOffset(BuildMI(MF, DL, TII.get(Opc), StackPtr),
                        FramePtr, false, -CSSize);
      MBB.insert(MBBI, MI);
    } else {
      BuildMI(MBB, MBBI, DL,
              TII.get(Is64Bit ? X86::MOV64rr : X86::MOV32rr), StackPtr)
        .addReg(FramePtr);
    }
  } else if (NumBytes) {
    // Adjust stack pointer back: ESP += numbytes.
    emitSPUpdate(MBB, MBBI, StackPtr, NumBytes, Is64Bit, TII);
  }

  // We're returning from function via eh_return.
  if (RetOpcode == X86::EH_RETURN || RetOpcode == X86::EH_RETURN64) {
    MBBI = prior(MBB.end());
    MachineOperand &DestAddr  = MBBI->getOperand(0);
    assert(DestAddr.isReg() && "Offset should be in register!");
    BuildMI(MBB, MBBI, DL,
            TII.get(Is64Bit ? X86::MOV64rr : X86::MOV32rr),
            StackPtr).addReg(DestAddr.getReg());
  } else if (RetOpcode == X86::TCRETURNri || RetOpcode == X86::TCRETURNdi ||
             RetOpcode== X86::TCRETURNri64 || RetOpcode == X86::TCRETURNdi64) {
    // Tail call return: adjust the stack pointer and jump to callee.
    MBBI = prior(MBB.end());
    MachineOperand &JumpTarget = MBBI->getOperand(0);
    MachineOperand &StackAdjust = MBBI->getOperand(1);
    assert(StackAdjust.isImm() && "Expecting immediate value.");

    // Adjust stack pointer.
    int StackAdj = StackAdjust.getImm();
    int MaxTCDelta = X86FI->getTCReturnAddrDelta();
    int Offset = 0;
    assert(MaxTCDelta <= 0 && "MaxTCDelta should never be positive");

    // Incoporate the retaddr area.
    Offset = StackAdj-MaxTCDelta;
    assert(Offset >= 0 && "Offset should never be negative");

    if (Offset) {
      // Check for possible merge with preceeding ADD instruction.
      Offset += mergeSPUpdates(MBB, MBBI, StackPtr, true);
      emitSPUpdate(MBB, MBBI, StackPtr, Offset, Is64Bit, TII);
    }

    // Jump to label or value in register.
    if (RetOpcode == X86::TCRETURNdi|| RetOpcode == X86::TCRETURNdi64)
      BuildMI(MBB, MBBI, DL, TII.get(X86::TAILJMPd)).
        addGlobalAddress(JumpTarget.getGlobal(), JumpTarget.getOffset());
    else if (RetOpcode== X86::TCRETURNri64)
      BuildMI(MBB, MBBI, DL, TII.get(X86::TAILJMPr64), JumpTarget.getReg());
    else
       BuildMI(MBB, MBBI, DL, TII.get(X86::TAILJMPr), JumpTarget.getReg());

    // Delete the pseudo instruction TCRETURN.
    MBB.erase(MBBI);
  } else if ((RetOpcode == X86::RET || RetOpcode == X86::RETI) &&
             (X86FI->getTCReturnAddrDelta() < 0)) {
    // Add the return addr area delta back since we are not tail calling.
    int delta = -1*X86FI->getTCReturnAddrDelta();
    MBBI = prior(MBB.end());

    // Check for possible merge with preceeding ADD instruction.
    delta += mergeSPUpdates(MBB, MBBI, StackPtr, true);
    emitSPUpdate(MBB, MBBI, StackPtr, delta, Is64Bit, TII);
  }
}

unsigned X86RegisterInfo::getRARegister() const {
  return Is64Bit ? X86::RIP     // Should have dwarf #16.
                 : X86::EIP;    // Should have dwarf #8.
}

unsigned X86RegisterInfo::getFrameRegister(MachineFunction &MF) const {
  return hasFP(MF) ? FramePtr : StackPtr;
}

void
X86RegisterInfo::getInitialFrameState(std::vector<MachineMove> &Moves) const {
  // Calculate amount of bytes used for return address storing
  int stackGrowth = (Is64Bit ? -8 : -4);

  // Initial state of the frame pointer is esp+4.
  MachineLocation Dst(MachineLocation::VirtualFP);
  MachineLocation Src(StackPtr, stackGrowth);
  Moves.push_back(MachineMove(0, Dst, Src));

  // Add return address to move list
  MachineLocation CSDst(StackPtr, stackGrowth);
  MachineLocation CSSrc(getRARegister());
  Moves.push_back(MachineMove(0, CSDst, CSSrc));
}

unsigned X86RegisterInfo::getEHExceptionRegister() const {
  llvm_unreachable("What is the exception register");
  return 0;
}

unsigned X86RegisterInfo::getEHHandlerRegister() const {
  llvm_unreachable("What is the exception handler register");
  return 0;
}

namespace llvm {
unsigned getX86SubSuperRegister(unsigned Reg, EVT VT, bool High) {
  switch (VT.getSimpleVT().SimpleTy) {
  default: return Reg;
  case MVT::i8:
    if (High) {
      switch (Reg) {
      default: return 0;
      case X86::AH: case X86::AL: case X86::AX: case X86::EAX: case X86::RAX:
        return X86::AH;
      case X86::DH: case X86::DL: case X86::DX: case X86::EDX: case X86::RDX:
        return X86::DH;
      case X86::CH: case X86::CL: case X86::CX: case X86::ECX: case X86::RCX:
        return X86::CH;
      case X86::BH: case X86::BL: case X86::BX: case X86::EBX: case X86::RBX:
        return X86::BH;
      }
    } else {
      switch (Reg) {
      default: return 0;
      case X86::AH: case X86::AL: case X86::AX: case X86::EAX: case X86::RAX:
        return X86::AL;
      case X86::DH: case X86::DL: case X86::DX: case X86::EDX: case X86::RDX:
        return X86::DL;
      case X86::CH: case X86::CL: case X86::CX: case X86::ECX: case X86::RCX:
        return X86::CL;
      case X86::BH: case X86::BL: case X86::BX: case X86::EBX: case X86::RBX:
        return X86::BL;
      case X86::SIL: case X86::SI: case X86::ESI: case X86::RSI:
        return X86::SIL;
      case X86::DIL: case X86::DI: case X86::EDI: case X86::RDI:
        return X86::DIL;
      case X86::BPL: case X86::BP: case X86::EBP: case X86::RBP:
        return X86::BPL;
      case X86::SPL: case X86::SP: case X86::ESP: case X86::RSP:
        return X86::SPL;
      case X86::R8B: case X86::R8W: case X86::R8D: case X86::R8:
        return X86::R8B;
      case X86::R9B: case X86::R9W: case X86::R9D: case X86::R9:
        return X86::R9B;
      case X86::R10B: case X86::R10W: case X86::R10D: case X86::R10:
        return X86::R10B;
      case X86::R11B: case X86::R11W: case X86::R11D: case X86::R11:
        return X86::R11B;
      case X86::R12B: case X86::R12W: case X86::R12D: case X86::R12:
        return X86::R12B;
      case X86::R13B: case X86::R13W: case X86::R13D: case X86::R13:
        return X86::R13B;
      case X86::R14B: case X86::R14W: case X86::R14D: case X86::R14:
        return X86::R14B;
      case X86::R15B: case X86::R15W: case X86::R15D: case X86::R15:
        return X86::R15B;
      }
    }
  case MVT::i16:
    switch (Reg) {
    default: return Reg;
    case X86::AH: case X86::AL: case X86::AX: case X86::EAX: case X86::RAX:
      return X86::AX;
    case X86::DH: case X86::DL: case X86::DX: case X86::EDX: case X86::RDX:
      return X86::DX;
    case X86::CH: case X86::CL: case X86::CX: case X86::ECX: case X86::RCX:
      return X86::CX;
    case X86::BH: case X86::BL: case X86::BX: case X86::EBX: case X86::RBX:
      return X86::BX;
    case X86::SIL: case X86::SI: case X86::ESI: case X86::RSI:
      return X86::SI;
    case X86::DIL: case X86::DI: case X86::EDI: case X86::RDI:
      return X86::DI;
    case X86::BPL: case X86::BP: case X86::EBP: case X86::RBP:
      return X86::BP;
    case X86::SPL: case X86::SP: case X86::ESP: case X86::RSP:
      return X86::SP;
    case X86::R8B: case X86::R8W: case X86::R8D: case X86::R8:
      return X86::R8W;
    case X86::R9B: case X86::R9W: case X86::R9D: case X86::R9:
      return X86::R9W;
    case X86::R10B: case X86::R10W: case X86::R10D: case X86::R10:
      return X86::R10W;
    case X86::R11B: case X86::R11W: case X86::R11D: case X86::R11:
      return X86::R11W;
    case X86::R12B: case X86::R12W: case X86::R12D: case X86::R12:
      return X86::R12W;
    case X86::R13B: case X86::R13W: case X86::R13D: case X86::R13:
      return X86::R13W;
    case X86::R14B: case X86::R14W: case X86::R14D: case X86::R14:
      return X86::R14W;
    case X86::R15B: case X86::R15W: case X86::R15D: case X86::R15:
      return X86::R15W;
    }
  case MVT::i32:
    switch (Reg) {
    default: return Reg;
    case X86::AH: case X86::AL: case X86::AX: case X86::EAX: case X86::RAX:
      return X86::EAX;
    case X86::DH: case X86::DL: case X86::DX: case X86::EDX: case X86::RDX:
      return X86::EDX;
    case X86::CH: case X86::CL: case X86::CX: case X86::ECX: case X86::RCX:
      return X86::ECX;
    case X86::BH: case X86::BL: case X86::BX: case X86::EBX: case X86::RBX:
      return X86::EBX;
    case X86::SIL: case X86::SI: case X86::ESI: case X86::RSI:
      return X86::ESI;
    case X86::DIL: case X86::DI: case X86::EDI: case X86::RDI:
      return X86::EDI;
    case X86::BPL: case X86::BP: case X86::EBP: case X86::RBP:
      return X86::EBP;
    case X86::SPL: case X86::SP: case X86::ESP: case X86::RSP:
      return X86::ESP;
    case X86::R8B: case X86::R8W: case X86::R8D: case X86::R8:
      return X86::R8D;
    case X86::R9B: case X86::R9W: case X86::R9D: case X86::R9:
      return X86::R9D;
    case X86::R10B: case X86::R10W: case X86::R10D: case X86::R10:
      return X86::R10D;
    case X86::R11B: case X86::R11W: case X86::R11D: case X86::R11:
      return X86::R11D;
    case X86::R12B: case X86::R12W: case X86::R12D: case X86::R12:
      return X86::R12D;
    case X86::R13B: case X86::R13W: case X86::R13D: case X86::R13:
      return X86::R13D;
    case X86::R14B: case X86::R14W: case X86::R14D: case X86::R14:
      return X86::R14D;
    case X86::R15B: case X86::R15W: case X86::R15D: case X86::R15:
      return X86::R15D;
    }
  case MVT::i64:
    switch (Reg) {
    default: return Reg;
    case X86::AH: case X86::AL: case X86::AX: case X86::EAX: case X86::RAX:
      return X86::RAX;
    case X86::DH: case X86::DL: case X86::DX: case X86::EDX: case X86::RDX:
      return X86::RDX;
    case X86::CH: case X86::CL: case X86::CX: case X86::ECX: case X86::RCX:
      return X86::RCX;
    case X86::BH: case X86::BL: case X86::BX: case X86::EBX: case X86::RBX:
      return X86::RBX;
    case X86::SIL: case X86::SI: case X86::ESI: case X86::RSI:
      return X86::RSI;
    case X86::DIL: case X86::DI: case X86::EDI: case X86::RDI:
      return X86::RDI;
    case X86::BPL: case X86::BP: case X86::EBP: case X86::RBP:
      return X86::RBP;
    case X86::SPL: case X86::SP: case X86::ESP: case X86::RSP:
      return X86::RSP;
    case X86::R8B: case X86::R8W: case X86::R8D: case X86::R8:
      return X86::R8;
    case X86::R9B: case X86::R9W: case X86::R9D: case X86::R9:
      return X86::R9;
    case X86::R10B: case X86::R10W: case X86::R10D: case X86::R10:
      return X86::R10;
    case X86::R11B: case X86::R11W: case X86::R11D: case X86::R11:
      return X86::R11;
    case X86::R12B: case X86::R12W: case X86::R12D: case X86::R12:
      return X86::R12;
    case X86::R13B: case X86::R13W: case X86::R13D: case X86::R13:
      return X86::R13;
    case X86::R14B: case X86::R14W: case X86::R14D: case X86::R14:
      return X86::R14;
    case X86::R15B: case X86::R15W: case X86::R15D: case X86::R15:
      return X86::R15;
    }
  }

  return Reg;
}
}

#include "X86GenRegisterInfo.inc"

namespace {
  struct VISIBILITY_HIDDEN MSAC : public MachineFunctionPass {
    static char ID;
    MSAC() : MachineFunctionPass(&ID) {}

    virtual bool runOnMachineFunction(MachineFunction &MF) {
      MachineFrameInfo *FFI = MF.getFrameInfo();
      MachineRegisterInfo &RI = MF.getRegInfo();

      // Calculate max stack alignment of all already allocated stack objects.
      unsigned MaxAlign = calculateMaxStackAlignment(FFI);

      // Be over-conservative: scan over all vreg defs and find, whether vector
      // registers are used. If yes - there is probability, that vector register
      // will be spilled and thus stack needs to be aligned properly.
      for (unsigned RegNum = TargetRegisterInfo::FirstVirtualRegister;
           RegNum < RI.getLastVirtReg(); ++RegNum)
        MaxAlign = std::max(MaxAlign, RI.getRegClass(RegNum)->getAlignment());

      if (FFI->getMaxAlignment() == MaxAlign)
        return false;

      FFI->setMaxAlignment(MaxAlign);
      return true;
    }

    virtual const char *getPassName() const {
      return "X86 Maximal Stack Alignment Calculator";
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      MachineFunctionPass::getAnalysisUsage(AU);
    }
  };

  char MSAC::ID = 0;
}

FunctionPass*
llvm::createX86MaxStackAlignmentCalculatorPass() { return new MSAC(); }
