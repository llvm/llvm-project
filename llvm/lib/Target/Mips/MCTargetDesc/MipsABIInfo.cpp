//===---- MipsABIInfo.cpp - Information about MIPS ABI's ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MipsABIInfo.h"
#include "Mips.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

// Note: this option is defined here to be visible from libLLVMMipsAsmParser
//       and libLLVMMipsCodeGen
cl::opt<bool>
EmitJalrReloc("mips-jalr-reloc", cl::Hidden,
              cl::desc("MIPS: Emit R_{MICRO}MIPS_JALR relocation with jalr"),
              cl::init(true));
cl::opt<bool>
    NoZeroDivCheck("mno-check-zero-division", cl::Hidden,
                   cl::desc("MIPS: Don't trap on integer division by zero."),
                   cl::init(false));

namespace {
static constexpr MCPhysReg O32IntRegs[] = {Mips::R4, Mips::R5, Mips::R6,
                                           Mips::R7};
static constexpr MCPhysReg NABIIntRegs[] = {Mips::R4, Mips::R5, Mips::R6,
                                            Mips::R7, Mips::R8, Mips::R9,
                                            Mips::R10, Mips::R11};
static constexpr MCPhysReg Mips64IntRegs[] = {
    Mips::R4_64, Mips::R5_64, Mips::R6_64, Mips::R7_64,
    Mips::R8_64, Mips::R9_64, Mips::R10_64, Mips::R11_64};

struct GPR {
  MCPhysReg Reg32;
  MCPhysReg Reg64;
};

static constexpr GPR OABITempRegs[] = {
    {Mips::R8, Mips::R8_64}, {Mips::R9, Mips::R9_64}, {Mips::R10, Mips::R10_64},
    {Mips::R11, Mips::R11_64}, {Mips::R12, Mips::R12_64}, {Mips::R13, Mips::R13_64},
    {Mips::R14, Mips::R14_64}, {Mips::R15, Mips::R15_64}, {Mips::R24, Mips::R24_64},
    {Mips::R25, Mips::R25_64},
};

static constexpr GPR NABITempRegs[] = {
    {Mips::R12, Mips::R12_64},
    {Mips::R13, Mips::R13_64},
    {Mips::R14, Mips::R14_64},
    {Mips::R15, Mips::R15_64},
    {Mips::NoRegister, Mips::NoRegister},
    {Mips::NoRegister, Mips::NoRegister},
    {Mips::NoRegister, Mips::NoRegister},
    {Mips::NoRegister, Mips::NoRegister},
    {Mips::R24, Mips::R24_64},
    {Mips::R25, Mips::R25_64},
};

static constexpr GPR PABITempRegs[] = {
    {Mips::R12, Mips::R12_64},
    {Mips::R13, Mips::R13_64},
    {Mips::R14, Mips::R14_64},
    {Mips::R15, Mips::R15_64},
    {Mips::R2, Mips::R2_64},
    {Mips::R3, Mips::R3_64},
    {Mips::NoRegister, Mips::NoRegister},
    {Mips::NoRegister, Mips::NoRegister},
    {Mips::R24, Mips::R24_64},
    {Mips::R25, Mips::R25_64},
};

static constexpr GPR SavedRegs[] = {
    {Mips::R16, Mips::R16_64}, {Mips::R17, Mips::R17_64}, {Mips::R18, Mips::R18_64},
    {Mips::R19, Mips::R19_64}, {Mips::R20, Mips::R20_64}, {Mips::R21, Mips::R21_64},
    {Mips::R22, Mips::R22_64}, {Mips::R23, Mips::R23_64},
};

static constexpr GPR ReturnRegs[] = {
    {Mips::R2, Mips::R2_64},
    {Mips::R3, Mips::R3_64},
};

MCRegister getReg(ArrayRef<GPR> Regs, unsigned I, bool Is64Bit) {
  assert(I < Regs.size() && "Invalid ABI register index");
  MCRegister Reg = Is64Bit ? Regs[I].Reg64 : Regs[I].Reg32;
  assert(Reg && "Register name is not defined by this ABI");
  return Reg;
}
} // namespace

ArrayRef<MCPhysReg> MipsABIInfo::getArgRegs(bool Is64Bit) const {
  assert(IsKnown() && "Unknown ABI");
  if (Is64Bit)
    return ArrayRef(Mips64IntRegs).take_front(IsO32() ? 4 : 8);
  return IsO32() ? ArrayRef(O32IntRegs) : ArrayRef(NABIIntRegs);
}

ArrayRef<MCPhysReg> MipsABIInfo::GetByValArgRegs() const {
  return getArgRegs(AreGprs64bit());
}

unsigned MipsABIInfo::getRegAltNameIndex() const {
  switch (ThisABI) {
  case ABI::O32:
    return Mips::OABIRegAltName;
  case ABI::N32:
  case ABI::N64:
    return Mips::NABIRegAltName;
  case ABI::Unknown:
    llvm_unreachable("Unknown ABI");
  }
  llvm_unreachable("Unhandled ABI");
}

MCRegister MipsABIInfo::getArgReg(unsigned I, bool Is64Bit) const {
  ArrayRef<MCPhysReg> Regs = getArgRegs(Is64Bit);
  assert(I < Regs.size() && "Invalid argument register");
  return Regs[I];
}

MCRegister MipsABIInfo::getTempReg(unsigned I, bool Is64Bit) const {
  switch (getRegAltNameIndex()) {
  case Mips::OABIRegAltName:
    return getReg(OABITempRegs, I, Is64Bit);
  case Mips::NABIRegAltName:
    return getReg(NABITempRegs, I, Is64Bit);
  case Mips::PABIRegAltName:
    return getReg(PABITempRegs, I, Is64Bit);
  default:
    llvm_unreachable("Unknown register naming convention");
  }
}

MCRegister MipsABIInfo::getSavedReg(unsigned I, bool Is64Bit) const {
  assert(IsKnown() && "Unknown ABI");
  return getReg(SavedRegs, I, Is64Bit);
}

MCRegister MipsABIInfo::getReturnReg(unsigned I, bool Is64Bit) const {
  switch (ThisABI) {
  case ABI::O32:
  case ABI::N32:
  case ABI::N64:
    return getReg(ReturnRegs, I, Is64Bit);
  case ABI::Unknown:
    llvm_unreachable("Unknown ABI");
  }
  llvm_unreachable("Unhandled ABI");
}

ArrayRef<MCPhysReg> MipsABIInfo::getVarArgRegs(bool isGP64bit) const {
  if (IsO32()) {
    if (isGP64bit)
      return ArrayRef(Mips64IntRegs);
    else
      return ArrayRef(O32IntRegs);
  }
  if (IsN32() || IsN64())
    return ArrayRef(Mips64IntRegs);
  llvm_unreachable("Unhandled ABI");
}

unsigned MipsABIInfo::GetCalleeAllocdArgSizeInBytes(CallingConv::ID CC) const {
  if (IsO32())
    return CC != CallingConv::Fast ? 16 : 0;
  if (IsN32() || IsN64())
    return 0;
  llvm_unreachable("Unhandled ABI");
}

MipsABIInfo MipsABIInfo::computeTargetABI(const Triple &TT, StringRef ABIName) {
  if (ABIName.starts_with("o32"))
    return MipsABIInfo::O32();
  if (ABIName.starts_with("n32"))
    return MipsABIInfo::N32();
  if (ABIName.starts_with("n64"))
    return MipsABIInfo::N64();
  if (TT.isABIN32())
    return MipsABIInfo::N32();
  assert(ABIName.empty() && "Unknown ABI option for MIPS");

  if (TT.isMIPS64())
    return MipsABIInfo::N64();
  return MipsABIInfo::O32();
}

unsigned MipsABIInfo::GetStackPtr() const {
  return ArePtrs64bit() ? Mips::R29_64 : Mips::R29;
}

unsigned MipsABIInfo::GetFramePtr() const {
  return ArePtrs64bit() ? Mips::R30_64 : Mips::R30;
}

unsigned MipsABIInfo::GetBasePtr() const { return getSavedRegPtr(7); }

unsigned MipsABIInfo::GetGlobalPtr() const {
  return ArePtrs64bit() ? Mips::R28_64 : Mips::R28;
}

unsigned MipsABIInfo::GetNullPtr() const {
  return ArePtrs64bit() ? Mips::R0_64 : Mips::R0;
}

unsigned MipsABIInfo::GetZeroReg() const {
  return AreGprs64bit() ? Mips::R0_64 : Mips::R0;
}

unsigned MipsABIInfo::GetPtrAdduOp() const {
  return ArePtrs64bit() ? Mips::DADDu : Mips::ADDu;
}

unsigned MipsABIInfo::GetPtrAddiuOp() const {
  return ArePtrs64bit() ? Mips::DADDiu : Mips::ADDiu;
}

unsigned MipsABIInfo::GetPtrSubuOp() const {
  return ArePtrs64bit() ? Mips::DSUBu : Mips::SUBu;
}

unsigned MipsABIInfo::GetPtrAndOp() const {
  return ArePtrs64bit() ? Mips::AND64 : Mips::AND;
}

unsigned MipsABIInfo::GetGPRMoveOp() const {
  return ArePtrs64bit() ? Mips::OR64 : Mips::OR;
}

unsigned MipsABIInfo::GetEhDataReg(unsigned I) const {
  assert(I < 4 && "Invalid EH data register");
  return getArgRegPtr(I);
}
