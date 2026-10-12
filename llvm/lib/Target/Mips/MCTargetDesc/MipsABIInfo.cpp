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
static constexpr MCPhysReg O32IntRegs[] = {Mips::A0, Mips::A1, Mips::A2,
                                           Mips::A3};
static constexpr MCPhysReg NABIIntRegs[] = {Mips::A0, Mips::A1, Mips::A2,
                                            Mips::A3, Mips::T0, Mips::T1,
                                            Mips::T2, Mips::T3};
static constexpr MCPhysReg Mips64IntRegs[] = {
    Mips::A0_64, Mips::A1_64, Mips::A2_64, Mips::A3_64,
    Mips::T0_64, Mips::T1_64, Mips::T2_64, Mips::T3_64};

struct GPR {
  MCPhysReg Reg32;
  MCPhysReg Reg64;
};

static constexpr GPR OABITempRegs[] = {
    {Mips::T0, Mips::T0_64}, {Mips::T1, Mips::T1_64}, {Mips::T2, Mips::T2_64},
    {Mips::T3, Mips::T3_64}, {Mips::T4, Mips::T4_64}, {Mips::T5, Mips::T5_64},
    {Mips::T6, Mips::T6_64}, {Mips::T7, Mips::T7_64}, {Mips::T8, Mips::T8_64},
    {Mips::T9, Mips::T9_64},
};

static constexpr GPR NABITempRegs[] = {
    {Mips::T4, Mips::T4_64},
    {Mips::T5, Mips::T5_64},
    {Mips::T6, Mips::T6_64},
    {Mips::T7, Mips::T7_64},
    {Mips::NoRegister, Mips::NoRegister},
    {Mips::NoRegister, Mips::NoRegister},
    {Mips::NoRegister, Mips::NoRegister},
    {Mips::NoRegister, Mips::NoRegister},
    {Mips::T8, Mips::T8_64},
    {Mips::T9, Mips::T9_64},
};

static constexpr GPR PABITempRegs[] = {
    {Mips::T4, Mips::T4_64},
    {Mips::T5, Mips::T5_64},
    {Mips::T6, Mips::T6_64},
    {Mips::T7, Mips::T7_64},
    {Mips::V0, Mips::V0_64},
    {Mips::V1, Mips::V1_64},
    {Mips::NoRegister, Mips::NoRegister},
    {Mips::NoRegister, Mips::NoRegister},
    {Mips::T8, Mips::T8_64},
    {Mips::T9, Mips::T9_64},
};

static constexpr GPR SavedRegs[] = {
    {Mips::S0, Mips::S0_64}, {Mips::S1, Mips::S1_64}, {Mips::S2, Mips::S2_64},
    {Mips::S3, Mips::S3_64}, {Mips::S4, Mips::S4_64}, {Mips::S5, Mips::S5_64},
    {Mips::S6, Mips::S6_64}, {Mips::S7, Mips::S7_64},
};

static constexpr GPR ReturnRegs[] = {
    {Mips::V0, Mips::V0_64},
    {Mips::V1, Mips::V1_64},
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
  return ArePtrs64bit() ? Mips::SP_64 : Mips::SP;
}

unsigned MipsABIInfo::GetFramePtr() const {
  return ArePtrs64bit() ? Mips::FP_64 : Mips::FP;
}

unsigned MipsABIInfo::GetBasePtr() const { return getSavedRegPtr(7); }

unsigned MipsABIInfo::GetGlobalPtr() const {
  return ArePtrs64bit() ? Mips::GP_64 : Mips::GP;
}

unsigned MipsABIInfo::GetNullPtr() const {
  return ArePtrs64bit() ? Mips::ZERO_64 : Mips::ZERO;
}

unsigned MipsABIInfo::GetZeroReg() const {
  return AreGprs64bit() ? Mips::ZERO_64 : Mips::ZERO;
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
