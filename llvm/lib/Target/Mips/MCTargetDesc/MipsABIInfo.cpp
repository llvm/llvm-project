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
static const MCPhysReg O32IntRegs[4] = {Mips::A0, Mips::A1, Mips::A2, Mips::A3};

static const MCPhysReg Mips64IntRegs[8] = {
    Mips::A0_64, Mips::A1_64, Mips::A2_64, Mips::A3_64,
    Mips::T0_64, Mips::T1_64, Mips::T2_64, Mips::T3_64};

static const MCPhysReg P32IntRegs[8] = {Mips::A0_NM, Mips::A1_NM, Mips::A2_NM,
                                        Mips::A3_NM, Mips::A4_NM, Mips::A5_NM,
                                        Mips::A6_NM, Mips::A7_NM};
}

ArrayRef<MCPhysReg> MipsABIInfo::GetByValArgRegs() const {
  if (IsO32())
    return ArrayRef(O32IntRegs);
  if (IsN32() || IsN64())
    return ArrayRef(Mips64IntRegs);
  if (IsP32())
    return ArrayRef(P32IntRegs);
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
  if (IsP32())
    return ArrayRef(P32IntRegs);
  llvm_unreachable("Unhandled ABI");
}

unsigned MipsABIInfo::GetCalleeAllocdArgSizeInBytes(CallingConv::ID CC) const {
  if (IsO32())
    return CC != CallingConv::Fast ? 16 : 0;
  if (IsN32() || IsN64() || IsP32())
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
  if (ABIName.starts_with("p32"))
    return MipsABIInfo::P32();
  if (TT.isABIN32())
    return MipsABIInfo::N32();
  assert(ABIName.empty() && "Unknown ABI option for MIPS");

  if (TT.isMIPS64())
    return MipsABIInfo::N64();
  if (TT.isNanoMips())
    return MipsABIInfo::P32();
  return MipsABIInfo::O32();
}

unsigned MipsABIInfo::GetStackPtr() const {
  return ArePtrs64bit() ? Mips::SP_64 : IsP32() ? Mips::SP_NM : Mips::SP;
}

unsigned MipsABIInfo::GetFramePtr() const {
  return ArePtrs64bit() ? Mips::FP_64 : IsP32() ? Mips::FP_NM : Mips::FP;
}

unsigned MipsABIInfo::GetBasePtr() const {
  return ArePtrs64bit() ? Mips::S7_64 : IsP32() ? Mips::S7_NM : Mips::S7;
}

unsigned MipsABIInfo::GetGlobalPtr() const {
  return ArePtrs64bit() ? Mips::GP_64 : IsP32() ? Mips::GP_NM : Mips::GP;
}

unsigned MipsABIInfo::GetNullPtr() const {
  return ArePtrs64bit() ? Mips::ZERO_64 : IsP32() ? Mips::ZERO_NM : Mips::ZERO;
}

unsigned MipsABIInfo::GetRetReg() const {
  return ArePtrs64bit() ? Mips::RA_64 : IsP32() ? Mips::RA_NM : Mips::RA;
}

unsigned MipsABIInfo::GetZeroReg() const {
  return AreGprs64bit() ? Mips::ZERO_64 : IsP32() ? Mips::ZERO_NM : Mips::ZERO;
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
  static const unsigned EhDataReg[] = {
    Mips::A0, Mips::A1, Mips::A2, Mips::A3
  };
  static const unsigned EhDataReg64[] = {
    Mips::A0_64, Mips::A1_64, Mips::A2_64, Mips::A3_64
  };

  return IsN64() ? EhDataReg64[I] : EhDataReg[I];
}
