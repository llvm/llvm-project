//===---- MipsABIInfo.cpp - Information about MIPS ABI's ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MipsABIInfo.h"
#include "MipsRegisterInfo.h"
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

namespace {
static const MCPhysReg O32IntRegs[4] = {Mips::A0, Mips::A1, Mips::A2, Mips::A3};

static const MCPhysReg Mips64IntRegs[8] = {
    Mips::A0_64, Mips::A1_64, Mips::A2_64, Mips::A3_64,
    Mips::T0_64, Mips::T1_64, Mips::T2_64, Mips::T3_64};
}

ArrayRef<MCPhysReg> MipsABIInfo::GetByValArgRegs() const {
  if (IsO32())
    return makeArrayRef(O32IntRegs);
  if (IsN32() || IsN64())
    return makeArrayRef(Mips64IntRegs);
  llvm_unreachable("Unhandled ABI");
}

ArrayRef<MCPhysReg> MipsABIInfo::GetVarArgRegs() const {
  if (IsO32())
    return makeArrayRef(O32IntRegs);
  if (IsN32() || IsN64())
    return makeArrayRef(Mips64IntRegs);
  llvm_unreachable("Unhandled ABI");
}

unsigned MipsABIInfo::GetCalleeAllocdArgSizeInBytes(CallingConv::ID CC) const {
  if (IsO32())
    return CC != CallingConv::Fast ? 16 : 0;
  if (IsN32() || IsN64() || IsP32())
    return 0;
  llvm_unreachable("Unhandled ABI");
}

MipsABIInfo MipsABIInfo::computeTargetABI(const Triple &TT, StringRef CPU,
                                          const MCTargetOptions &Options) {
  if (Options.getABIName().startswith("o32"))
    return MipsABIInfo::O32();
  if (Options.getABIName().startswith("n32"))
    return MipsABIInfo::N32();
  if (Options.getABIName().startswith("n64"))
    return MipsABIInfo::N64();
  if (TT.getEnvironment() == llvm::Triple::GNUABIN32)
    return MipsABIInfo::N32();
  assert(Options.getABIName().empty() && "Unknown ABI option for MIPS");

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

unsigned MipsABIInfo::GetZeroReg() const {
  return AreGprs64bit() ? Mips::ZERO_64 : IsP32() ? Mips::ZERO_NM : Mips::ZERO;
}

unsigned MipsABIInfo::GetPtrAdduOp() const {
  return ArePtrs64bit() ? Mips::DADDu : IsP32() ? Mips::ADDu_NM : Mips::ADDu;
}

unsigned MipsABIInfo::GetPtrAddiuOp() const {
  return ArePtrs64bit() ? Mips::DADDiu : IsP32() ? Mips::ADDiu_NM : Mips::ADDiu;
}

unsigned MipsABIInfo::GetPtrSubuOp() const {
  return ArePtrs64bit() ? Mips::DSUBu : IsP32() ? Mips::SUBu_NM : Mips::SUBu;
}

unsigned MipsABIInfo::GetPtrAndOp() const {
  return ArePtrs64bit() ? Mips::AND64 : IsP32() ? Mips::AND_NM : Mips::AND;
}

unsigned MipsABIInfo::GetGPRMoveOp() const {
  // TODO: Add nanoMIPS MOVE instruction. Avoid assert to keep tests passing.
  return ArePtrs64bit() ? Mips::OR64 : Mips::OR;
}

unsigned MipsABIInfo::GetEhDataReg(unsigned I) const {
  static const unsigned EhDataReg[] = {
    Mips::A0, Mips::A1, Mips::A2, Mips::A3
  };
  static const unsigned EhDataReg64[] = {
    Mips::A0_64, Mips::A1_64, Mips::A2_64, Mips::A3_64
  };

  assert(!IsP32() && "GetEhDataReg NYI for nanoMIPS");
  return IsN64() ? EhDataReg64[I] : EhDataReg[I];
}

