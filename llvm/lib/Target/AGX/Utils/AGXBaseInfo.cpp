//===-- AGXBaseInfo.cpp - AGX shared definitions --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AGXBaseInfo.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define GET_REGINFO_ENUM
#include "AGXGenRegisterInfo.inc"

using namespace llvm;

double AGX::decodeFPImm8(unsigned Imm) {
  double Sign = (Imm & 0x80) ? -1.0 : 1.0;
  unsigned E = (Imm >> 4) & 7;
  unsigned F = Imm & 0xf;
  if (E == 0)
    return Sign * F / 64.0;
  return Sign * double(0x10 | F) * std::ldexp(1.0, int(E) - 7);
}

std::optional<unsigned> AGX::encodeFPImm8(double V) {
  for (unsigned I = 0; I < 256; ++I) {
    double D = decodeFPImm8(I);
    if (D == V && std::signbit(D) == std::signbit(V))
      return I;
  }
  return std::nullopt;
}

std::string AGX::formatFPImm8(unsigned Imm) {
  // Every 8-bit float is a multiple of 1/64 below 32, so six decimals are
  // exact. Print like Python's repr: no trailing zeros, at least one.
  char Buf[32];
  std::snprintf(Buf, sizeof(Buf), "%.6f", decodeFPImm8(Imm));
  std::string S(Buf);
  while (S.size() > 2 && S.back() == '0' && S[S.size() - 2] != '.')
    S.pop_back();
  return S;
}

StringRef AGX::getICondName(unsigned CC) {
  switch (CC) {
  case 0b0000: return "ueq";
  case 0b0001: return "ult";
  case 0b0010: return "ugt";
  case 0b0100: return "seq";
  case 0b0101: return "slt";
  case 0b0110: return "sgt";
  case 0b1000: return "nueq";
  case 0b1001: return "ugte";
  case 0b1010: return "ulte";
  case 0b1100: return "nseq";
  case 0b1101: return "sgte";
  case 0b1110: return "slte";
  default: return "";
  }
}

StringRef AGX::getFCondName(unsigned CC) {
  switch (CC) {
  case 0b0000: return "eq";
  case 0b0001: return "lt";
  case 0b0010: return "gt";
  case 0b0011: return "ltn";
  case 0b0101: return "gte";
  case 0b0110: return "lte";
  case 0b0111: return "gtn";
  case 0b1000: return "neq";
  case 0b1001: return "nlt";
  case 0b1010: return "ngt";
  case 0b1011: return "nltn";
  case 0b1101: return "ngte";
  case 0b1110: return "nlte";
  case 0b1111: return "ngtn";
  default: return "";
  }
}

std::optional<unsigned> AGX::parseICond(StringRef Name) {
  for (unsigned CC = 0; CC < 16; ++CC)
    if (!getICondName(CC).empty() && getICondName(CC) == Name)
      return CC;
  return std::nullopt;
}

std::optional<unsigned> AGX::parseFCond(StringRef Name) {
  for (unsigned CC = 0; CC < 16; ++CC)
    if (!getFCondName(CC).empty() && getFCondName(CC) == Name)
      return CC;
  return std::nullopt;
}

StringRef AGX::getMemFormatName(unsigned Fmt) {
  switch (Fmt) {
  case FMT_I8: return "i8";
  case FMT_I16: return "i16";
  case FMT_I32: return "i32";
  case FMT_F16: return "f16";
  case FMT_U8NORM: return "u8norm";
  case FMT_S8NORM: return "s8norm";
  case FMT_U16NORM: return "u16norm";
  case FMT_S16NORM: return "s16norm";
  case FMT_RGB10A2: return "rgb10a2";
  case FMT_SRGBA8: return "srgba8";
  case FMT_RG11B10F: return "rg11b10f";
  case FMT_RGB9E5: return "rgb9e5";
  case 14: return "rg11b10f.rtz";
  case 15: return "rgb9e5.rtz";
  default: return "";
  }
}

std::optional<unsigned> AGX::parseMemFormat(StringRef Name) {
  for (unsigned F = 0; F < 16; ++F)
    if (!getMemFormatName(F).empty() && getMemFormatName(F) == Name)
      return F;
  return std::nullopt;
}

StringRef AGX::getAtomicOpName(unsigned Op) {
  static const char *const Names[] = {"add",  "sub",  "xchg", "cmpxchg",
                                      "umin", "imin", "umax", "imax",
                                      "and",  "or",   "xor"};
  return Op < std::size(Names) ? Names[Op] : "";
}

std::optional<unsigned> AGX::parseAtomicOp(StringRef Name) {
  for (unsigned Op = 0; Op < 16; ++Op)
    if (!getAtomicOpName(Op).empty() && getAtomicOpName(Op) == Name)
      return Op;
  return std::nullopt;
}

StringRef AGX::getConvertModeName(unsigned Mode) {
  switch (Mode) {
  case 0: return "u8_to_f";
  case 1: return "s8_to_f";
  case 4: return "f_to_u16";
  case 5: return "f_to_s16";
  case 6: return "u16_to_f";
  case 7: return "s16_to_f";
  case 8: return "f_to_u32";
  case 9: return "f_to_s32";
  case 10: return "u32_to_f";
  case 11: return "s32_to_f";
  default: return "";
  }
}

std::optional<unsigned> AGX::parseConvertMode(StringRef Name) {
  for (unsigned M = 0; M < 64; ++M)
    if (!getConvertModeName(M).empty() && getConvertModeName(M) == Name)
      return M;
  return std::nullopt;
}

StringRef AGX::getRoundModeName(unsigned Mode) {
  switch (Mode) {
  case 0: return "rtz";
  case 1: return "rte";
  default: return "";
  }
}

std::optional<unsigned> AGX::parseRoundMode(StringRef Name) {
  return StringSwitch<std::optional<unsigned>>(Name)
      .Case("rtz", 0)
      .Case("rte", 1)
      .Default(std::nullopt);
}

StringRef AGX::getSpecialRegName(unsigned SR) {
  switch (SR) {
  case 0: return "threadgroup_position_in_grid.x";
  case 1: return "threadgroup_position_in_grid.y";
  case 2: return "threadgroup_position_in_grid.z";
  case 4: return "threads_per_threadgroup.x";
  case 5: return "threads_per_threadgroup.y";
  case 6: return "threads_per_threadgroup.z";
  case 8: return "dispatch_threads_per_threadgroup.x";
  case 9: return "dispatch_threads_per_threadgroup.y";
  case 10: return "dispatch_threads_per_threadgroup.z";
  case 14: return "samples_log2";
  case 20: return "core_index";
  case 21: return "vm_slot";
  case 48: return "thread_position_in_threadgroup.x";
  case 49: return "thread_position_in_threadgroup.y";
  case 50: return "thread_position_in_threadgroup.z";
  case 51: return "thread_index_in_threadgroup";
  case 52: return "thread_index_in_simdgroup";
  case 53: return "simdgroup_index_in_threadgroup";
  case 56: return "active_thread_index_in_quadgroup";
  case 57: return "total_active_threads_in_quadgroup";
  case 58: return "active_thread_index_in_simdgroup";
  case 59: return "total_active_threads_in_simdgroup";
  case 60: return "internal_coverage_mask";
  case 62: return "backfacing";
  case 63: return "is_active_thread";
  case 80: return "thread_position_in_grid.x";
  case 81: return "thread_position_in_grid.y";
  case 82: return "thread_position_in_grid.z";
  case 124: return "input_sample_mask";
  case 144: return "opfifo_cmd";
  case 146: return "opfifo_data_l";
  case 147: return "opfifo_data_h";
  default: return "";
  }
}

std::string AGX::getMemMaskName(unsigned Mask) {
  std::string S;
  if (Mask & 1)
    S += 'x';
  if (Mask & 2)
    S += 'y';
  if (Mask & 4)
    S += 'z';
  if (Mask & 8)
    S += 'w';
  return S;
}

std::optional<unsigned> AGX::parseMemMask(StringRef Name) {
  for (unsigned M = 1; M < 16; ++M)
    if (getMemMaskName(M) == Name)
      return M;
  return std::nullopt;
}

unsigned AGX::RegInfo::getSizeInBits() const {
  switch (Kind) {
  case RegKind::GPR16:
  case RegKind::UGPR16:
  case RegKind::Nest:
    return 16;
  case RegKind::GPR32:
  case RegKind::UGPR32:
    return 32;
  case RegKind::GPR64:
  case RegKind::UGPR64:
    return 64;
  case RegKind::GPR96:
    return 96;
  case RegKind::GPR128:
    return 128;
  case RegKind::Invalid:
    break;
  }
  return 0;
}

namespace {
struct KindClass {
  AGX::RegKind Kind;
  unsigned ClassID;
};
} // namespace

static const KindClass KindClasses[] = {
    {AGX::RegKind::GPR16, AGX::GPR16RegClassID},
    {AGX::RegKind::GPR32, AGX::GPR32RegClassID},
    {AGX::RegKind::GPR64, AGX::GPR32x2RegClassID},
    {AGX::RegKind::GPR96, AGX::GPR96RegClassID},
    {AGX::RegKind::GPR128, AGX::GPR128RegClassID},
    {AGX::RegKind::UGPR16, AGX::UGPR16RegClassID},
    {AGX::RegKind::UGPR32, AGX::UGPR32RegClassID},
    {AGX::RegKind::UGPR64, AGX::UGPR64RegClassID},
};

AGX::RegInfo AGX::getRegInfo(MCRegister Reg, const MCRegisterInfo &MRI) {
  RegInfo Info;
  if (!Reg)
    return Info;
  for (const KindClass &KC : KindClasses) {
    const MCRegisterClass &RC = MRI.getRegClass(KC.ClassID);
    if (!RC.contains(Reg))
      continue;
    Info.Kind = KC.Kind;
    MCRegister First = Reg;
    if (KC.Kind == RegKind::GPR64 || KC.Kind == RegKind::GPR96 ||
        KC.Kind == RegKind::GPR128 || KC.Kind == RegKind::UGPR64)
      First = MRI.getSubReg(Reg, AGX::sub0);
    Info.HalfIndex = MRI.getEncodingValue(First);
    return Info;
  }
  return Info;
}

MCRegister AGX::getRegister(RegKind Kind, unsigned HalfIndex,
                            const MCRegisterInfo &MRI) {
  unsigned ClassID = 0;
  unsigned Index = HalfIndex;
  switch (Kind) {
  case RegKind::GPR16:
    ClassID = AGX::GPR16RegClassID;
    break;
  case RegKind::UGPR16:
    ClassID = AGX::UGPR16RegClassID;
    break;
  case RegKind::GPR32:
  case RegKind::GPR64:
  case RegKind::GPR96:
  case RegKind::GPR128:
  case RegKind::UGPR32:
  case RegKind::UGPR64:
    if (HalfIndex & 1)
      return MCRegister();
    Index = HalfIndex / 2;
    ClassID = Kind == RegKind::GPR32    ? AGX::GPR32RegClassID
              : Kind == RegKind::GPR64  ? AGX::GPR32x2RegClassID
              : Kind == RegKind::GPR96  ? AGX::GPR96RegClassID
              : Kind == RegKind::GPR128 ? AGX::GPR128RegClassID
              : Kind == RegKind::UGPR32 ? AGX::UGPR32RegClassID
                                        : AGX::UGPR64RegClassID;
    break;
  case RegKind::Nest:
  case RegKind::Invalid:
    return MCRegister();
  }
  const MCRegisterClass &RC = MRI.getRegClass(ClassID);
  if (Index >= RC.getNumRegs())
    return MCRegister();
  MCRegister Reg = RC.getRegister(Index);
  RegInfo Info = getRegInfo(Reg, MRI);
  if (Info.Kind != Kind || Info.HalfIndex != HalfIndex)
    return MCRegister();
  return Reg;
}
