//===-- NVPTXBaseInfo.h - Top-level definitions for NVPTX -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains small standalone helper functions and enum definitions for
// the NVPTX target useful for the compiler back-end and the MC libraries.
// As such, it deliberately does not include references to LLVM core
// code gen types, passes, etc..
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NVPTX_MCTARGETDESC_NVPTXBASEINFO_H
#define LLVM_LIB_TARGET_NVPTX_MCTARGETDESC_NVPTXBASEINFO_H

#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/NVPTXAddrSpace.h"
namespace llvm {

using namespace NVPTXAS;

namespace NVPTX {

// PTX virtual registers are numbered per register class. The asm printer
// packs the class into the upper bits of the register number so that the inst
// printer can recover the name the register was declared with.
enum class VirtualRegisterKind : unsigned {
  Physical = 0,
  B1 = 1,
  B16 = 2,
  B32 = 3,
  B64 = 4,
  B128 = 5,
};

constexpr unsigned VirtualRegisterKindShift = 27;
constexpr unsigned VirtualRegisterNumMask =
    (1u << VirtualRegisterKindShift) - 1;

// The packed registers are carried in MCOperands, so they must stay inside the
// range MCRegister reserves for physical registers.
static_assert(((static_cast<unsigned>(VirtualRegisterKind::B128)
                << VirtualRegisterKindShift) |
               VirtualRegisterNumMask) <= MCRegister::LastPhysicalReg,
              "Packed virtual register does not fit in an MCRegister");

/// The name prefix shared by all virtual registers of \p Kind.
inline StringRef getVirtualRegisterPrefix(VirtualRegisterKind Kind) {
  switch (Kind) {
  case VirtualRegisterKind::B1:
    return "%p";
  case VirtualRegisterKind::B16:
    return "%rs";
  case VirtualRegisterKind::B32:
    return "%r";
  case VirtualRegisterKind::B64:
    return "%rd";
  case VirtualRegisterKind::B128:
    return "%rq";
  case VirtualRegisterKind::Physical:
    break;
  }
  llvm_unreachable("Invalid virtual register kind");
}

} // namespace NVPTX

namespace NVPTXII {
enum {
  // These must be kept in sync with TSFlags in NVPTXInstrFormats.td
  // clang-format off
  IsTexFlag            =  0x40,
  IsSuldMask           = 0x180,
  IsSuldShift          =   0x7,
  IsSustFlag           = 0x200,
  IsSurfTexQueryFlag   = 0x400,
  IsTexModeUnifiedFlag = 0x800,
  // clang-format on
};
} // namespace NVPTXII

} // namespace llvm
#endif
