//===-- DPUFixupKinds.h - DPU Specific Fixup Entries ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_DPU_MCTARGETDESC_DPUFIXUPKINDS_H
#define LLVM_LIB_TARGET_DPU_MCTARGETDESC_DPUFIXUPKINDS_H

#include "llvm/MC/MCFixup.h"

namespace llvm {
namespace DPU {
// Although most of the current fixup types reflect a unique relocation
// one can have multiple fixup types for a given relocation and thus need
// to be uniquely named.
//
// This table *must* be in the same order as
// MCFixupKindInfo Infos[DPU::NumTargetFixupKinds]
// in DPUAsmBackend.cpp.
enum Fixups {
  FIXUP_DPU_NONE = FirstTargetFixupKind,

  FIXUP_DPU_32,
  FIXUP_DPU_PC,
  FIXUP_DPU_IMM5,
  FIXUP_DPU_IMM8_DMA,
  FIXUP_DPU_IMM24_PC,
  FIXUP_DPU_IMM27_PC,
  FIXUP_DPU_IMM28_PC_OPC8,
  FIXUP_DPU_IMM8_STR,
  FIXUP_DPU_IMM12_STR,
  FIXUP_DPU_IMM16_STR,
  FIXUP_DPU_IMM16_ATM,
  FIXUP_DPU_IMM24,
  FIXUP_DPU_IMM24_RB,
  FIXUP_DPU_IMM27,
  FIXUP_DPU_IMM28,
  FIXUP_DPU_IMM32,
  FIXUP_DPU_IMM32_ZERO_RB,
  FIXUP_DPU_IMM17_24,
  FIXUP_DPU_IMM32_DUS_RB,

  // Marker
  LastTargetFixupKind,
  NumTargetFixupKinds = LastTargetFixupKind - FirstTargetFixupKind
};

static inline void applyDPUFixup(uint64_t &Data, uint64_t Value, Fixups Kind) {
  switch (Kind) {
  default:
    llvm_unreachable("Invalid kind!");
  case FIXUP_DPU_NONE:
    break;
  case FIXUP_DPU_32:
    Data |= (((Value >> 0) & 0xffffffffl) << 0);
    break;
  case FIXUP_DPU_PC:
    Data |= (((Value >> 0) & 0xffffl) << 0);
    break;
  case FIXUP_DPU_IMM5:
    Data |= (((Value >> 0) & 0xfl) << 20) | (((Value >> 4) & 0x1l) << 28);
    break;
  case FIXUP_DPU_IMM8_DMA:
    Data |= (((Value >> 0) & 0xffl) << 24);
    break;
  case FIXUP_DPU_IMM24_PC:
    Data |= (((Value >> 0) & 0xfl) << 20) | (((Value >> 4) & 0xfl) << 16);
    break;
  case FIXUP_DPU_IMM27_PC:
    Data |= (((Value >> 0) & 0xfl) << 20) | (((Value >> 4) & 0xfl) << 16) |
            (((Value >> 8) & 0x7l) << 39);
    break;
  case FIXUP_DPU_IMM28_PC_OPC8:
    Data |= (((Value >> 0) & 0xfl) << 20) | (((Value >> 4) & 0xfl) << 16) |
            (((Value >> 8) & 0x7l) << 39) | (((Value >> 11) & 0x1l) << 44);
    break;
  case FIXUP_DPU_IMM8_STR:
    Data |= (((Value >> 0) & 0xfl) << 16) | (((Value >> 4) & 0xfl) << 0);
    break;
  case FIXUP_DPU_IMM12_STR:
    Data |= (((Value >> 0) & 0xfl) << 20) | (((Value >> 4) & 0x7l) << 39) |
            (((Value >> 7) & 0x1l) << 24) | (((Value >> 8) & 0x1l) << 15) |
            (((Value >> 9) & 0x1l) << 14) | (((Value >> 10) & 0x1l) << 13) |
            (((Value >> 11) & 0x1l) << 12);
    break;
  case FIXUP_DPU_IMM16_STR:
    Data |= (((Value >> 0) & 0xfl) << 16) | (((Value >> 4) & 0xfffl) << 0);
    break;
  case FIXUP_DPU_IMM16_ATM:
    Data |= (((Value >> 0) & 0xfl) << 20) | (((Value >> 4) & 0xfl) << 16) |
            (((Value >> 8) & 0x1fl) << 26) | (((Value >> 13) & 0x7l) << 39);
    break;
  case FIXUP_DPU_IMM24:
    Data |= (((Value >> 0) & 0xfl) << 20) | (((Value >> 4) & 0xfl) << 16) |
            (((Value >> 8) & 0x1l) << 15) | (((Value >> 9) & 0x1l) << 14) |
            (((Value >> 10) & 0x1l) << 13) | (((Value >> 11) & 0x1l) << 12) |
            (((Value >> 12) & 0xfffl) << 0);
    break;
  case FIXUP_DPU_IMM24_RB:
    Data |= (((Value >> 0) & 0xfl) << 20) | (((Value >> 4) & 0x7l) << 39) |
            (((Value >> 7) & 0x1l) << 24) | (((Value >> 8) & 0x1l) << 15) |
            (((Value >> 9) & 0x1l) << 14) | (((Value >> 10) & 0x1l) << 13) |
            (((Value >> 11) & 0x1l) << 12) | (((Value >> 12) & 0xfffl) << 0);
    break;
  case FIXUP_DPU_IMM27:
    Data |= (((Value >> 0) & 0xfl) << 20) | (((Value >> 4) & 0xfl) << 16) |
            (((Value >> 8) & 0x1l) << 15) | (((Value >> 9) & 0x1l) << 14) |
            (((Value >> 10) & 0x1l) << 13) | (((Value >> 11) & 0x1l) << 12) |
            (((Value >> 12) & 0xfffl) << 0) | (((Value >> 24) & 0x7l) << 39);
    break;
  case FIXUP_DPU_IMM28:
    Data |= (((Value >> 0) & 0xfl) << 20) | (((Value >> 4) & 0xfl) << 16) |
            (((Value >> 8) & 0x1l) << 15) | (((Value >> 9) & 0x1l) << 14) |
            (((Value >> 10) & 0x1l) << 13) | (((Value >> 11) & 0x1l) << 12) |
            (((Value >> 12) & 0xfffl) << 0) | (((Value >> 24) & 0x7l) << 39) |
            (((Value >> 27) & 0x1l) << 44);
    break;
  case FIXUP_DPU_IMM32:
    Data |= (((Value >> 0) & 0xfl) << 20) | (((Value >> 4) & 0xfl) << 16) |
            (((Value >> 8) & 0x1l) << 15) | (((Value >> 9) & 0x1l) << 14) |
            (((Value >> 10) & 0x1l) << 13) | (((Value >> 11) & 0x1l) << 12) |
            (((Value >> 12) & 0xfffl) << 0) | (((Value >> 24) & 0xffl) << 24);
    break;
  case FIXUP_DPU_IMM32_ZERO_RB:
    Data |= (((Value >> 0) & 0xfl) << 20) | (((Value >> 4) & 0x7l) << 34) |
            (((Value >> 7) & 0x1l) << 39) | (((Value >> 8) & 0x1l) << 15) |
            (((Value >> 9) & 0x1l) << 14) | (((Value >> 10) & 0x1l) << 13) |
            (((Value >> 11) & 0x1l) << 12) | (((Value >> 12) & 0xfffl) << 0) |
            (((Value >> 24) & 0xffl) << 24);
    break;
  case FIXUP_DPU_IMM17_24:
    Data |= (((Value >> 0) & 0xfl) << 20) | (((Value >> 4) & 0xfl) << 16) |
            (((Value >> 8) & 0x1l) << 15) | (((Value >> 9) & 0x1l) << 14) |
            (((Value >> 10) & 0x1l) << 13) | (((Value >> 11) & 0x1l) << 12) |
            (((Value >> 12) & 0x1fl) << 0) | (((Value >> 16) & 0x1l) << 5) |
            (((Value >> 16) & 0x1l) << 6) | (((Value >> 16) & 0x1l) << 7) |
            (((Value >> 16) & 0x1l) << 8) | (((Value >> 16) & 0x1l) << 9) |
            (((Value >> 16) & 0x1l) << 10) | (((Value >> 16) & 0x1l) << 11);
    break;
  case FIXUP_DPU_IMM32_DUS_RB:
    Data |= (((Value >> 0) & 0xfl) << 20) | (((Value >> 4) & 0x7l) << 34) |
            (((Value >> 7) & 0x1l) << 44) | (((Value >> 8) & 0x1l) << 15) |
            (((Value >> 9) & 0x1l) << 14) | (((Value >> 10) & 0x1l) << 13) |
            (((Value >> 11) & 0x1l) << 12) | (((Value >> 12) & 0xfffl) << 0) |
            (((Value >> 24) & 0xffl) << 24);
    break;
  }
}

} // namespace DPU
} // namespace llvm

#endif // LLVM_LIB_TARGET_DPU_MCTARGETDESC_DPUFIXUPKINDS_H
