//===-- EZHFixupKinds.h - EZH Specific Fixup Entries --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_EZH_MCTARGETDESC_EZHFIXUPKINDS_H
#define LLVM_LIB_TARGET_EZH_MCTARGETDESC_EZHFIXUPKINDS_H

#include "llvm/MC/MCFixup.h"

namespace llvm {
namespace EZH {
// Although most of the current fixup types reflect a unique relocation
// one can have multiple fixup types for a given relocation and thus need
// to be uniquely named.
//
// This table *must* be in the save order of
// MCFixupKindInfo Infos[EZH::NumTargetFixupKinds]
// in EZHAsmBackend.cpp.
//
enum Fixups {
  // Results in R_EZH_NONE
  FIXUP_EZH_NONE = FirstTargetFixupKind,

  FIXUP_EZH_21,      // 21-bit symbol relocation
  FIXUP_EZH_21_F,    // 21-bit symbol relocation, last two bits masked to 0
  FIXUP_EZH_25,      // 25-bit branch targets
  FIXUP_EZH_32,      // general 32-bit relocation
  FIXUP_EZH_HI16,    // upper 16-bits of a symbolic relocation
  FIXUP_EZH_LO16,    // lower 16-bits of a symbolic relocation
  FIXUP_EZH_8_PCREL, // 8-bit PC-relative word offset

  // Marker
  LastTargetFixupKind,
  NumTargetFixupKinds = LastTargetFixupKind - FirstTargetFixupKind
};
} // namespace EZH
} // namespace llvm

#endif // LLVM_LIB_TARGET_EZH_MCTARGETDESC_EZHFIXUPKINDS_H
