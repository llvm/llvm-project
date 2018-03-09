//===-- Nios2FixupKinds.h - Nios2 Specific Fixup Entries --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NIOS2_MCTARGETDESC_NIOS2FIXUPKINDS_H
#define LLVM_LIB_TARGET_NIOS2_MCTARGETDESC_NIOS2FIXUPKINDS_H

#include "llvm/MC/MCFixup.h"

namespace llvm {
namespace Nios2 {
// Although most of the current fixup types reflect a unique relocation
// one can have multiple fixup types for a given relocation and thus need
// to be uniquely named.
//
// This table *must* be in the save order of
// MCFixupKindInfo Infos[Nios2::NumTargetFixupKinds]
// in Nios2AsmBackend.cpp.
enum Fixups {
  // Pure upper 32 bit fixup resulting in - R_NIOS2_32.
  fixup_Nios2_32 = FirstTargetFixupKind,

  // Pure upper 16 bit fixup resulting in - R_NIOS2_HI16.
  fixup_Nios2_HI16,

  // Pure lower 16 bit fixup resulting in - R_NIOS2_LO16.
  fixup_Nios2_LO16,

  // Marker
  LastTargetFixupKind,
  NumTargetFixupKinds = LastTargetFixupKind - FirstTargetFixupKind
};
} // namespace Nios2
} // namespace llvm

#endif // LLVM_NIOS2_NIOS2FIXUPKINDS_H
