//===- LoongArchFixupKinds.h - LoongArch Specific Fixup Entries -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_LOONGARCH_MCTARGETDESC_LOONGARCHFIXUPKINDS_H
#define LLVM_LIB_TARGET_LOONGARCH_MCTARGETDESC_LOONGARCHFIXUPKINDS_H

#include "llvm/MC/MCFixup.h"

#undef LoongArch

namespace llvm {
namespace LoongArch {
//
// This table *must* be in the same order of
// MCFixupKindInfo Infos[LoongArch::NumTargetFixupKinds] in
// LoongArchAsmBackend.cpp.
//
enum Fixups {
  // 26-bit fixup for symbol references in the b/bl instructions.
  fixup_loongarch_b26 = FirstTargetFixupKind,
  // 20-bit fixup corresponding to %pc_hi20(foo) for instruction pcalau12i.
  fixup_loongarch_pcala_hi20,
  // 12-bit fixup corresponding to %pc_lo12(foo) for instructions addi.w/d.
  fixup_loongarch_pcala_lo12,
  // TODO: Add more fixup kind.

  // Used as a sentinel, must be the last.
  fixup_loongarch_invalid,
  NumTargetFixupKinds = fixup_loongarch_invalid - FirstTargetFixupKind
};
} // end namespace LoongArch
} // end namespace llvm

#endif
