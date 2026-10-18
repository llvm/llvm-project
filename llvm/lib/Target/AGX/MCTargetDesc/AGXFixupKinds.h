//===-- AGXFixupKinds.h - AGX specific fixup entries ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AGX_MCTARGETDESC_AGXFIXUPKINDS_H
#define LLVM_LIB_TARGET_AGX_MCTARGETDESC_AGXFIXUPKINDS_H

#include "llvm/MC/MCFixup.h"

namespace llvm {
namespace AGX {
enum Fixups {
  /// 32-bit branch offset, relative to the start of the branch instruction.
  /// The field starts 2 bytes into the instruction.
  fixup_agx_branch32 = FirstTargetFixupKind,

  LastTargetFixupKind,
  NumTargetFixupKinds = LastTargetFixupKind - FirstTargetFixupKind
};
} // namespace AGX
} // namespace llvm

#endif // LLVM_LIB_TARGET_AGX_MCTARGETDESC_AGXFIXUPKINDS_H
