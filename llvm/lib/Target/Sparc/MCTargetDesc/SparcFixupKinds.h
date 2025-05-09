//===-- SparcFixupKinds.h - Sparc Specific Fixup Entries --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SPARC_MCTARGETDESC_SPARCFIXUPKINDS_H
#define LLVM_LIB_TARGET_SPARC_MCTARGETDESC_SPARCFIXUPKINDS_H

#include "llvm/MC/MCFixup.h"

namespace llvm {
  namespace Sparc {
  // clang-format off
    enum Fixups {
      // fixup_sparc_call30 - 30-bit PC relative relocation for call
      fixup_sparc_call30 = FirstTargetFixupKind,

      /// fixup_sparc_13 - 13-bit fixup
      fixup_sparc_13,

      // Marker
      LastTargetFixupKind,
      NumTargetFixupKinds = LastTargetFixupKind - FirstTargetFixupKind
    };
  // clang-format on
  }
}

#endif
