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

      /// fixup_sparc_hi22  - 22-bit fixup corresponding to %hi(foo)
      /// for sethi
      fixup_sparc_hi22,

      /// fixup_sparc_lo10  - 10-bit fixup corresponding to %lo(foo)
      fixup_sparc_lo10,

      /// fixup_sparc_hh  -  22-bit fixup corresponding to %hh(foo)
      fixup_sparc_hh,

      /// fixup_sparc_hm  -  10-bit fixup corresponding to %hm(foo)
      fixup_sparc_hm,

      /// fixup_sparc_lm  -  22-bit fixup corresponding to %lm(foo)
      fixup_sparc_lm,

      /// 22-bit fixup corresponding to %hix(foo)
      fixup_sparc_hix22,
      /// 13-bit fixup corresponding to %lox(foo)
      fixup_sparc_lox10,

      // Marker
      LastTargetFixupKind,
      NumTargetFixupKinds = LastTargetFixupKind - FirstTargetFixupKind
    };
  // clang-format on
  }
}

#endif
