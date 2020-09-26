//===-- P2FixupKinds.h - P2 Specific Fixup Entries ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_P2_MCTARGETDESC_P2FIXUPKINDS_H
#define LLVM_LIB_TARGET_P2_MCTARGETDESC_P2FIXUPKINDS_H

#include "llvm/MC/MCFixup.h"

namespace llvm {
    namespace P2 {
        // Although most of the current fixup types reflect a unique relocation
        // one can have multiple fixup types for a given relocation and thus need
        // to be uniquely named.
        //
        // This table *must* be in the save order of
        // MCFixupKindInfo Infos[P2::NumTargetFixupKinds]
        // in P2AsmBackend.cpp.
        //
        enum Fixups {
            // Pure 32 bit fixup
            fixup_P2_32 = FirstTargetFixupKind,

            // 32 bit PC relative fixup (i don't think these would ever exist)
            fixup_P2_PC32,

            // 20 bit fixup for calls
            fixup_P2_20,

            // 20 bit pc-relative fixup for jumps
            fixup_P2_PC20,

            // 20+ bit fixup for global addresses
            fixup_P2_AUG20,

            // 9 bit fixup for cog based functions
            fixup_P2_COG9,

            // 9 bit PC relative fixup. same as PC20, but divides by 4
            fixup_P2_PCCOG9,

            // Marker
            LastTargetFixupKind,
            NumTargetFixupKinds = LastTargetFixupKind - FirstTargetFixupKind
        };
    }
}

#endif // LLVM_P2_P2FIXUPKINDS_H

