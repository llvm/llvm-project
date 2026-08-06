//===-- SuperHFixupKinds.h - SuperH Specific Fixup Entries ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPERH_FIXUP_KINDS_H
#define LLVM_SUPERH_FIXUP_KINDS_H

#include "llvm/MC/MCFixup.h"

namespace llvm {
namespace SuperH {

/// The set of supported fixups.
///
/// Although most of the current fixup types reflect a unique relocation
/// one can have multiple fixup types for a given relocation and thus need
/// to be uniquely named.
///
/// \note This table *must* be in the same order of
///       MCFixupKindInfo Infos[SuperH::NumTargetFixupKinds]
///       in `SuperHAsmBackend.cpp`.
enum Fixups {
  // The following fields follow the instructions in the C ABI Specification
  // which can be found at https://www.renesas.com/en/document/mat/superh-cc-compiler-package-v904-users-manual?r=1169516
  //
  //
  //  Name                 Value          Field           Calculation

  /// R_SH_GOT32           160            word32          G + A
  fixup_got32 = FirstTargetFixupKind,

  /// R_SH_GOT_LOW16       169            T_32s10for16    (G + A) & 65535
  fixup_got_low16,

  /// R_SH_GOT_MEDLOW16    170            T_32u10for16    ((G + A) >> 16) & 65535
  fixup_got_medlow16,

  /// R_SH_GOT_MEDHI16     171            T_32u10for16    ((G + A) >> 32) & 65535
  fixup_got_medhi16,

  /// R_SH_GOT_HI16        172            T_32u10for16    ((G + A) >> 48) & 65535
  fixup_got_hi16,

  /// R_SH_GOT10BY4        189            V_32s10for10    (G + A) / 4
  fixup_got10by4,

  /// R_SH_GOT10BY8        191            V_32s10for10    (G + A) / 8
  fixup_got10by8,

  /// R_SH_PLT32           161            word32          L + A - P
  fixup_plt32,

  // Marker
  LastTargetFixupKind,
  NumTargetFixupKinds = LastTargetFixupKind - FirstTargetFixupKind,
};

} // namespace SuperH
} // namespace llvm

#endif // LLVM_SUPERH_FIXUP_KINDS_H
