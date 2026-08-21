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
namespace SH {

// clang-format off

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
  // NOTE:  Some relocation names are misspelled in the official manual,
  //        The misspellings are corrected as follows:
  //          - R_SH_PLT_MEWLOW16 -> R_SH_PLT_MEDLOW16
  //
  //  Name                Value          Field           Calculation

  // R_SH_GOT32           160            word32          G + A
  fixup_got32 = FirstTargetFixupKind,

  // R_SH_GOT_LOW16       169            T_32s10for16    (G + A) & 65535
  fixup_got_low16,

  // R_SH_GOT_MEDLOW16    170            T_32u10for16    ((G + A) >> 16) & 65535
  fixup_got_medlow16,

  // R_SH_GOT_MEDHI16     171            T_32u10for16    ((G + A) >> 32) & 65535
  fixup_got_medhi16,

  // R_SH_GOT_HI16        172            T_32u10for16    ((G + A) >> 48) & 65535
  fixup_got_hi16,

  // R_SH_GOT10BY4        189            V_32s10for10    (G + A) / 4
  fixup_got10by4,

  // R_SH_GOT10BY8        191            V_32s10for10    (G + A) / 8
  fixup_got10by8,

  // R_SH_PLT32           161            word32          L + A - P
  fixup_plt32,

  // R_SH_PLT_LOW16       177            T_32s10for16    (L + A - P) & 65535
  fixup_plt_low16,

  // R_SH_PLT_MEDLOW16    178            T_32u10for16    ((L + A - P) >> 16) & 65535
  fixup_plt_medlow16,

  // R_SH_PLT_MEDHI16     179            T_32u10for16    ((L + A - P) >> 32) & 65535
  fixup_plt_medhi16,

  // R_SH_PLT_HI16        180            T_32u10for16    ((L + A - P) >> 48) & 65535
  fixup_plt_hi16,

  // R_SH_GOTPLT32        168            word32          G + A
  fixup_gotplt32,

  // R_SH_GOTPLT_LOW16    169            T_32s10for16    (G + A) & 65535
  fixup_gotplt_low16,

  // R_SH_GOTPLT_MEDLOW16 170            T_32u10for16    ((G + A) >> 16) & 65535
  fixup_gotplt_medlow16,

  // R_SH_GOTPLT_MEDHI16  171            T_32u10for16    ((G + A) >> 32) & 65535
  fixup_gotplt_medhi16,

  // R_SH_GOTPLT_HI16     172            T_32u10for16    ((G + A) >> 48) & 65535
  fixup_gotplt_hi16,

  // R_SH_GOTOFF          166            word32          S + A - GOT
  fixup_gotoff,

  // R_SH_GOTOFF_LOW16    181            T_32s10for16    (S + A - GOT) & 65535
  fixup_gotoff_low16,

  // R_SH_GOTOFF_MEDLOW16 182            T_32u10for16    ((S + A - GOT) >> 16) & 65535
  fixup_gotoff_medlow16,

  // R_SH_GOTOFF_MEDHI16  183            T_32u10for16    ((S + A - GOT) >> 32) & 65535
  fixup_gotoff_medhi16,

  // R_SH_GOTOFF_HI16     184            T_32u10for16    ((S + A - GOT) >> 48) & 65535
  fixup_gotoff_hi16,

  // R_SH_GOTPC           167            word32          GOT + A - P
  fixup_gotpc,

  // R_SH_GOTPC_LOW16     185            T_32s10for16    (GOT + A - P) & 65535
  fixup_gotpc_low16,

  // R_SH_GOTPC_MEDLOW16  186            T_32u10for16    ((GOT + A - P) >> 16) & 65535
  fixup_gotpc_medlow16,

  // R_SH_GOTPC_MEDHI16   187            T_32u10for16    ((GOT + A - P) >> 32) & 65535
  fixup_gotpc_medhi16,

  // R_SH_GOTPC_HI16      188            T_32u10for16    ((GOT + A - P) >> 48) & 65535
  fixup_gotpc_hi16,

  // R_SH_COPY            162            none            none
  fixup_copy,

  // R_SH_COPY64          193            none            none
  fixup_copy64,

  // R_SH_GLOB_DAT        163            word32          S
  fixup_glob_dat,

  // R_SH_GLOB_DAT64      194            word64          S
  fixup_glob_dat64,

  // R_SH_JMP_SLOT        164            word32          S
  fixup_jump_slot,

  // R_SH_JMP_SLOT64      195            word64          S
  fixup_jump_slot64,

  // R_SH_RELATIVE        165            word32          B + A
  fixup_relative,

  // R_SH_RELATIVE64      196            word64          B + A
  fixup_relative64,

  // R_SH_DIR32           1              word32          S + A
  fixup_dir32,

  // R_SH_REL32           2              word32          S + A - P
  fixup_rel32,

  // R_SH_64              254            word64          S + A
  fixup_64,

  // R_SH_64_PCREL        255            word64          S + A - P
  fixup_64_pcrel,

  // Marker
  LastTargetFixupKind,
  NumTargetFixupKinds = LastTargetFixupKind - FirstTargetFixupKind,
};

} // namespace SuperH
} // namespace llvm

#endif // LLVM_SUPERH_FIXUP_KINDS_H
