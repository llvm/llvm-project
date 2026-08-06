//===-- SuperHFixupKinds.h - AVR Specific Fixup Entries ---------*- C++ -*-===//
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
///       MCFixupKindInfo Infos[AVR::NumTargetFixupKinds]
///       in `AVRAsmBackend.cpp`.
enum Fixups {

  // Fixup which uses 12 bits and is PC relative.
  // Used in specific displacement operands.
  fixup_12_pcrel = FirstTargetFixupKind,

  // Fixup which uses 8 bits and is PC relative.
  fixup_8_pcrel,

  // Fixup which uses 4 bits and is PC relative.
  fixup_4_pcrel,

  // Marker
  LastTargetFixupKind,
  NumTargetFixupKinds = LastTargetFixupKind - FirstTargetFixupKind
};

} // namespace SuperH
} // namespace llvm

#endif // LLVM_SUPERH_FIXUP_KINDS_H
