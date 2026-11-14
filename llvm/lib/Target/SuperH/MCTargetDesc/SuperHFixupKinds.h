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
enum Fixups {
  fixup_32 = FirstTargetFixupKind,
  
  /// A fixup that represents a PC-relative address scaled by 2.
  fixup_pcrel4_by2,

  /// A fixup that represents a PC-relative address scaled by 4.
  fixup_pcrel4_by4,

  /// A fixup that represents a PC-relative address scaled by 4.
  fixup_pcrel8_by4,

  /// A fixup that represents a PC-relative address scaled by 2,
  /// with a 4-byte offset.
  fixup_pcrel8_4by2,

  /// A fixup that represents a PC-relative address scaled by 4,
  /// with a 4-byte offset.
  fixup_pcrel8_4by4,

  /// A fixup that represents a PC-relative address scaled by 2,
  /// with a 4-byte offset.
  fixup_pcrel12_4by2,

  // Marker
  LastTargetFixupKind,
  NumTargetFixupKinds = LastTargetFixupKind - FirstTargetFixupKind,
};

} // namespace SuperH
} // namespace llvm

#endif // LLVM_SUPERH_FIXUP_KINDS_H
