//===-- M88kMCFixups.h - M88k-specific fixup entries ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_M88K_MCTARGETDESC_M88KMCFIXUPS_H
#define LLVM_LIB_TARGET_M88K_MCTARGETDESC_M88KMCFIXUPS_H

#include "llvm/MC/MCFixup.h"

namespace llvm {
namespace M88k {
enum FixupKind {
  // These correspond directly to R_88K_* relocations.
  FK_88K_NONE = FirstTargetFixupKind, // R_88K_NONE
  FK_88K_HI, // R_88K_16H = upper 16-bits of a symbolic relocation
  FK_88K_LO, // R_88K_16H = lower 16-bits of a symbolic relocation

  // Marker
  LastTargetFixupKind,
  NumTargetFixupKinds = LastTargetFixupKind - FirstTargetFixupKind
};
} // end namespace M88k
} // end namespace llvm

#endif
