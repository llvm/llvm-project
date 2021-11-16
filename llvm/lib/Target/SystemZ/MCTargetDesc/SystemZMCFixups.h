//===-- SystemZMCFixups.h - SystemZ-specific fixup entries ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SYSTEMZ_MCTARGETDESC_SYSTEMZMCFIXUPS_H
#define LLVM_LIB_TARGET_SYSTEMZ_MCTARGETDESC_SYSTEMZMCFIXUPS_H

#include "llvm/MC/MCFixup.h"

namespace llvm {
namespace SystemZ {
enum FixupKind {
  // These correspond directly to R_390_* relocations.
  FK_390_PC12DBL = FirstTargetFixupKind,
  FK_390_PC16DBL,
  FK_390_PC24DBL,
  FK_390_PC32DBL,
  FK_390_TLS_CALL,
  FK_390_12,
  FK_390_20,

  // Marker
  LastTargetFixupKind,
  NumTargetFixupKinds = LastTargetFixupKind - FirstTargetFixupKind
};
} // end namespace SystemZ
} // end namespace llvm

#endif
