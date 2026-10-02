//===-- AMDGPUFixupKinds.h - AMDGPU Specific Fixup Entries ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUFIXUPKINDS_H
#define LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUFIXUPKINDS_H

#include "llvm/MC/MCFixup.h"

namespace llvm {
namespace AMDGPU {
enum Fixups {
  /// 16-bit PC relative fixup for SOPP branch instructions.
  fixup_si_sopp_br = FirstTargetFixupKind,

  /// 32-bit PC relative fixup for the literal of s_add_pc_i64 when it is used
  /// as a long branch.
  fixup_si_add_pc_lit32,

  // Marker
  LastTargetFixupKind,
  NumTargetFixupKinds = LastTargetFixupKind - FirstTargetFixupKind
};
}
}

#endif
