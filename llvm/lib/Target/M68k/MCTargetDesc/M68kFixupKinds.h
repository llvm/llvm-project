//===-- M68kFixupKinds.h - M68k Specific Fixup Entries ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains M68k specific fixup entries.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_M68k_MCTARGETDESC_M68kFIXUPKINDS_H
#define LLVM_LIB_TARGET_M68k_MCTARGETDESC_M68kFIXUPKINDS_H

#include "llvm/MC/MCFixup.h"

namespace llvm {
static inline unsigned getFixupKindLog2Size(unsigned Kind) {
  switch (Kind) {
  case FK_Data_1:
    return 0;
  case FK_Data_2:
    return 1;
  case FK_Data_4:
    return 2;
  }
  llvm_unreachable("invalid fixup kind!");
}
} // namespace llvm

#endif // LLVM_LIB_TARGET_M68k_MCTARGETDESC_M68kFIXUPKINDS_H
