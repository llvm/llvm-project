//====- VEMCExpr.h - VE specific MC expression classes --------*- C++ -*-=====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes VE-specific MCExprs, used for modifiers like
// "%hi" or "%lo" etc.,
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_VE_MCTARGETDESC_VEMCEXPR_H
#define LLVM_LIB_TARGET_VE_MCTARGETDESC_VEMCEXPR_H

#include "VEFixupKinds.h"
#include "llvm/MC/MCExpr.h"

namespace llvm {

class StringRef;

namespace VE {
enum Specifier {
  S_None,

  S_REFLONG = MCSymbolRefExpr::FirstTargetSpecifier,
  S_HI32,        // @hi
  S_LO32,        // @lo
  S_PC_HI32,     // @pc_hi
  S_PC_LO32,     // @pc_lo
  S_GOT_HI32,    // @got_hi
  S_GOT_LO32,    // @got_lo
  S_GOTOFF_HI32, // @gotoff_hi
  S_GOTOFF_LO32, // @gotoff_lo
  S_PLT_HI32,    // @plt_hi
  S_PLT_LO32,    // @plt_lo
  S_TLS_GD_HI32, // @tls_gd_hi
  S_TLS_GD_LO32, // @tls_gd_lo
  S_TPOFF_HI32,  // @tpoff_hi
  S_TPOFF_LO32,  // @tpoff_lo
};

VE::Fixups getFixupKind(uint8_t S);
} // namespace VE

} // namespace llvm

#endif
