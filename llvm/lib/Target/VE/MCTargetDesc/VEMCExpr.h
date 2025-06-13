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
class VEMCExpr : public MCSpecifierExpr {
public:
  enum Specifier {
    VK_None,

    VK_REFLONG = MCSymbolRefExpr::FirstTargetSpecifier,
    VK_HI32,        // @hi
    VK_LO32,        // @lo
    VK_PC_HI32,     // @pc_hi
    VK_PC_LO32,     // @pc_lo
    VK_GOT_HI32,    // @got_hi
    VK_GOT_LO32,    // @got_lo
    VK_GOTOFF_HI32, // @gotoff_hi
    VK_GOTOFF_LO32, // @gotoff_lo
    VK_PLT_HI32,    // @plt_hi
    VK_PLT_LO32,    // @plt_lo
    VK_TLS_GD_HI32, // @tls_gd_hi
    VK_TLS_GD_LO32, // @tls_gd_lo
    VK_TPOFF_HI32,  // @tpoff_hi
    VK_TPOFF_LO32,  // @tpoff_lo
  };

private:
  explicit VEMCExpr(const MCExpr *Expr, Specifier S)
      : MCSpecifierExpr(Expr, S) {}

public:
  static const VEMCExpr *create(Specifier Kind, const MCExpr *Expr,
                                MCContext &Ctx);

  bool evaluateAsRelocatableImpl(MCValue &Res,
                                 const MCAssembler *Asm) const override;

  static VE::Fixups getFixupKind(Spec S);
};

} // namespace llvm

#endif
