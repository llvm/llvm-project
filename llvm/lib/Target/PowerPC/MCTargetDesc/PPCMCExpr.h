//===-- PPCMCExpr.h - PPC specific MC expression classes --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_POWERPC_MCTARGETDESC_PPCMCEXPR_H
#define LLVM_LIB_TARGET_POWERPC_MCTARGETDESC_PPCMCEXPR_H

#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCValue.h"
#include <optional>

namespace llvm {

class PPCMCExpr : public MCTargetExpr {
public:
  enum VariantKind {
    VK_None,

    // We currently use both MCSymbolRefExpr::VariantKind and
    // PPCMCExpr::VariantKind. Start at a larger number to avoid conflicts.
    VK_LO = 200,
    VK_HI,
    VK_HA,
    VK_HIGH,
    VK_HIGHA,
    VK_HIGHER,
    VK_HIGHERA,
    VK_HIGHEST,
    VK_HIGHESTA,

    VK_AIX_TLSGD,       // symbol@gd
    VK_AIX_TLSGDM,      // symbol@m
    VK_AIX_TLSIE,       // symbol@ie
    VK_AIX_TLSLD,       // symbol@ld
    VK_AIX_TLSLE,       // symbol@le
    VK_AIX_TLSML,       // symbol@ml
    VK_DTPMOD,          // symbol@dtpmod
    VK_DTPREL,          // symbol@dprel
    VK_DTPREL_HA,       // symbol@dtprel@ha
    VK_DTPREL_HI,       // symbol@dtprel@h
    VK_DTPREL_HIGH,     // symbol@dtprel@high
    VK_DTPREL_HIGHA,    // symbol@dtprel@higha
    VK_DTPREL_HIGHER,   // symbol@dtprel@higher
    VK_DTPREL_HIGHERA,  // symbol@dtprel@highera
    VK_DTPREL_HIGHEST,  // symbol@dtprel@highest
    VK_DTPREL_HIGHESTA, // symbol@dtprel@highesta
    VK_DTPREL_LO,       // symbol@dtprel@l
    VK_GOT,             // symbol@got
    VK_GOT_DTPREL,      // symbol@got@dtprel
    VK_GOT_DTPREL_HA,   // symbol@got@dtprel@ha
    VK_GOT_DTPREL_HI,   // symbol@got@dtprel@h
    VK_GOT_DTPREL_LO,   // symbol@got@dtprel@l
    VK_GOT_HA,          // symbol@got@ha
    VK_GOT_HI,          // symbol@got@h
    VK_GOT_LO,          // symbol@got@l
    VK_GOT_PCREL,       // symbol@got@pcrel
    VK_GOT_TLSGD,       // symbol@got@tlsgd
    VK_GOT_TLSGD_HA,    // symbol@got@tlsgd@ha
    VK_GOT_TLSGD_HI,    // symbol@got@tlsgd@h
    VK_GOT_TLSGD_LO,    // symbol@got@tlsgd@l
    VK_GOT_TLSGD_PCREL, // symbol@got@tlsgd@pcrel
    VK_GOT_TLSLD,       // symbol@got@tlsld
    VK_GOT_TLSLD_HA,    // symbol@got@tlsld@ha
    VK_GOT_TLSLD_HI,    // symbol@got@tlsld@h
    VK_GOT_TLSLD_LO,    // symbol@got@tlsld@l
    VK_GOT_TLSLD_PCREL, // symbol@got@tlsld@pcrel
    VK_GOT_TPREL,       // symbol@got@tprel
    VK_GOT_TPREL_HA,    // symbol@got@tprel@ha
    VK_GOT_TPREL_HI,    // symbol@got@tprel@h
    VK_GOT_TPREL_LO,    // symbol@got@tprel@l
    VK_GOT_TPREL_PCREL, // symbol@got@tprel@pcrel
    VK_L,               // symbol@l
    VK_LOCAL,           // symbol@local
    VK_NOTOC,           // symbol@notoc
    VK_PCREL,
    VK_PCREL_OPT,      // .reloc expr, R_PPC64_PCREL_OPT, expr
    VK_PLT,            // symbol@plt
    VK_TLS,            // symbol@tls
    VK_TLSGD,          // symbol@tlsgd
    VK_TLSLD,          // symbol@tlsld
    VK_TLS_PCREL,      // symbol@tls@pcrel
    VK_TOC,            // symbol@toc
    VK_TOCBASE,        // symbol@tocbase
    VK_TOC_HA,         // symbol@toc@ha
    VK_TOC_HI,         // symbol@toc@h
    VK_TOC_LO,         // symbol@toc@l
    VK_TPREL,          // symbol@tprel
    VK_TPREL_HA,       // symbol@tprel@ha
    VK_TPREL_HI,       // symbol@tprel@h
    VK_TPREL_HIGH,     // symbol@tprel@high
    VK_TPREL_HIGHA,    // symbol@tprel@higha
    VK_TPREL_HIGHER,   // symbol@tprel@higher
    VK_TPREL_HIGHERA,  // symbol@tprel@highera
    VK_TPREL_HIGHEST,  // symbol@tprel@highest
    VK_TPREL_HIGHESTA, // symbol@tprel@highesta
    VK_TPREL_LO,       // symbol@tprel@l
    VK_U,              // symbol@u
  };

private:
  const VariantKind Kind;
  const MCExpr *Expr;

  std::optional<int64_t> evaluateAsInt64(int64_t Value) const;

  explicit PPCMCExpr(VariantKind Kind, const MCExpr *Expr)
      : Kind(Kind), Expr(Expr) {}

public:
  /// @name Construction
  /// @{

  static const PPCMCExpr *create(VariantKind Kind, const MCExpr *Expr,
                                 MCContext &Ctx);

  static const PPCMCExpr *createLo(const MCExpr *Expr, MCContext &Ctx) {
    return create(VK_LO, Expr, Ctx);
  }

  static const PPCMCExpr *createHi(const MCExpr *Expr, MCContext &Ctx) {
    return create(VK_HI, Expr, Ctx);
  }

  static const PPCMCExpr *createHa(const MCExpr *Expr, MCContext &Ctx) {
    return create(VK_HA, Expr, Ctx);
  }

  /// @}
  /// @name Accessors
  /// @{

  /// getOpcode - Get the kind of this expression.
  VariantKind getKind() const { return Kind; }

  /// getSubExpr - Get the child of this expression.
  const MCExpr *getSubExpr() const { return Expr; }

  /// @}

  void printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const override;
  bool evaluateAsRelocatableImpl(MCValue &Res,
                                 const MCAssembler *Asm) const override;
  void visitUsedExpr(MCStreamer &Streamer) const override;
  MCFragment *findAssociatedFragment() const override {
    return getSubExpr()->findAssociatedFragment();
  }

  bool evaluateAsConstant(int64_t &Res) const;

  static bool classof(const MCExpr *E) {
    return E->getKind() == MCExpr::Target;
  }
};

static inline PPCMCExpr::VariantKind
getVariantKind(const MCSymbolRefExpr *SRE) {
  return PPCMCExpr::VariantKind(SRE->getKind());
}

} // end namespace llvm

#endif
