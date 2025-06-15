//===-- PPCMCAsmInfo.h - PPC asm properties --------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the PowerPC MCAsmInfo classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_POWERPC_MCTARGETDESC_PPCMCASMINFO_H
#define LLVM_LIB_TARGET_POWERPC_MCTARGETDESC_PPCMCASMINFO_H

#include "llvm/MC/MCAsmInfoELF.h"
#include "llvm/MC/MCAsmInfoXCOFF.h"

namespace llvm {
class Triple;

class PPCELFMCAsmInfo : public MCAsmInfoELF {
  void anchor() override;

public:
  explicit PPCELFMCAsmInfo(bool is64Bit, const Triple &);
  void printSpecifierExpr(raw_ostream &OS,
                          const MCSpecifierExpr &Expr) const override;
  bool evaluateAsRelocatableImpl(const MCSpecifierExpr &Expr, MCValue &Res,
                                 const MCAssembler *Asm) const override;
};

class PPCXCOFFMCAsmInfo : public MCAsmInfoXCOFF {
  void anchor() override;

public:
  explicit PPCXCOFFMCAsmInfo(bool is64Bit, const Triple &);
  void printSpecifierExpr(raw_ostream &OS,
                          const MCSpecifierExpr &Expr) const override;
  bool evaluateAsRelocatableImpl(const MCSpecifierExpr &Expr, MCValue &Res,
                                 const MCAssembler *Asm) const override;
};

namespace PPC {
enum Specifier {
  S_None,

  S_LO,
  S_HI,
  S_HA,
  S_HIGH,
  S_HIGHA,
  S_HIGHER,
  S_HIGHERA,
  S_HIGHEST,
  S_HIGHESTA,

  S_AIX_TLSGD,       // symbol@gd
  S_AIX_TLSGDM,      // symbol@m
  S_AIX_TLSIE,       // symbol@ie
  S_AIX_TLSLD,       // symbol@ld
  S_AIX_TLSLE,       // symbol@le
  S_AIX_TLSML,       // symbol@ml
  S_DTPMOD,          // symbol@dtpmod
  S_DTPREL,          // symbol@dprel
  S_DTPREL_HA,       // symbol@dtprel@ha
  S_DTPREL_HI,       // symbol@dtprel@h
  S_DTPREL_HIGH,     // symbol@dtprel@high
  S_DTPREL_HIGHA,    // symbol@dtprel@higha
  S_DTPREL_HIGHER,   // symbol@dtprel@higher
  S_DTPREL_HIGHERA,  // symbol@dtprel@highera
  S_DTPREL_HIGHEST,  // symbol@dtprel@highest
  S_DTPREL_HIGHESTA, // symbol@dtprel@highesta
  S_DTPREL_LO,       // symbol@dtprel@l
  S_GOT,             // symbol@got
  S_GOT_DTPREL,      // symbol@got@dtprel
  S_GOT_DTPREL_HA,   // symbol@got@dtprel@ha
  S_GOT_DTPREL_HI,   // symbol@got@dtprel@h
  S_GOT_DTPREL_LO,   // symbol@got@dtprel@l
  S_GOT_HA,          // symbol@got@ha
  S_GOT_HI,          // symbol@got@h
  S_GOT_LO,          // symbol@got@l
  S_GOT_PCREL,       // symbol@got@pcrel
  S_GOT_TLSGD,       // symbol@got@tlsgd
  S_GOT_TLSGD_HA,    // symbol@got@tlsgd@ha
  S_GOT_TLSGD_HI,    // symbol@got@tlsgd@h
  S_GOT_TLSGD_LO,    // symbol@got@tlsgd@l
  S_GOT_TLSGD_PCREL, // symbol@got@tlsgd@pcrel
  S_GOT_TLSLD,       // symbol@got@tlsld
  S_GOT_TLSLD_HA,    // symbol@got@tlsld@ha
  S_GOT_TLSLD_HI,    // symbol@got@tlsld@h
  S_GOT_TLSLD_LO,    // symbol@got@tlsld@l
  S_GOT_TLSLD_PCREL, // symbol@got@tlsld@pcrel
  S_GOT_TPREL,       // symbol@got@tprel
  S_GOT_TPREL_HA,    // symbol@got@tprel@ha
  S_GOT_TPREL_HI,    // symbol@got@tprel@h
  S_GOT_TPREL_LO,    // symbol@got@tprel@l
  S_GOT_TPREL_PCREL, // symbol@got@tprel@pcrel
  S_L,               // symbol@l
  S_LOCAL,           // symbol@local
  S_NOTOC,           // symbol@notoc
  S_PCREL,
  S_PCREL_OPT,      // .reloc expr, R_PPC64_PCREL_OPT, expr
  S_PLT,            // symbol@plt
  S_TLS,            // symbol@tls
  S_TLSGD,          // symbol@tlsgd
  S_TLSLD,          // symbol@tlsld
  S_TLS_PCREL,      // symbol@tls@pcrel
  S_TOC,            // symbol@toc
  S_TOCBASE,        // symbol@tocbase
  S_TOC_HA,         // symbol@toc@ha
  S_TOC_HI,         // symbol@toc@h
  S_TOC_LO,         // symbol@toc@l
  S_TPREL,          // symbol@tprel
  S_TPREL_HA,       // symbol@tprel@ha
  S_TPREL_HI,       // symbol@tprel@h
  S_TPREL_HIGH,     // symbol@tprel@high
  S_TPREL_HIGHA,    // symbol@tprel@higha
  S_TPREL_HIGHER,   // symbol@tprel@higher
  S_TPREL_HIGHERA,  // symbol@tprel@highera
  S_TPREL_HIGHEST,  // symbol@tprel@highest
  S_TPREL_HIGHESTA, // symbol@tprel@highesta
  S_TPREL_LO,       // symbol@tprel@l
  S_U,              // symbol@u
};
}

} // namespace llvm

#endif
