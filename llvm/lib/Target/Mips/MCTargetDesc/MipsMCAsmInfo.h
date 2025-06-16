//===-- MipsMCAsmInfo.h - Mips Asm Info ------------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MipsMCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MIPS_MCTARGETDESC_MIPSMCASMINFO_H
#define LLVM_LIB_TARGET_MIPS_MCTARGETDESC_MIPSMCASMINFO_H

#include "MCTargetDesc/MipsMCExpr.h"
#include "llvm/MC/MCAsmInfoCOFF.h"
#include "llvm/MC/MCAsmInfoELF.h"
#include "llvm/MC/MCFixup.h"

namespace llvm {
class Triple;

class MipsELFMCAsmInfo : public MCAsmInfoELF {
  void anchor() override;

public:
  explicit MipsELFMCAsmInfo(const Triple &TheTriple,
                            const MCTargetOptions &Options);
  void printSpecifierExpr(raw_ostream &OS,
                          const MCSpecifierExpr &Expr) const override;
  bool evaluateAsRelocatableImpl(const MCSpecifierExpr &Expr, MCValue &Res,
                                 const MCAssembler *Asm) const override;
};

class MipsCOFFMCAsmInfo : public MCAsmInfoGNUCOFF {
  void anchor() override;

public:
  explicit MipsCOFFMCAsmInfo();
  void printSpecifierExpr(raw_ostream &OS,
                          const MCSpecifierExpr &Expr) const override;
  bool evaluateAsRelocatableImpl(const MCSpecifierExpr &Expr, MCValue &Res,
                                 const MCAssembler *Asm) const override;
};

namespace Mips {
using Specifier = uint16_t;
enum {
  S_None,
  S_CALL_HI16 = FirstTargetFixupKind,
  S_CALL_LO16,
  S_DTPREL,
  S_DTPREL_HI,
  S_DTPREL_LO,
  S_GOT,
  S_GOTTPREL,
  S_GOT_CALL,
  S_GOT_DISP,
  S_GOT_HI16,
  S_GOT_LO16,
  S_GOT_OFST,
  S_GOT_PAGE,
  S_GPREL,
  S_HI,
  S_HIGHER,
  S_HIGHEST,
  S_LO,
  S_NEG,
  S_PCREL_HI16,
  S_PCREL_LO16,
  S_TLSGD,
  S_TLSLDM,
  S_TPREL_HI,
  S_TPREL_LO,
  S_Special,
};

bool isGpOff(const MCSpecifierExpr &E);
}

} // namespace llvm

#endif
