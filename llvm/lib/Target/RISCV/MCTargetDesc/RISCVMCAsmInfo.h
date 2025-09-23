//===-- RISCVMCAsmInfo.h - RISC-V Asm Info ---------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the RISCVMCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_MCTARGETDESC_RISCVMCASMINFO_H
#define LLVM_LIB_TARGET_RISCV_MCTARGETDESC_RISCVMCASMINFO_H

#include "llvm/MC/MCAsmInfoELF.h"
#include "llvm/MC/MCFixup.h"

namespace llvm {
class Triple;

class RISCVMCAsmInfo : public MCAsmInfoELF {
  void anchor() override;

public:
  explicit RISCVMCAsmInfo(const Triple &TargetTriple);

  const MCExpr *getExprForFDESymbol(const MCSymbol *Sym, unsigned Encoding,
                                    MCStreamer &Streamer) const override;
  void printSpecifierExpr(raw_ostream &OS,
                          const MCSpecifierExpr &Expr) const override;
};

namespace RISCV {
using Specifier = uint16_t;
// Specifiers mapping to relocation types below FirstTargetFixupKind are
// encoded literally, with these exceptions:
enum {
  S_None,
  // Specifiers mapping to distinct relocation types.
  S_LO = FirstTargetFixupKind,
  S_PCREL_LO,
  S_TPREL_LO,
  // Vendor-specific relocation types might conflict across vendors.
  // Refer to them using Specifier constants.
  S_QC_ABS20,
};

Specifier parseSpecifierName(StringRef name);
StringRef getSpecifierName(Specifier Kind);
} // namespace RISCV

} // namespace llvm

#endif
