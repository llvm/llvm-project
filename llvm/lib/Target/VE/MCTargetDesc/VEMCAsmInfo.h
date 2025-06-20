//===- VEMCAsmInfo.h - VE asm properties -----------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the VEMCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_VE_MCTARGETDESC_VEMCASMINFO_H
#define LLVM_LIB_TARGET_VE_MCTARGETDESC_VEMCASMINFO_H

#include "llvm/MC/MCAsmInfoELF.h"

namespace llvm {

class Triple;

class VEELFMCAsmInfo : public MCAsmInfoELF {
  void anchor() override;

public:
  explicit VEELFMCAsmInfo(const Triple &TheTriple);
  void printSpecifierExpr(raw_ostream &OS,
                          const MCSpecifierExpr &Expr) const override;
  bool evaluateAsRelocatableImpl(const MCSpecifierExpr &Expr, MCValue &Res,
                                 const MCAssembler *Asm) const override;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_VE_MCTARGETDESC_VEMCASMINFO_H
