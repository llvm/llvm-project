//===-- PPCMCExpr.h - PPC specific MC expression classes --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_POWERPC_MCTARGETDESC_PPCMCEXPR_H
#define LLVM_LIB_TARGET_POWERPC_MCTARGETDESC_PPCMCEXPR_H

#include "PPCMCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCValue.h"
#include <optional>

namespace llvm {

namespace PPCMCExpr {
using Specifier = uint16_t;
}

static inline PPCMCExpr::Specifier getSpecifier(const MCSymbolRefExpr *SRE) {
  return PPCMCExpr::Specifier(SRE->getKind());
}

namespace PPC {
bool evaluateAsConstant(const MCSpecifierExpr &Expr, int64_t &Res);
bool evaluateAsRelocatableImpl(const MCSpecifierExpr &Expr, MCValue &Res,
                               const MCAssembler *Asm);
} // namespace PPC

} // end namespace llvm

#endif
