//=-- LoongArchTargetObjectFile.cpp - LoongArch Object Info -------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LoongArchTargetObjectFile.h"
#include "MCTargetDesc/LoongArchMCAsmInfo.h"

using namespace llvm;

const MCExpr *LoongArchELFTargetObjectFile::getDebugThreadLocalSymbol(
    const MCSymbol *Sym) const {
  return MCSpecifierExpr::create(MCSymbolRefExpr::create(Sym, getContext()),
                                 LoongArchMCExpr::VK_DTPREL, getContext());
}
