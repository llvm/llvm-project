//===-- SystemZTargetObjectFile.cpp - SystemZ Object Info -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SystemZTargetObjectFile.h"
#include "MCTargetDesc/SystemZMCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

const MCExpr *SystemZELFTargetObjectFile::getDebugThreadLocalSymbol(
    const MCSymbol *Sym) const {
  return MCSymbolRefExpr::create(Sym, SystemZ::S_DTPOFF, getContext());
}
