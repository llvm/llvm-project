//===- MCSymbolGOFF.cpp - GOFF Symbol Representation ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCSymbolGOFF.h"
#include "llvm/BinaryFormat/GOFF.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

GOFF::ESDBindingScope MCSymbolGOFF::getBindingScope() const {
  return isExternal() ? (isExported() ? GOFF::ESD_BSC_ImportExport
                                      : GOFF::ESD_BSC_Library)
                      : GOFF::ESD_BSC_Section;
}

bool MCSymbolGOFF::hasLDAttributes() const {
  return !isTemporary() && isDefined() &&
         static_cast<MCSectionGOFF &>(getSection()).isED();
}

bool MCSymbolGOFF::hasERAttributes() const {
  return !isTemporary() && !isDefined() && isExternal();
}
