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

bool MCSymbolGOFF::hasLDAttributes() const {
  return !isTemporary() && isDefined() &&
         static_cast<MCSectionGOFF &>(getSection()).isED();
}

GOFF::LDAttr MCSymbolGOFF::getLDAttributes() const {
  assert(hasLDAttributes() && "Symbol does not have LD attributes");

  GOFF::ESDBindingScope BindingScope =
      isExternal()
          ? (isExported() ? GOFF::ESD_BSC_ImportExport : GOFF::ESD_BSC_Library)
          : GOFF::ESD_BSC_Section;
  GOFF::ESDBindingStrength BindingStrength =
      isWeak() ? GOFF::ESDBindingStrength::ESD_BST_Weak
               : GOFF::ESDBindingStrength::ESD_BST_Strong;
  return GOFF::LDAttr{false,   CodeData,           BindingStrength,
                      Linkage, GOFF::ESD_AMODE_64, BindingScope};
}

bool MCSymbolGOFF::hasERAttributes() const {
  return !isTemporary() && !isDefined() && isExternal();
}

GOFF::ERAttr MCSymbolGOFF::getERAttributes() const {
  assert(hasERAttributes() && "Symbol does not have ER attributes");

  GOFF::ESDBindingScope BindingScope =
      isExternal()
          ? (isExported() ? GOFF::ESD_BSC_ImportExport : GOFF::ESD_BSC_Library)
          : GOFF::ESD_BSC_Section;
  GOFF::ESDBindingStrength BindingStrength =
      isWeak() ? GOFF::ESDBindingStrength::ESD_BST_Weak
               : GOFF::ESDBindingStrength::ESD_BST_Strong;
  return GOFF::ERAttr{CodeData, BindingStrength, Linkage, GOFF::ESD_AMODE_64,
                      BindingScope};
}
