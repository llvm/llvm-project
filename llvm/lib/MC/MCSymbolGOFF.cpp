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

void MCSymbolGOFF::initAttributes() {
  if (hasLDAttributes())
    return;

  if (isDefined()) {
    MCSectionGOFF &Section = static_cast<MCSectionGOFF &>(getSection());
    GOFF::ESDBindingScope BindingScope =
        isExternal() ? (isExported() ? GOFF::ESD_BSC_ImportExport
                                     : GOFF::ESD_BSC_Library)
                     : GOFF::ESD_BSC_Section;
    GOFF::ESDBindingStrength BindingStrength =
        isWeak() ? GOFF::ESDBindingStrength::ESD_BST_Weak
                 : GOFF::ESDBindingStrength::ESD_BST_Strong;
    if (Section.isED()) {
      setLDAttributes(GOFF::LDAttr{false, GOFF::ESD_EXE_CODE, BindingStrength,
                                   GOFF::ESD_LT_XPLink, GOFF::ESD_AMODE_64,
                                   BindingScope});
    } else if (Section.isPR()) {
      // For data symbols, the attributes are already determind in TLOFI.
      // TODO Does it make sense to it to here?
    } else
      llvm_unreachable("Unexpected section type for label");
  }
  // TODO Handle external symbol.
}
