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
  // Temporary labels are not emitted into the object file.
  if (isTemporary())
    return;

  // Do not initialize the attributes multiple times.
  if (hasLDAttributes() || hasERAttributes())
    return;

  GOFF::ESDBindingScope BindingScope =
      isExternal()
          ? (isExported() ? GOFF::ESD_BSC_ImportExport : GOFF::ESD_BSC_Library)
          : GOFF::ESD_BSC_Section;
  GOFF::ESDBindingStrength BindingStrength =
      isWeak() ? GOFF::ESDBindingStrength::ESD_BST_Weak
               : GOFF::ESDBindingStrength::ESD_BST_Strong;

  if (isDefined()) {
    MCSectionGOFF &Section = static_cast<MCSectionGOFF &>(getSection());
    if (Section.isED()) {
      setLDAttributes(GOFF::LDAttr{false, CodeData, BindingStrength,
                                   GOFF::ESD_LT_XPLink, GOFF::ESD_AMODE_64,
                                   BindingScope});
    } else if (Section.isPR()) {
      // For data symbols, the attributes are already determind in TLOFI.
      // TODO Does it make sense to it to here?
    } else
      llvm_unreachable("Unexpected section type for label");
  } else {
    setERAttributes(GOFF::ERAttr{CodeData, BindingStrength, GOFF::ESD_LT_XPLink,
                                 GOFF::ESD_AMODE_64, BindingScope});
  }
}
