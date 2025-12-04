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

bool MCSymbolGOFF::setSymbolAttribute(MCSymbolAttr Attribute) {
  switch (Attribute) {
  case MCSA_Invalid:
  case MCSA_Cold:
  case MCSA_ELF_TypeIndFunction:
  case MCSA_ELF_TypeTLS:
  case MCSA_ELF_TypeCommon:
  case MCSA_ELF_TypeNoType:
  case MCSA_ELF_TypeGnuUniqueObject:
  case MCSA_LGlobal:
  case MCSA_Extern:
  case MCSA_Exported:
  case MCSA_IndirectSymbol:
  case MCSA_Internal:
  case MCSA_LazyReference:
  case MCSA_NoDeadStrip:
  case MCSA_SymbolResolver:
  case MCSA_AltEntry:
  case MCSA_PrivateExtern:
  case MCSA_Protected:
  case MCSA_Reference:
  case MCSA_WeakDefinition:
  case MCSA_WeakDefAutoPrivate:
  case MCSA_WeakAntiDep:
  case MCSA_Memtag:
    return false;

  case MCSA_ELF_TypeFunction:
    setCodeData(GOFF::ESDExecutable::ESD_EXE_CODE);
    break;
  case MCSA_ELF_TypeObject:
    setCodeData(GOFF::ESDExecutable::ESD_EXE_DATA);
    break;
  case MCSA_OSLinkage:
    setLinkage(GOFF::ESDLinkageType::ESD_LT_OS);
    break;
  case MCSA_XPLinkage:
    setLinkage(GOFF::ESDLinkageType::ESD_LT_XPLink);
    break;
  case MCSA_Global:
    setExternal(true);
    break;
  case MCSA_Local:
    setExternal(false);
    break;
  case MCSA_Weak:
  case MCSA_WeakReference:
    setExternal(true);
    setWeak();
    break;
  case MCSA_Hidden:
    setHidden(true);
    break;
  }

  return true;
}
