//===- MCAsmInfoGOFF.cpp - MCGOFFAsmInfo properties -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines certain target specific asm properties for GOFF (z/OS)
/// based targets.
///
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAsmInfoGOFF.h"
#include "llvm/BinaryFormat/GOFF.h"
#include "llvm/MC/MCSectionGOFF.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

MCAsmInfoGOFF::MCAsmInfoGOFF() {
  Data64bitsDirective = "\t.quad\t";
  HasDotTypeDotSizeDirective = false;
  PrivateGlobalPrefix = "L#";
  PrivateLabelPrefix = "L#";
  ZeroDirective = "\t.space\t";
}

static void emitCATTR(raw_ostream &OS, StringRef Name, GOFF::ESDRmode Rmode,
                      GOFF::ESDAlignment Alignment,
                      GOFF::ESDLoadingBehavior LoadBehavior,
                      GOFF::ESDExecutable Executable, bool IsReadOnly,
                      uint32_t SortKey, uint8_t FillByteValue,
                      StringRef PartName) {
  OS << Name << " CATTR ";
  OS << "ALIGN(" << static_cast<unsigned>(Alignment) << "),"
     << "FILL(" << static_cast<unsigned>(FillByteValue) << ")";
  switch (LoadBehavior) {
  case GOFF::ESD_LB_Deferred:
    OS << ",DEFLOAD";
    break;
  case GOFF::ESD_LB_NoLoad:
    OS << ",NOLOAD";
    break;
  default:
    break;
  }
  switch (Executable) {
  case GOFF::ESD_EXE_CODE:
    OS << ",EXECUTABLE";
    break;
  case GOFF::ESD_EXE_DATA:
    OS << ",NOTEXECUTABLE";
    break;
  default:
    break;
  }
  if (IsReadOnly)
    OS << ",READONLY";
  if (Rmode != GOFF::ESD_RMODE_None) {
    OS << ',';
    OS << "RMODE(";
    switch (Rmode) {
    case GOFF::ESD_RMODE_24:
      OS << "24";
      break;
    case GOFF::ESD_RMODE_31:
      OS << "31";
      break;
    case GOFF::ESD_RMODE_64:
      OS << "64";
      break;
    }
    OS << ')';
  }
  if (SortKey)
    OS << ",PRIORITY(" << SortKey << ")";
  if (!PartName.empty())
    OS << ",PART(" << PartName << ")";
  OS << '\n';
}

static void emitXATTR(raw_ostream &OS, StringRef Name,
                      GOFF::ESDLinkageType Linkage,
                      GOFF::ESDExecutable Executable,
                      GOFF::ESDBindingScope BindingScope) {
  OS << Name << " XATTR ";
  OS << "LINKAGE(" << (Linkage == GOFF::ESD_LT_OS ? "OS" : "XPLINK") << "),";
  if (Executable != GOFF::ESD_EXE_Unspecified)
    OS << "REFERENCE(" << (Executable == GOFF::ESD_EXE_CODE ? "CODE" : "DATA")
       << "),";
  if (BindingScope != GOFF::ESD_BSC_Unspecified) {
    OS << "SCOPE(";
    switch (BindingScope) {
    case GOFF::ESD_BSC_Section:
      OS << "SECTION";
      break;
    case GOFF::ESD_BSC_Module:
      OS << "MODULE";
      break;
    case GOFF::ESD_BSC_Library:
      OS << "LIBRARY";
      break;
    case GOFF::ESD_BSC_ImportExport:
      OS << "EXPORT";
      break;
    default:
      break;
    }
    OS << ')';
  }
  OS << '\n';
}

void MCAsmInfoGOFF::printSwitchToSection(const MCSection &Section,
                                         uint32_t Subsection, const Triple &T,
                                         raw_ostream &OS) const {
  auto &Sec =
      const_cast<MCSectionGOFF &>(static_cast<const MCSectionGOFF &>(Section));
  switch (Sec.SymbolType) {
  case GOFF::ESD_ST_SectionDefinition: {
    OS << Sec.getName() << " CSECT\n";
    Sec.Emitted = true;
    break;
  }
  case GOFF::ESD_ST_ElementDefinition: {
    printSwitchToSection(*Sec.getParent(), Subsection, T, OS);
    if (!Sec.Emitted) {
      emitCATTR(OS, Sec.getName(), Sec.EDAttributes.Rmode,
                Sec.EDAttributes.Alignment, Sec.EDAttributes.LoadBehavior,
                GOFF::ESD_EXE_Unspecified, Sec.EDAttributes.IsReadOnly, 0,
                Sec.EDAttributes.FillByteValue, StringRef());
      Sec.Emitted = true;
    } else
      OS << Sec.getName() << " CATTR\n";
    break;
  }
  case GOFF::ESD_ST_PartReference: {
    MCSectionGOFF *ED = Sec.getParent();
    printSwitchToSection(*ED->getParent(), Subsection, T, OS);
    if (!Sec.Emitted) {
      emitCATTR(OS, ED->getName(), ED->getEDAttributes().Rmode,
                ED->EDAttributes.Alignment, ED->EDAttributes.LoadBehavior,
                Sec.PRAttributes.Executable, ED->EDAttributes.IsReadOnly,
                Sec.PRAttributes.SortKey, ED->EDAttributes.FillByteValue,
                Sec.getName());
      emitXATTR(OS, Sec.getName(), Sec.PRAttributes.Linkage,
                Sec.PRAttributes.Executable, Sec.PRAttributes.BindingScope);
      ED->Emitted = true;
      Sec.Emitted = true;
    } else
      OS << ED->getName() << " CATTR PART(" << Sec.getName() << ")\n";
    break;
  }
  default:
    llvm_unreachable("Wrong section type");
  }
}
