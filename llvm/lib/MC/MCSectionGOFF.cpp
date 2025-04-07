//===- MCSectionGOFF.cpp - GOFF Code Section Representation ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCSectionGOFF.h"
#include "llvm/BinaryFormat/GOFF.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
void emitRMode(raw_ostream &OS, GOFF::ESDRmode Rmode, bool UseParenthesis) {
  if (Rmode != GOFF::ESD_RMODE_None) {
    OS << "RMODE" << (UseParenthesis ? '(' : ' ');
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
    case GOFF::ESD_RMODE_None:
      break;
    }
    if (UseParenthesis)
      OS << ')';
  }
}

void emitCATTR(raw_ostream &OS, StringRef Name, StringRef ParentName,
               bool EmitAmodeAndRmode, GOFF::ESDAmode Amode,
               GOFF::ESDRmode Rmode, GOFF::ESDAlignment Alignment,
               GOFF::ESDLoadingBehavior LoadBehavior,
               GOFF::ESDExecutable Executable, bool IsReadOnly,
               StringRef PartName) {
  if (EmitAmodeAndRmode && Amode != GOFF::ESD_AMODE_None) {
    OS << ParentName << " AMODE ";
    switch (Amode) {
    case GOFF::ESD_AMODE_24:
      OS << "24";
      break;
    case GOFF::ESD_AMODE_31:
      OS << "31";
      break;
    case GOFF::ESD_AMODE_ANY:
      OS << "ANY";
      break;
    case GOFF::ESD_AMODE_64:
      OS << "64";
      break;
    case GOFF::ESD_AMODE_MIN:
      OS << "ANY64";
      break;
    case GOFF::ESD_AMODE_None:
      break;
    }
    OS << "\n";
  }
  if (EmitAmodeAndRmode && Rmode != GOFF::ESD_RMODE_None) {
    OS << ParentName << ' ';
    emitRMode(OS, Rmode, /*UseParenthesis=*/false);
    OS << "\n";
  }
  OS << Name << " CATTR ";
  OS << "ALIGN(" << static_cast<unsigned>(Alignment) << ")";
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
    emitRMode(OS, Rmode, /*UseParenthesis=*/true);
  }
  if (!PartName.empty())
    OS << ",PART(" << PartName << ")";
  OS << '\n';
}
} // namespace

void MCSectionGOFF::printSwitchToSection(const MCAsmInfo &MAI, const Triple &T,
                                         raw_ostream &OS,
                                         uint32_t Subsection) const {
  switch (SymbolType) {
  case GOFF::ESD_ST_SectionDefinition: {
    OS << Name << " CSECT\n";
    Emitted = true;
    break;
  }
  case GOFF::ESD_ST_ElementDefinition: {
    bool ParentEmitted = getParent()->Emitted;
    getParent()->printSwitchToSection(MAI, T, OS, Subsection);
    if (!Emitted) {
      emitCATTR(OS, Name, getParent()->getName(), !ParentEmitted,
                EDAttributes.Amode, EDAttributes.Rmode, EDAttributes.Alignment,
                EDAttributes.LoadBehavior, EDAttributes.Executable,
                EDAttributes.IsReadOnly, StringRef());
      Emitted = true;
    } else
      OS << Name << " CATTR ,\n";
    break;
  }
  case GOFF::ESD_ST_PartReference: {
    MCSectionGOFF *ED = getParent();
    bool SDEmitted = ED->getParent()->Emitted;
    ED->getParent()->printSwitchToSection(MAI, T, OS, Subsection);
    if (!Emitted) {
      emitCATTR(OS, ED->getName(), ED->getParent()->getName(), !SDEmitted,
                PRAttributes.Amode, getParent()->EDAttributes.Rmode,
                PRAttributes.Alignment, getParent()->EDAttributes.LoadBehavior,
                PRAttributes.Executable, PRAttributes.IsReadOnly, Name);
      ED->Emitted = true;
      Emitted = true;
    } else
      OS << ED->getName() << " CATTR ,\n";
    break;
  }
  default:
    llvm_unreachable("Wrong section type");
  }
}