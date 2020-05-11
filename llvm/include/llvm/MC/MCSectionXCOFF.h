//===- MCSectionXCOFF.h - XCOFF Machine Code Sections -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the MCSectionXCOFF class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCSECTIONXCOFF_H
#define LLVM_MC_MCSECTIONXCOFF_H

#include "llvm/BinaryFormat/XCOFF.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSymbolXCOFF.h"

namespace llvm {

// This class represents an XCOFF `Control Section`, more commonly referred to
// as a csect. A csect represents the smallest possible unit of data/code which
// will be relocated as a single block. A csect can either be:
// 1) Initialized: The Type will be XTY_SD, and the symbols inside the csect
//    will have a label definition representing their offset within the csect.
// 2) Uninitialized: The Type will be XTY_CM, it will contain a single symbol,
//    and may not contain label definitions.
// 3) An external reference providing a symbol table entry for a symbol
//    contained in another XCOFF object file. External reference csects are not
//    implemented yet.
class MCSectionXCOFF final : public MCSection {
  friend class MCContext;

  XCOFF::StorageMappingClass MappingClass;
  XCOFF::SymbolType Type;
  XCOFF::StorageClass StorageClass;
  MCSymbolXCOFF *const QualName;
  static constexpr unsigned DefaultAlignVal = 4;

  MCSectionXCOFF(StringRef Name, XCOFF::StorageMappingClass SMC,
                 XCOFF::SymbolType ST, XCOFF::StorageClass SC, SectionKind K,
                 MCSymbolXCOFF *QualName, MCSymbol *Begin)
      : MCSection(SV_XCOFF, Name, K, Begin), MappingClass(SMC), Type(ST),
        StorageClass(SC), QualName(QualName) {
    assert((ST == XCOFF::XTY_SD || ST == XCOFF::XTY_CM || ST == XCOFF::XTY_ER) &&
           "Invalid or unhandled type for csect.");
    assert(QualName != nullptr && "QualName is needed.");
    QualName->setStorageClass(SC);
    QualName->setRepresentedCsect(this);
    // A csect is 4 byte aligned by default, except for undefined symbol csects.
    if (Type != XCOFF::XTY_ER)
      setAlignment(Align(DefaultAlignVal));
  }

  void printCsectDirective(raw_ostream &OS) const;

public:
  ~MCSectionXCOFF();

  static bool classof(const MCSection *S) {
    return S->getVariant() == SV_XCOFF;
  }

  XCOFF::StorageMappingClass getMappingClass() const { return MappingClass; }
  XCOFF::StorageClass getStorageClass() const { return StorageClass; }
  XCOFF::SymbolType getCSectType() const { return Type; }
  MCSymbolXCOFF *getQualNameSymbol() const { return QualName; }

  void PrintSwitchToSection(const MCAsmInfo &MAI, const Triple &T,
                            raw_ostream &OS,
                            const MCExpr *Subsection) const override;
  bool UseCodeAlign() const override;
  bool isVirtualSection() const override;
};

} // end namespace llvm

#endif
