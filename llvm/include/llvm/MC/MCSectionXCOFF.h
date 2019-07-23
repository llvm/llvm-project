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

#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/XCOFF.h"
#include "llvm/MC/MCSection.h"

namespace llvm {

class MCSymbol;

// This class represents an XCOFF `Control Section`, more commonly referred to
// as a csect. A csect represents the smallest possible unit of data/code which
// will be relocated as a single block. A csect can either be:
// 1) Initialized: The Type will be XTY_SD, and the symbols inside the csect
//    will have a label definition representing their offset within the csect.
// 2) Uninitialized: The Type will be XTY_CM, it will contain a single symbol,
//    and may not contain label definitions.
class MCSectionXCOFF final : public MCSection {
  friend class MCContext;

  StringRef Name;
  XCOFF::StorageMappingClass MappingClass;
  XCOFF::SymbolType Type;

  MCSectionXCOFF(StringRef Section, XCOFF::StorageMappingClass SMC,
                 XCOFF::SymbolType ST, SectionKind K, MCSymbol *Begin)
      : MCSection(SV_XCOFF, K, Begin), Name(Section), MappingClass(SMC),
        Type(ST) {
    assert((ST == XCOFF::XTY_SD || ST == XCOFF::XTY_CM) &&
           "Unexpected type for csect.");
  }

public:
  ~MCSectionXCOFF();

  static bool classof(const MCSection *S) {
    return S->getVariant() == SV_XCOFF;
  }

  StringRef getSectionName() const { return Name; }
  XCOFF::StorageMappingClass getMappingClass() const { return MappingClass; }
  XCOFF::SymbolType getCSectType() const { return Type; }

  void PrintSwitchToSection(const MCAsmInfo &MAI, const Triple &T,
                            raw_ostream &OS,
                            const MCExpr *Subsection) const override;
  bool UseCodeAlign() const override;
  bool isVirtualSection() const override;
};

} // end namespace llvm

#endif
