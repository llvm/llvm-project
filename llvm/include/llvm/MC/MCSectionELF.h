//===- MCSectionELF.h - ELF Machine Code Sections ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the MCSectionELF class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCSECTIONELF_H
#define LLVM_MC_MCSECTIONELF_H

#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/MC/SectionKind.h"

namespace llvm {

class MCSymbol;

/// This represents a section on linux, lots of unix variants and some bare
/// metal systems.
class MCSectionELF final : public MCSection {
  /// This is the name of the section.  The referenced memory is owned by
  /// TargetLoweringObjectFileELF's ELFUniqueMap.
  StringRef SectionName;

  /// This is the sh_type field of a section, drawn from the enums below.
  unsigned Type;

  /// This is the sh_flags field of a section, drawn from the enums below.
  unsigned Flags;

  unsigned UniqueID;

  /// The size of each entry in this section. This size only makes sense for
  /// sections that contain fixed-sized entries. If a section does not contain
  /// fixed-sized entries 'EntrySize' will be 0.
  unsigned EntrySize;

  const MCSymbolELF *Group;

  /// Used by SHF_LINK_ORDER. If non-null, the sh_link field will be set to the
  /// section header index of the section where LinkedToSym is defined.
  const MCSymbol *LinkedToSym;

private:
  friend class MCContext;

  MCSectionELF(StringRef Section, unsigned type, unsigned flags, SectionKind K,
               unsigned entrySize, const MCSymbolELF *group, unsigned UniqueID,
               MCSymbol *Begin, const MCSymbolELF *LinkedToSym)
      : MCSection(SV_ELF, K, Begin), SectionName(Section), Type(type),
        Flags(flags), UniqueID(UniqueID), EntrySize(entrySize), Group(group),
        LinkedToSym(LinkedToSym) {
    if (Group)
      Group->setIsSignature();
  }

  void setSectionName(StringRef Name) { SectionName = Name; }

public:
  /// Decides whether a '.section' directive should be printed before the
  /// section name
  bool ShouldOmitSectionDirective(StringRef Name, const MCAsmInfo &MAI) const;

  StringRef getSectionName() const { return SectionName; }
  unsigned getType() const { return Type; }
  unsigned getFlags() const { return Flags; }
  unsigned getEntrySize() const { return EntrySize; }
  void setFlags(unsigned F) { Flags = F; }
  const MCSymbolELF *getGroup() const { return Group; }

  void PrintSwitchToSection(const MCAsmInfo &MAI, const Triple &T,
                            raw_ostream &OS,
                            const MCExpr *Subsection) const override;
  bool UseCodeAlign() const override;
  bool isVirtualSection() const override;

  bool isUnique() const { return UniqueID != ~0U; }
  unsigned getUniqueID() const { return UniqueID; }

  const MCSection *getLinkedToSection() const {
    return &LinkedToSym->getSection();
  }
  const MCSymbol *getLinkedToSymbol() const { return LinkedToSym; }

  static bool classof(const MCSection *S) {
    return S->getVariant() == SV_ELF;
  }
};

} // end namespace llvm

#endif // LLVM_MC_MCSECTIONELF_H
