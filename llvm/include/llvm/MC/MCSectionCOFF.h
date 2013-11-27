//===- MCSectionCOFF.h - COFF Machine Code Sections -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MCSectionCOFF class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCSECTIONCOFF_H
#define LLVM_MC_MCSECTIONCOFF_H

#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCSection.h"
#include "llvm/Support/COFF.h"

namespace llvm {
class MCSymbol;

/// MCSectionCOFF - This represents a section on Windows
  class MCSectionCOFF : public MCSection {
    // The memory for this string is stored in the same MCContext as *this.
    StringRef SectionName;

    // FIXME: The following fields should not be mutable, but are for now so
    // the asm parser can honor the .linkonce directive.

    /// Characteristics - This is the Characteristics field of a section,
    /// drawn from the enums below.
    mutable unsigned Characteristics;

    /// The COMDAT symbol of this section. Only valid if this is a COMDAT
    /// section. Two COMDAT sections are merged if they have the same
    /// COMDAT symbol.
    const MCSymbol *COMDATSymbol;

    /// Selection - This is the Selection field for the section symbol, if
    /// it is a COMDAT section (Characteristics & IMAGE_SCN_LNK_COMDAT) != 0
    mutable int Selection;

    /// Assoc - This is name of the associated section, if it is a COMDAT
    /// section (Characteristics & IMAGE_SCN_LNK_COMDAT) != 0 with an
    /// associative Selection (IMAGE_COMDAT_SELECT_ASSOCIATIVE).
    mutable const MCSectionCOFF *Assoc;

  private:
    friend class MCContext;
    MCSectionCOFF(StringRef Section, unsigned Characteristics,
                  const MCSymbol *COMDATSymbol, int Selection,
                  const MCSectionCOFF *Assoc, SectionKind K)
        : MCSection(SV_COFF, K), SectionName(Section),
          Characteristics(Characteristics), COMDATSymbol(COMDATSymbol),
          Selection(Selection), Assoc(Assoc) {
      assert ((Characteristics & 0x00F00000) == 0 &&
        "alignment must not be set upon section creation");
      assert ((Selection == COFF::IMAGE_COMDAT_SELECT_ASSOCIATIVE) ==
              (Assoc != 0) &&
        "associative COMDAT section must have an associated section");
    }
    ~MCSectionCOFF();

  public:
    /// ShouldOmitSectionDirective - Decides whether a '.section' directive
    /// should be printed before the section name
    bool ShouldOmitSectionDirective(StringRef Name, const MCAsmInfo &MAI) const;

    StringRef getSectionName() const { return SectionName; }
    virtual std::string getLabelBeginName() const {
      return SectionName.str() + "_begin";
    }
    virtual std::string getLabelEndName() const {
      return SectionName.str() + "_end";
    }
    unsigned getCharacteristics() const { return Characteristics; }
    int getSelection() const { return Selection; }
    const MCSectionCOFF *getAssocSection() const { return Assoc; }

    void setSelection(int Selection, const MCSectionCOFF *Assoc = 0) const;

    virtual void PrintSwitchToSection(const MCAsmInfo &MAI,
                                      raw_ostream &OS,
                                      const MCExpr *Subsection) const;
    virtual bool UseCodeAlign() const;
    virtual bool isVirtualSection() const;

    static bool classof(const MCSection *S) {
      return S->getVariant() == SV_COFF;
    }
  };

} // end namespace llvm

#endif
