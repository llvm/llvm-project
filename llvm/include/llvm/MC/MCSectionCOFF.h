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

#include "llvm/MC/MCSection.h"
#include "llvm/Support/COFF.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

/// MCSectionCOFF - This represents a section on Windows
  class MCSectionCOFF : public MCSection {
    // The memory for this string is stored in the same MCContext as *this.
    StringRef SectionName;

    /// Characteristics - This is the Characteristics field of a section,
    //  drawn from the enums below.
    unsigned Characteristics;

    /// Selection - This is the Selection field for the section symbol, if
    /// it is a COMDAT section (Characteristics & IMAGE_SCN_LNK_COMDAT) != 0
    int Selection;

  private:
    friend class MCContext;
    MCSectionCOFF(StringRef Section, unsigned Characteristics,
                  int Selection, SectionKind K)
      : MCSection(SV_COFF, K), SectionName(Section),
        Characteristics(Characteristics), Selection (Selection) {
      assert ((Characteristics & 0x00F00000) == 0 &&
        "alignment must not be set upon section creation");
    }
    ~MCSectionCOFF();

  public:
    /// ShouldOmitSectionDirective - Decides whether a '.section' directive
    /// should be printed before the section name
    bool ShouldOmitSectionDirective(StringRef Name, const MCAsmInfo &MAI) const;

    StringRef getSectionName() const { return SectionName; }
    unsigned getCharacteristics() const { return Characteristics; }
    int getSelection () const { return Selection; }

    virtual void PrintSwitchToSection(const MCAsmInfo &MAI,
                                      raw_ostream &OS) const;
    virtual bool UseCodeAlign() const;
    virtual bool isVirtualSection() const;

    static bool classof(const MCSection *S) {
      return S->getVariant() == SV_COFF;
    }
  };

} // end namespace llvm

#endif
