//===- MCSectionWasm.h - Wasm Machine Code Sections -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MCSectionWasm class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCSECTIONWASM_H
#define LLVM_MC_MCSECTIONWASM_H

#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSymbolWasm.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

class MCSymbol;

/// This represents a section on wasm.
class MCSectionWasm final : public MCSection {
  /// This is the name of the section.  The referenced memory is owned by
  /// TargetLoweringObjectFileWasm's WasmUniqueMap.
  StringRef SectionName;

  /// This is the sh_type field of a section, drawn from the enums below.
  unsigned Type;

  /// This is the sh_flags field of a section, drawn from the enums below.
  unsigned Flags;

  unsigned UniqueID;

  const MCSymbolWasm *Group;

private:
  friend class MCContext;
  MCSectionWasm(StringRef Section, unsigned type, unsigned flags, SectionKind K,
                const MCSymbolWasm *group, unsigned UniqueID, MCSymbol *Begin)
      : MCSection(SV_Wasm, K, Begin), SectionName(Section), Type(type),
        Flags(flags), UniqueID(UniqueID), Group(group) {
  }

  void setSectionName(StringRef Name) { SectionName = Name; }

public:
  ~MCSectionWasm();

  /// Decides whether a '.section' directive should be printed before the
  /// section name
  bool ShouldOmitSectionDirective(StringRef Name, const MCAsmInfo &MAI) const;

  StringRef getSectionName() const { return SectionName; }
  unsigned getType() const { return Type; }
  unsigned getFlags() const { return Flags; }
  void setFlags(unsigned F) { Flags = F; }
  const MCSymbolWasm *getGroup() const { return Group; }

  void PrintSwitchToSection(const MCAsmInfo &MAI, const Triple &T,
                            raw_ostream &OS,
                            const MCExpr *Subsection) const override;
  bool UseCodeAlign() const override;
  bool isVirtualSection() const override;

  bool isUnique() const { return UniqueID != ~0U; }
  unsigned getUniqueID() const { return UniqueID; }

  static bool classof(const MCSection *S) { return S->getVariant() == SV_Wasm; }
};

} // end namespace llvm

#endif
