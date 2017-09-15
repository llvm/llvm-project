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
private:
  /// This is the name of the section.  The referenced memory is owned by
  /// TargetLoweringObjectFileWasm's WasmUniqueMap.
  StringRef SectionName;

  /// This is the type of the section, from the enums in BinaryFormat/Wasm.h
  unsigned Type;

  unsigned UniqueID;

  const MCSymbolWasm *Group;

  // The offset of the MC function/data section in the wasm code/data section.
  // For data relocations the offset is relative to start of the data payload
  // itself and does not include the size of the section header.
  uint64_t SectionOffset;

  // For data sections, this is the offset of the corresponding wasm data
  // segment
  uint64_t MemoryOffset;

  friend class MCContext;
  MCSectionWasm(StringRef Section, unsigned type, SectionKind K,
                const MCSymbolWasm *group, unsigned UniqueID, MCSymbol *Begin)
      : MCSection(SV_Wasm, K, Begin), SectionName(Section), Type(type),
        UniqueID(UniqueID), Group(group), SectionOffset(0) {
    assert(type == wasm::WASM_SEC_CODE || type == wasm::WASM_SEC_DATA);
  }

  void setSectionName(StringRef Name) { SectionName = Name; }

public:
  ~MCSectionWasm();

  /// Decides whether a '.section' directive should be printed before the
  /// section name
  bool ShouldOmitSectionDirective(StringRef Name, const MCAsmInfo &MAI) const;

  StringRef getSectionName() const { return SectionName; }
  unsigned getType() const { return Type; }
  const MCSymbolWasm *getGroup() const { return Group; }

  void PrintSwitchToSection(const MCAsmInfo &MAI, const Triple &T,
                            raw_ostream &OS,
                            const MCExpr *Subsection) const override;
  bool UseCodeAlign() const override;
  bool isVirtualSection() const override;

  bool isUnique() const { return UniqueID != ~0U; }
  unsigned getUniqueID() const { return UniqueID; }

  uint64_t getSectionOffset() const { return SectionOffset; }
  void setSectionOffset(uint64_t Offset) { SectionOffset = Offset; }

  uint32_t getMemoryOffset() const { return MemoryOffset; }
  void setMemoryOffset(uint32_t Offset) { MemoryOffset = Offset; }

  static bool classof(const MCSection *S) { return S->getVariant() == SV_Wasm; }
};

} // end namespace llvm

#endif
