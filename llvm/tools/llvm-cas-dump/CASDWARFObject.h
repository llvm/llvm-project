//===-----------------------------------------------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_CAS_DUMP_CASDWARFOBJECT_H
#define LLVM_TOOLS_LLVM_CAS_DUMP_CASDWARFOBJECT_H

#include "llvm/DebugInfo/DWARF/DWARFCompileUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugAbbrev.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLine.h"
#include "llvm/DebugInfo/DWARF/DWARFObject.h"
#include "llvm/MCCAS/MCCASObjectV1.h"

namespace llvm {

/// A DWARFObject implementation that just supports enough to
/// dwarfdump section contents.
class CASDWARFObject : public DWARFObject {
  bool Is64Bit = true;
  bool IsLittleEndian = true;
  Triple Target;
  SmallVector<uint8_t> DebugStringSection;
  SmallVector<uint8_t> DebugAbbrevSection;
  SmallVector<char> DebugInfoSection;
  SmallVector<StringRef> CUDataVec;
  SmallVector<char> DebugLineSection;
  DenseMap<cas::ObjectRef, unsigned> MapOfStringOffsets;

  const mccasformats::v1::MCSchema &Schema;

  /// Function to get the MCObjectProxy for a ObjectRef \p CASObj and pass it to
  /// discoverDwarfSections(mccasformats::v1::MCObjectProxy MCObj);
  Error discoverDwarfSections(cas::ObjectRef CASObj);

public:
  CASDWARFObject(const mccasformats::v1::MCSchema &Schema) : Schema(Schema) {}

  /// Copy contents of any dwarf sections that are interesting for dwarfdump to
  /// work.
  Error discoverDwarfSections(mccasformats::v1::MCObjectProxy MCObj);

  /// Dump MCObj as textual DWARF output.
  Error dump(raw_ostream &OS, int Indent, DWARFContext &DWARFCtx,
             mccasformats::v1::MCObjectProxy MCObj, bool Verbose);

  StringRef getFileName() const override { return "CAS"; }
  ArrayRef<SectionName> getSectionNames() const override { return {}; }
  bool isLittleEndian() const override { return IsLittleEndian; }
  uint8_t getAddressSize() const override { return Is64Bit ? 8 : 4; }
  StringRef getAbbrevSection() const override {
    return toStringRef(DebugAbbrevSection);
  }
  StringRef getStrSection() const override {
    return llvm::toStringRef(DebugStringSection);
  }
  std::optional<RelocAddrEntry> find(const DWARFSection &Sec,
                                uint64_t Pos) const override {
    return {};
  }
  DenseMap<cas::ObjectRef, unsigned> &getMapOfStringOffsets() {
    return MapOfStringOffsets;
  }
};
} // namespace llvm

#endif
