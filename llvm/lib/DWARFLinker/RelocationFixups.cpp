//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DWARFLinker/RelocationFixups.h"
#include "llvm/ADT/STLExtras.h"

using namespace llvm;
using namespace dwarf_linker;

void RelocationFixups::noteAttribute(dwarf::Tag Tag, dwarf::Attribute Attr,
                                     uint64_t InputStart, uint64_t InputEnd,
                                     uint64_t OutputOffset,
                                     const AttributeSource &Source) {
  if (!Source.Expression) {
    DIEFixups.push_back({RelocSectionKind::DebugInfo, InputStart, InputEnd,
                         int64_t(OutputOffset) - int64_t(InputStart)});
  } else {
    uint64_t ExpressionStart = InputEnd - Source.Expression->InputSize;
    for (const ClonedExpression::Operation &Op :
         Source.Expression->Operations) {
      uint64_t Start = Op.InputStart;
      if (Op.InputSection == RelocSectionKind::DebugInfo)
        Start += ExpressionStart;
      DIEFixups.push_back(
          {Op.InputSection, Start, Start + (Op.InputEnd - Op.InputStart),
           int64_t(OutputOffset + Op.OutputOffset) - int64_t(Start)});
    }
  }

  // A compile unit's low_pc is computed rather than taken from its entry. A
  // high_pc can end where the next symbol starts, in the entry of that symbol's
  // low_pc, which the relocation of the high_pc would misdescribe.
  bool IsLowPC = Attr == dwarf::DW_AT_low_pc;
  if (Source.Entry && Attr != dwarf::DW_AT_high_pc &&
      !(IsLowPC && Tag == dwarf::DW_TAG_compile_unit))
    DebugAddrFixups.push_back({*Source.Entry, IsLowPC});
}

void RelocationFixups::finishDIE(uint64_t AbbrevNumberSize) {
  for (DebugInfoFixup &Fixup : DIEFixups) {
    Fixup.LinkedOffset += AbbrevNumberSize;
    DebugInfoFixups.push_back(Fixup);
  }
  DIEFixups.clear();
}

void RelocationFixups::save(AddressesMap &Addresses, uint64_t UnitOffset,
                            uint8_t AddrSize) const {
  for (const DebugInfoFixup &Fixup : DebugInfoFixups)
    Addresses.updateAndSaveValidRelocs(
        Fixup.InputSection, RelocSectionKind::DebugInfo, UnitOffset,
        Fixup.LinkedOffset, Fixup.InputStart, Fixup.InputEnd);

  if (!DebugAddrTableOffset)
    return;

  // Entries holding the same address are merged, so symbols can share one: a
  // DW_AT_call_return_pc ending a function holds the low_pc of the next. Only
  // one relocation can describe an entry. The low_pc's names the right symbol.
  SmallVector<DebugAddrFixup> Sorted(DebugAddrFixups);
  llvm::stable_sort(
      Sorted, [](const DebugAddrFixup &LHS, const DebugAddrFixup &RHS) {
        return std::make_pair(LHS.Entry.OutputIndex, !LHS.IsLowPC) <
               std::make_pair(RHS.Entry.OutputIndex, !RHS.IsLowPC);
      });

  std::optional<uint64_t> SavedIndex;
  for (const DebugAddrFixup &Fixup : Sorted) {
    if (SavedIndex == Fixup.Entry.OutputIndex)
      continue;
    uint64_t InputOffset = Fixup.Entry.InputOffset;
    uint64_t OutputOffset =
        *DebugAddrTableOffset + Fixup.Entry.OutputIndex * AddrSize;
    if (Addresses.updateAndSaveValidRelocs(
            RelocSectionKind::DebugAddr, RelocSectionKind::DebugAddr,
            UnitOffset, int64_t(OutputOffset) - int64_t(InputOffset),
            InputOffset, InputOffset + AddrSize))
      SavedIndex = Fixup.Entry.OutputIndex;
  }
}

void RelocationFixups::clear() {
  DIEFixups.clear();
  DebugInfoFixups.clear();
  DebugAddrFixups.clear();
  DebugAddrTableOffset.reset();
}
