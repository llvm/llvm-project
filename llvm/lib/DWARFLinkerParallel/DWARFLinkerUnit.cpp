//===- DWARFLinkerUnit.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DWARFLinkerUnit.h"
#include "DWARFEmitterImpl.h"
#include "DebugLineSectionEmitter.h"

namespace llvm {
namespace dwarflinker_parallel {

void DwarfUnit::assignAbbrev(DIEAbbrev &Abbrev) {
  // Check the set for priors.
  FoldingSetNodeID ID;
  Abbrev.Profile(ID);
  void *InsertToken;

  DIEAbbrev *InSet = AbbreviationsSet.FindNodeOrInsertPos(ID, InsertToken);
  // If it's newly added.
  if (InSet) {
    // Assign existing abbreviation number.
    Abbrev.setNumber(InSet->getNumber());
  } else {
    // Add to abbreviation list.
    Abbreviations.push_back(
        std::make_unique<DIEAbbrev>(Abbrev.getTag(), Abbrev.hasChildren()));
    for (const auto &Attr : Abbrev.getData())
      Abbreviations.back()->AddAttribute(Attr);
    AbbreviationsSet.InsertNode(Abbreviations.back().get(), InsertToken);
    // Assign the unique abbreviation number.
    Abbrev.setNumber(Abbreviations.size());
    Abbreviations.back()->setNumber(Abbreviations.size());
  }
}

Error DwarfUnit::emitAbbreviations() {
  const std::vector<std::unique_ptr<DIEAbbrev>> &Abbrevs = getAbbreviations();
  if (Abbrevs.empty())
    return Error::success();

  SectionDescriptor &AbbrevSection =
      getOrCreateSectionDescriptor(DebugSectionKind::DebugAbbrev);

  // For each abbreviation.
  for (const auto &Abbrev : Abbrevs)
    emitDwarfAbbrevEntry(*Abbrev, AbbrevSection);

  // Mark end of abbreviations.
  encodeULEB128(0, AbbrevSection.OS);

  return Error::success();
}

void DwarfUnit::emitDwarfAbbrevEntry(const DIEAbbrev &Abbrev,
                                     SectionDescriptor &AbbrevSection) {
  // Emit the abbreviations code (base 1 index.)
  encodeULEB128(Abbrev.getNumber(), AbbrevSection.OS);

  // Emit the abbreviations data.
  // Emit its Dwarf tag type.
  encodeULEB128(Abbrev.getTag(), AbbrevSection.OS);

  // Emit whether it has children DIEs.
  encodeULEB128((unsigned)Abbrev.hasChildren(), AbbrevSection.OS);

  // For each attribute description.
  const SmallVectorImpl<DIEAbbrevData> &Data = Abbrev.getData();
  for (unsigned i = 0, N = Data.size(); i < N; ++i) {
    const DIEAbbrevData &AttrData = Data[i];

    // Emit attribute type.
    encodeULEB128(AttrData.getAttribute(), AbbrevSection.OS);

    // Emit form type.
    encodeULEB128(AttrData.getForm(), AbbrevSection.OS);

    // Emit value for DW_FORM_implicit_const.
    if (AttrData.getForm() == dwarf::DW_FORM_implicit_const)
      encodeSLEB128(AttrData.getValue(), AbbrevSection.OS);
  }

  // Mark end of abbreviation.
  encodeULEB128(0, AbbrevSection.OS);
  encodeULEB128(0, AbbrevSection.OS);
}

Error DwarfUnit::emitDebugInfo(Triple &TargetTriple) {
  DIE *OutUnitDIE = getOutUnitDIE();
  if (OutUnitDIE == nullptr)
    return Error::success();

  // FIXME: Remove dependence on DwarfEmitterImpl/AsmPrinter and emit DIEs
  // directly.

  SectionDescriptor &OutSection =
      getOrCreateSectionDescriptor(DebugSectionKind::DebugInfo);
  DwarfEmitterImpl Emitter(DWARFLinker::OutputFileType::Object, OutSection.OS);
  if (Error Err = Emitter.init(TargetTriple, "__DWARF"))
    return Err;

  // Emit compile unit header.
  Emitter.emitCompileUnitHeader(*this);
  size_t OffsetToAbbreviationTableOffset =
      (getFormParams().Version >= 5) ? 8 : 6;
  OutSection.notePatch(DebugOffsetPatch{
      OffsetToAbbreviationTableOffset,
      &getOrCreateSectionDescriptor(DebugSectionKind::DebugAbbrev)});

  // Emit DIEs.
  Emitter.emitDIE(*OutUnitDIE);
  Emitter.finish();

  // Set start offset ans size for .debug_info section.
  OutSection.setSizesForSectionCreatedByAsmPrinter();
  return Error::success();
}

Error DwarfUnit::emitDebugLine(Triple &TargetTriple,
                               const DWARFDebugLine::LineTable &OutLineTable) {
  DebugLineSectionEmitter DebugLineEmitter(TargetTriple, *this);

  return DebugLineEmitter.emit(OutLineTable);
}

} // end of namespace dwarflinker_parallel
} // end of namespace llvm
