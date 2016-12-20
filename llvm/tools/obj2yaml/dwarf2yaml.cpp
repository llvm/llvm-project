//===------ dwarf2yaml.cpp - obj2yaml conversion tool -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Error.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugArangeSet.h"
#include "llvm/ObjectYAML/DWARFYAML.h"

using namespace llvm;

void dumpDebugAbbrev(DWARFContextInMemory &DCtx, DWARFYAML::Data &Y) {
  auto AbbrevSetPtr = DCtx.getDebugAbbrev();
  if (AbbrevSetPtr) {
    for (auto AbbrvDeclSet : *AbbrevSetPtr) {
      for (auto AbbrvDecl : AbbrvDeclSet.second) {
        DWARFYAML::Abbrev Abbrv;
        Abbrv.Code = AbbrvDecl.getCode();
        Abbrv.Tag = AbbrvDecl.getTag();
        Abbrv.Children = AbbrvDecl.hasChildren() ? dwarf::DW_CHILDREN_yes
                                                 : dwarf::DW_CHILDREN_no;
        for (auto Attribute : AbbrvDecl.attributes()) {
          DWARFYAML::AttributeAbbrev AttAbrv;
          AttAbrv.Attribute = Attribute.Attr;
          AttAbrv.Form = Attribute.Form;
          Abbrv.Attributes.push_back(AttAbrv);
        }
        Y.AbbrevDecls.push_back(Abbrv);
      }
    }
  }
}

void dumpDebugStrings(DWARFContextInMemory &DCtx, DWARFYAML::Data &Y) {
  StringRef RemainingTable = DCtx.getStringSection();
  while (RemainingTable.size() > 0) {
    auto SymbolPair = RemainingTable.split('\0');
    RemainingTable = SymbolPair.second;
    Y.DebugStrings.push_back(SymbolPair.first);
  }
}

void dumpDebugARanges(DWARFContextInMemory &DCtx, DWARFYAML::Data &Y) {
  DataExtractor ArangesData(DCtx.getARangeSection(), DCtx.isLittleEndian(), 0);
  uint32_t Offset = 0;
  DWARFDebugArangeSet Set;

  while (Set.extract(ArangesData, &Offset)) {
    DWARFYAML::ARange Range;
    Range.Length = Set.getHeader().Length;
    Range.Version = Set.getHeader().Version;
    Range.CuOffset = Set.getHeader().CuOffset;
    Range.AddrSize = Set.getHeader().AddrSize;
    Range.SegSize = Set.getHeader().SegSize;
    for (auto Descriptor : Set.descriptors()) {
      DWARFYAML::ARangeDescriptor Desc;
      Desc.Address = Descriptor.Address;
      Desc.Length = Descriptor.Length;
      Range.Descriptors.push_back(Desc);
    }
    Y.ARanges.push_back(Range);
  }
}

void dumpPubSection(DWARFContextInMemory &DCtx, DWARFYAML::PubSection &Y,
                    StringRef Section) {
  DataExtractor PubSectionData(Section, DCtx.isLittleEndian(), 0);
  uint32_t Offset = 0;
  Y.Length = PubSectionData.getU32(&Offset);
  Y.Version = PubSectionData.getU16(&Offset);
  Y.UnitOffset = PubSectionData.getU32(&Offset);
  Y.UnitSize = PubSectionData.getU32(&Offset);
  while (Offset < Y.Length) {
    DWARFYAML::PubEntry NewEntry;
    NewEntry.DieOffset = PubSectionData.getU32(&Offset);
    if (Y.IsGNUStyle)
      NewEntry.Descriptor = PubSectionData.getU8(&Offset);
    NewEntry.Name = PubSectionData.getCStr(&Offset);
    Y.Entries.push_back(NewEntry);
  }
}

void dumpDebugPubSections(DWARFContextInMemory &DCtx, DWARFYAML::Data &Y) {
  Y.PubNames.IsGNUStyle = false;
  dumpPubSection(DCtx, Y.PubNames, DCtx.getPubNamesSection());

  Y.PubTypes.IsGNUStyle = false;
  dumpPubSection(DCtx, Y.PubTypes, DCtx.getPubTypesSection());

  Y.GNUPubNames.IsGNUStyle = true;
  dumpPubSection(DCtx, Y.GNUPubNames, DCtx.getGnuPubNamesSection());

  Y.GNUPubTypes.IsGNUStyle = true;
  dumpPubSection(DCtx, Y.GNUPubTypes, DCtx.getGnuPubTypesSection());
}

std::error_code dwarf2yaml(DWARFContextInMemory &DCtx,
                           DWARFYAML::Data &Y) {
  dumpDebugAbbrev(DCtx, Y);
  dumpDebugStrings(DCtx, Y);
  dumpDebugARanges(DCtx, Y);
  dumpDebugPubSections(DCtx, Y);

  return obj2yaml_error::success;
}
