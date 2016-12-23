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
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/ObjectYAML/DWARFYAML.h"

#include <algorithm>

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

void dumpDebugInfo(DWARFContextInMemory &DCtx, DWARFYAML::Data &Y) {
  for (const auto &CU : DCtx.compile_units()) {
    DWARFYAML::Unit NewUnit;
    NewUnit.Length = CU->getLength();
    NewUnit.Version = CU->getVersion();
    NewUnit.AbbrOffset = CU->getAbbreviations()->getOffset();
    NewUnit.AddrSize = CU->getAddressByteSize();
    for (auto DIE : CU->dies()) {
      DWARFYAML::Entry NewEntry;
      DataExtractor EntryData = CU->getDebugInfoExtractor();
      uint32_t offset = DIE.getOffset();

      assert(EntryData.isValidOffset(offset) && "Invalid DIE Offset");
      if (!EntryData.isValidOffset(offset))
        continue;

      NewEntry.AbbrCode = EntryData.getULEB128(&offset);

      auto AbbrevDecl = DIE.getAbbreviationDeclarationPtr();
      if (AbbrevDecl) {
        for (const auto &AttrSpec : AbbrevDecl->attributes()) {
          DWARFYAML::FormValue NewValue;
          NewValue.Value = 0xDEADBEEFDEADBEEF;
          DWARFDie DIEWrapper(CU.get(), &DIE);
          auto FormValue = DIEWrapper.getAttributeValue(AttrSpec.Attr);
          if(!FormValue)
            return;
          auto Form = FormValue.getValue().getForm();
          bool indirect = false;
          do {
            indirect = false;
            switch (Form) {
            case dwarf::DW_FORM_addr:
            case dwarf::DW_FORM_GNU_addr_index:
              if (auto Val = FormValue.getValue().getAsAddress())
                NewValue.Value = Val.getValue();
              break;
            case dwarf::DW_FORM_ref_addr:
            case dwarf::DW_FORM_ref1:
            case dwarf::DW_FORM_ref2:
            case dwarf::DW_FORM_ref4:
            case dwarf::DW_FORM_ref8:
            case dwarf::DW_FORM_ref_udata:
            case dwarf::DW_FORM_ref_sig8:
              if (auto Val = FormValue.getValue().getAsReferenceUVal())
                NewValue.Value = Val.getValue();
              break;
            case dwarf::DW_FORM_exprloc:
            case dwarf::DW_FORM_block:
            case dwarf::DW_FORM_block1:
            case dwarf::DW_FORM_block2:
            case dwarf::DW_FORM_block4:
              if (auto Val = FormValue.getValue().getAsBlock()) {
                auto BlockData = Val.getValue();
                std::copy(BlockData.begin(), BlockData.end(),
                          std::back_inserter(NewValue.BlockData));
              }
              NewValue.Value = NewValue.BlockData.size();
              break;
            case dwarf::DW_FORM_data1:
            case dwarf::DW_FORM_flag:
            case dwarf::DW_FORM_data2:
            case dwarf::DW_FORM_data4:
            case dwarf::DW_FORM_data8:
            case dwarf::DW_FORM_sdata:
            case dwarf::DW_FORM_udata:
              if (auto Val = FormValue.getValue().getAsUnsignedConstant())
                NewValue.Value = Val.getValue();
              break;
            case dwarf::DW_FORM_string:
              if (auto Val = FormValue.getValue().getAsCString())
                NewValue.CStr = Val.getValue();
              break;
            case dwarf::DW_FORM_indirect:
              indirect = true;
              if (auto Val = FormValue.getValue().getAsUnsignedConstant()) {
                NewValue.Value = Val.getValue();
                NewEntry.Values.push_back(NewValue);
                Form = static_cast<dwarf::Form>(Val.getValue());
              }
              break;
            case dwarf::DW_FORM_strp:
            case dwarf::DW_FORM_sec_offset:
            case dwarf::DW_FORM_GNU_ref_alt:
            case dwarf::DW_FORM_GNU_strp_alt:
            case dwarf::DW_FORM_line_strp:
            case dwarf::DW_FORM_strp_sup:
            case dwarf::DW_FORM_ref_sup:
            case dwarf::DW_FORM_GNU_str_index:
              if (auto Val = FormValue.getValue().getAsCStringOffset())
                NewValue.Value = Val.getValue();
              break;
            case dwarf::DW_FORM_flag_present:
              NewValue.Value = 1;
              break;
            default:
              break;
            }
          } while (indirect);
          NewEntry.Values.push_back(NewValue);
        }
      }

      NewUnit.Entries.push_back(NewEntry);
    }
    Y.CompileUnits.push_back(NewUnit);
  }
}

std::error_code dwarf2yaml(DWARFContextInMemory &DCtx, DWARFYAML::Data &Y) {
  dumpDebugAbbrev(DCtx, Y);
  dumpDebugStrings(DCtx, Y);
  dumpDebugARanges(DCtx, Y);
  dumpDebugPubSections(DCtx, Y);
  dumpDebugInfo(DCtx, Y);
  return obj2yaml_error::success;
}
