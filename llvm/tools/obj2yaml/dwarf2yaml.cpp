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

std::error_code dwarf2yaml(DWARFContextInMemory &DCtx,
                           DWARFYAML::Data &Y) {
  dumpDebugAbbrev(DCtx, Y);
  dumpDebugStrings(DCtx, Y);
  dumpDebugARanges(DCtx, Y);

  return obj2yaml_error::success;
}
