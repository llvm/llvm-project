//===- DWARFYAML.cpp - DWARF YAMLIO implementation ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines classes for handling the YAML representation of DWARF Debug
// Info.
//
//===----------------------------------------------------------------------===//

#include "llvm/ObjectYAML/DWARFYAML.h"

namespace llvm {

bool DWARFYAML::Data::isEmpty() const {
  return 0 == DebugStrings.size() + AbbrevDecls.size();
}

namespace yaml {

void MappingTraits<DWARFYAML::Data>::mapping(
    IO &IO, DWARFYAML::Data &DWARF) {
  IO.mapOptional("debug_str", DWARF.DebugStrings);
  IO.mapOptional("debug_abbrev", DWARF.AbbrevDecls);
  if(!DWARF.ARanges.empty() || !IO.outputting())
    IO.mapOptional("debug_aranges", DWARF.ARanges);
}

void MappingTraits<DWARFYAML::Abbrev>::mapping(
    IO &IO, DWARFYAML::Abbrev &Abbrev) {
  IO.mapRequired("Code", Abbrev.Code);
  IO.mapRequired("Tag", Abbrev.Tag);
  IO.mapRequired("Children", Abbrev.Children);
  IO.mapRequired("Attributes", Abbrev.Attributes);
}

void MappingTraits<DWARFYAML::AttributeAbbrev>::mapping(
    IO &IO, DWARFYAML::AttributeAbbrev &AttAbbrev) {
  IO.mapRequired("Attribute", AttAbbrev.Attribute);
  IO.mapRequired("Form", AttAbbrev.Form);
}

void MappingTraits<DWARFYAML::ARangeDescriptor>::mapping(
    IO &IO, DWARFYAML::ARangeDescriptor &Descriptor) {
  IO.mapRequired("Address", Descriptor.Address);
  IO.mapRequired("Length", Descriptor.Length);
}

void MappingTraits<DWARFYAML::ARange>::mapping(IO &IO,
                                                DWARFYAML::ARange &Range) {
  IO.mapRequired("Length", Range.Length);
  IO.mapRequired("Version", Range.Version);
  IO.mapRequired("CuOffset", Range.CuOffset);
  IO.mapRequired("AddrSize", Range.AddrSize);
  IO.mapRequired("SegSize", Range.SegSize);
  IO.mapRequired("Descriptors", Range.Descriptors);
}

} // namespace llvm::yaml

} // namespace llvm
