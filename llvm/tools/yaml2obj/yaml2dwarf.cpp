//===- yaml2dwarf - Convert YAML to DWARF binary data ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief The DWARF component of yaml2obj.
///
//===----------------------------------------------------------------------===//

#include "llvm/ObjectYAML/DWARFYAML.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

void ZeroFillBytes(raw_ostream &OS, size_t Size) {
  std::vector<uint8_t> FillData;
  FillData.insert(FillData.begin(), Size, 0);
  OS.write(reinterpret_cast<char *>(FillData.data()), Size);
}

void yaml2debug_str(raw_ostream &OS, const DWARFYAML::Data &DI) {
  for (auto Str : DI.DebugStrings) {
    OS.write(Str.data(), Str.size());
    OS.write('\0');
  }
}

void yaml2debug_abbrev(raw_ostream &OS, const DWARFYAML::Data &DI) {
  for (auto AbbrevDecl : DI.AbbrevDecls) {
    encodeULEB128(AbbrevDecl.Code, OS);
    encodeULEB128(AbbrevDecl.Tag, OS);
    OS.write(AbbrevDecl.Children);
    for (auto Attr : AbbrevDecl.Attributes) {
      encodeULEB128(Attr.Attribute, OS);
      encodeULEB128(Attr.Form, OS);
    }
    encodeULEB128(0, OS);
    encodeULEB128(0, OS);
  }
}

void yaml2debug_aranges(raw_ostream &OS, const DWARFYAML::Data &DI) {
  for (auto Range : DI.ARanges) {
    auto HeaderStart = OS.tell();
    OS.write(reinterpret_cast<char *>(&Range.Length), 4);
    OS.write(reinterpret_cast<char *>(&Range.Version), 2);
    OS.write(reinterpret_cast<char *>(&Range.CuOffset), 4);
    OS.write(reinterpret_cast<char *>(&Range.AddrSize), 1);
    OS.write(reinterpret_cast<char *>(&Range.SegSize), 1);

    auto HeaderSize = OS.tell() - HeaderStart;
    auto FirstDescriptor = alignTo(HeaderSize, Range.AddrSize * 2);
    ZeroFillBytes(OS, FirstDescriptor - HeaderSize);

    for (auto Descriptor : Range.Descriptors) {
      OS.write(reinterpret_cast<char *>(&Descriptor.Address), Range.AddrSize);
      OS.write(reinterpret_cast<char *>(&Descriptor.Length), Range.AddrSize);
    }
    ZeroFillBytes(OS, Range.AddrSize * 2);
  }
}

void yaml2pubsection(raw_ostream &OS, const DWARFYAML::PubSection &Sect) {
  OS.write(reinterpret_cast<const char *>(&Sect.Length), 4);
  OS.write(reinterpret_cast<const char *>(&Sect.Version), 2);
  OS.write(reinterpret_cast<const char *>(&Sect.UnitOffset), 4);
  OS.write(reinterpret_cast<const char *>(&Sect.UnitSize), 4);
  for (auto Entry : Sect.Entries) {
    OS.write(reinterpret_cast<const char *>(&Entry.DieOffset), 4);
    if (Sect.IsGNUStyle)
      OS.write(reinterpret_cast<const char *>(&Entry.Descriptor), 4);
    OS.write(Entry.Name.data(), Entry.Name.size());
    OS.write('\0');
  }
}
