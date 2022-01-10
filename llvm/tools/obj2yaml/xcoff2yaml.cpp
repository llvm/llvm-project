//===------ xcoff2yaml.cpp - XCOFF YAMLIO implementation --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "obj2yaml.h"
#include "llvm/Object/XCOFFObjectFile.h"
#include "llvm/ObjectYAML/XCOFFYAML.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/YAMLTraits.h"

using namespace llvm;
using namespace llvm::object;
namespace {

class XCOFFDumper {
  const object::XCOFFObjectFile &Obj;
  XCOFFYAML::Object YAMLObj;
  void dumpHeader();
  Error dumpSections();
  Error dumpSymbols();
  template <typename Shdr, typename Reloc>
  Error dumpSections(ArrayRef<Shdr> Sections);

public:
  XCOFFDumper(const object::XCOFFObjectFile &obj) : Obj(obj) {}
  Error dump();
  XCOFFYAML::Object &getYAMLObj() { return YAMLObj; }
};
} // namespace

Error XCOFFDumper::dump() {
  dumpHeader();
  if (Error E = dumpSections())
    return E;
  return dumpSymbols();
}

void XCOFFDumper::dumpHeader() {
  YAMLObj.Header.Magic = Obj.getMagic();
  YAMLObj.Header.NumberOfSections = Obj.getNumberOfSections();
  YAMLObj.Header.TimeStamp = Obj.getTimeStamp();
  YAMLObj.Header.SymbolTableOffset = Obj.is64Bit()
                                         ? Obj.getSymbolTableOffset64()
                                         : Obj.getSymbolTableOffset32();
  YAMLObj.Header.NumberOfSymTableEntries =
      Obj.is64Bit() ? Obj.getNumberOfSymbolTableEntries64()
                    : Obj.getRawNumberOfSymbolTableEntries32();
  YAMLObj.Header.AuxHeaderSize = Obj.getOptionalHeaderSize();
  YAMLObj.Header.Flags = Obj.getFlags();
}

Error XCOFFDumper::dumpSections() {
  if (Obj.is64Bit())
    return dumpSections<XCOFFSectionHeader64, XCOFFRelocation64>(
        Obj.sections64());
  return dumpSections<XCOFFSectionHeader32, XCOFFRelocation32>(
      Obj.sections32());
}

template <typename Shdr, typename Reloc>
Error XCOFFDumper::dumpSections(ArrayRef<Shdr> Sections) {
  std::vector<XCOFFYAML::Section> &YamlSections = YAMLObj.Sections;
  for (const Shdr &S : Sections) {
    XCOFFYAML::Section YamlSec;
    YamlSec.SectionName = S.getName();
    YamlSec.Address = S.PhysicalAddress;
    YamlSec.Size = S.SectionSize;
    YamlSec.NumberOfRelocations = S.NumberOfRelocations;
    YamlSec.NumberOfLineNumbers = S.NumberOfLineNumbers;
    YamlSec.FileOffsetToData = S.FileOffsetToRawData;
    YamlSec.FileOffsetToRelocations = S.FileOffsetToRelocationInfo;
    YamlSec.FileOffsetToLineNumbers = S.FileOffsetToLineNumberInfo;
    YamlSec.Flags = S.Flags;

    // Dump section data.
    if (S.FileOffsetToRawData) {
      DataRefImpl SectionDRI;
      SectionDRI.p = reinterpret_cast<uintptr_t>(&S);
      Expected<ArrayRef<uint8_t>> SecDataRefOrErr =
          Obj.getSectionContents(SectionDRI);
      if (!SecDataRefOrErr)
        return SecDataRefOrErr.takeError();
      YamlSec.SectionData = SecDataRefOrErr.get();
    }

    // Dump relocations.
    if (S.NumberOfRelocations) {
      auto RelRefOrErr = Obj.relocations<Shdr, Reloc>(S);
      if (!RelRefOrErr)
        return RelRefOrErr.takeError();
      for (const Reloc &R : RelRefOrErr.get()) {
        XCOFFYAML::Relocation YamlRel;
        YamlRel.Type = R.Type;
        YamlRel.Info = R.Info;
        YamlRel.SymbolIndex = R.SymbolIndex;
        YamlRel.VirtualAddress = R.VirtualAddress;
        YamlSec.Relocations.push_back(YamlRel);
      }
    }
    YamlSections.push_back(YamlSec);
  }
  return Error::success();
}

Error XCOFFDumper::dumpSymbols() {
  std::vector<XCOFFYAML::Symbol> &Symbols = YAMLObj.Symbols;

  for (const SymbolRef &S : Obj.symbols()) {
    DataRefImpl SymbolDRI = S.getRawDataRefImpl();
    const XCOFFSymbolRef SymbolEntRef = Obj.toSymbolRef(SymbolDRI);
    XCOFFYAML::Symbol Sym;

    Expected<StringRef> SymNameRefOrErr = Obj.getSymbolName(SymbolDRI);
    if (!SymNameRefOrErr) {
      return SymNameRefOrErr.takeError();
    }
    Sym.SymbolName = SymNameRefOrErr.get();

    Sym.Value = SymbolEntRef.getValue();

    Expected<StringRef> SectionNameRefOrErr =
        Obj.getSymbolSectionName(SymbolEntRef);
    if (!SectionNameRefOrErr)
      return SectionNameRefOrErr.takeError();

    Sym.SectionName = SectionNameRefOrErr.get();

    Sym.Type = SymbolEntRef.getSymbolType();
    Sym.StorageClass = SymbolEntRef.getStorageClass();
    Sym.NumberOfAuxEntries = SymbolEntRef.getNumberOfAuxEntries();

    Symbols.push_back(std::move(Sym));
  }

  return Error::success();
}

Error xcoff2yaml(raw_ostream &Out, const object::XCOFFObjectFile &Obj) {
  XCOFFDumper Dumper(Obj);

  if (Error E = Dumper.dump())
    return E;

  yaml::Output Yout(Out);
  Yout << Dumper.getYAMLObj();

  return Error::success();
}
