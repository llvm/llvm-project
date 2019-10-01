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
#include "llvm/Support/YAMLTraits.h"

using namespace llvm;
using namespace llvm::object;
namespace {

class XCOFFDumper {
  const object::XCOFFObjectFile &Obj;
  XCOFFYAML::Object YAMLObj;
  void dumpHeader();
  std::error_code dumpSymbols();

public:
  XCOFFDumper(const object::XCOFFObjectFile &obj) : Obj(obj) {}
  std::error_code dump();
  XCOFFYAML::Object &getYAMLObj() { return YAMLObj; }
};
} // namespace

std::error_code XCOFFDumper::dump() {
  std::error_code EC;
  dumpHeader();
  EC = dumpSymbols();
  return EC;
}

void XCOFFDumper::dumpHeader() {
  const XCOFFFileHeader *FileHdrPtr = Obj.getFileHeader();

  YAMLObj.Header.Magic = FileHdrPtr->Magic;
  YAMLObj.Header.NumberOfSections = FileHdrPtr->NumberOfSections;
  YAMLObj.Header.TimeStamp = FileHdrPtr->TimeStamp;
  YAMLObj.Header.SymbolTableOffset = FileHdrPtr->SymbolTableOffset;
  YAMLObj.Header.NumberOfSymTableEntries = FileHdrPtr->NumberOfSymTableEntries;
  YAMLObj.Header.AuxHeaderSize = FileHdrPtr->AuxHeaderSize;
  YAMLObj.Header.Flags = FileHdrPtr->Flags;
}

std::error_code XCOFFDumper::dumpSymbols() {
  std::vector<XCOFFYAML::Symbol> &Symbols = YAMLObj.Symbols;

  for (const SymbolRef &S : Obj.symbols()) {
    DataRefImpl SymbolDRI = S.getRawDataRefImpl();
    const XCOFFSymbolEntry *SymbolEntPtr = Obj.toSymbolEntry(SymbolDRI);
    XCOFFYAML::Symbol Sym;

    Expected<StringRef> SymNameRefOrErr = Obj.getSymbolName(SymbolDRI);
    if (!SymNameRefOrErr) {
      return errorToErrorCode(SymNameRefOrErr.takeError());
    }
    Sym.SymbolName = SymNameRefOrErr.get();

    Sym.Value = SymbolEntPtr->Value;

    Expected<StringRef> SectionNameRefOrErr =
        Obj.getSymbolSectionName(SymbolEntPtr);
    if (!SectionNameRefOrErr)
      return errorToErrorCode(SectionNameRefOrErr.takeError());

    Sym.SectionName = SectionNameRefOrErr.get();

    Sym.Type = SymbolEntPtr->SymbolType;
    Sym.StorageClass = SymbolEntPtr->StorageClass;
    Sym.NumberOfAuxEntries = SymbolEntPtr->NumberOfAuxEntries;
    Symbols.push_back(Sym);
  }

  return std::error_code();
}

std::error_code xcoff2yaml(raw_ostream &Out,
                           const object::XCOFFObjectFile &Obj) {
  XCOFFDumper Dumper(Obj);

  if (std::error_code EC = Dumper.dump())
    return EC;

  yaml::Output Yout(Out);
  Yout << Dumper.getYAMLObj();

  return std::error_code();
}
