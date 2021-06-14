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
  dumpHeader();
  return dumpSymbols();
}

void XCOFFDumper::dumpHeader() {

  YAMLObj.Header.Magic = Obj.getMagic();
  YAMLObj.Header.NumberOfSections = Obj.getNumberOfSections();
  YAMLObj.Header.TimeStamp = Obj.getTimeStamp();

  // TODO FIXME only dump 32 bit header for now.
  if (Obj.is64Bit())
    report_fatal_error("64-bit XCOFF files not supported yet.");
  YAMLObj.Header.SymbolTableOffset = Obj.getSymbolTableOffset32();

  YAMLObj.Header.NumberOfSymTableEntries =
      Obj.getRawNumberOfSymbolTableEntries32();
  YAMLObj.Header.AuxHeaderSize = Obj.getOptionalHeaderSize();
  YAMLObj.Header.Flags = Obj.getFlags();
}

std::error_code XCOFFDumper::dumpSymbols() {
  std::vector<XCOFFYAML::Symbol> &Symbols = YAMLObj.Symbols;

  for (const SymbolRef &S : Obj.symbols()) {
    DataRefImpl SymbolDRI = S.getRawDataRefImpl();
    const XCOFFSymbolRef SymbolEntRef = Obj.toSymbolRef(SymbolDRI);
    XCOFFYAML::Symbol Sym;

    Expected<StringRef> SymNameRefOrErr = Obj.getSymbolName(SymbolDRI);
    if (!SymNameRefOrErr) {
      return errorToErrorCode(SymNameRefOrErr.takeError());
    }
    Sym.SymbolName = SymNameRefOrErr.get();

    Sym.Value = SymbolEntRef.getValue();

    Expected<StringRef> SectionNameRefOrErr =
        Obj.getSymbolSectionName(SymbolEntRef);
    if (!SectionNameRefOrErr)
      return errorToErrorCode(SectionNameRefOrErr.takeError());

    Sym.SectionName = SectionNameRefOrErr.get();

    Sym.Type = SymbolEntRef.getSymbolType();
    Sym.StorageClass = SymbolEntRef.getStorageClass();
    Sym.NumberOfAuxEntries = SymbolEntRef.getNumberOfAuxEntries();
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
