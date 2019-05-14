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

public:
  XCOFFDumper(const object::XCOFFObjectFile &obj);
  XCOFFYAML::Object &getYAMLObj() { return YAMLObj; }
};
} // namespace

XCOFFDumper::XCOFFDumper(const object::XCOFFObjectFile &obj) : Obj(obj) {
  dumpHeader();
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

std::error_code xcoff2yaml(raw_ostream &Out,
                           const object::XCOFFObjectFile &Obj) {
  XCOFFDumper Dumper(Obj);
  yaml::Output Yout(Out);
  Yout << Dumper.getYAMLObj();

  return std::error_code();
}
