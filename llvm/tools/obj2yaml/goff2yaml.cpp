//===------ goff2yaml.cpp - obj2yaml conversion tool ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "obj2yaml.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Object/GOFFObjectFile.h"
#include "llvm/ObjectYAML/ObjectYAML.h"
#include "llvm/Support/ConvertEBCDIC.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

static std::string getFixedLengthEBCDICString(DataExtractor &Data,
                                              DataExtractor::Cursor &C,
                                              uint64_t Length,
                                              StringRef TrimChars = {"\0", 1}) {
  StringRef FixedLenStr = Data.getBytes(C, Length);
  SmallString<16> Str;
  ConverterEBCDIC::convertToUTF8(FixedLenStr, Str);
  return Str.str().trim(TrimChars).str();
}

class GOFFDumper {
  const object::GOFFObjectFile &Obj;
  GOFFYAML::Object YAMLObj;

  Error dumpHeader(ArrayRef<uint8_t> Records);
  Error dumpExternalSymbol(ArrayRef<uint8_t> Records);
  Error dumpText(ArrayRef<uint8_t> Records);
  Error dumpRelocationDirectory(ArrayRef<uint8_t> Records);
  Error dumpDeferredLength(ArrayRef<uint8_t> Records);
  Error dumpEnd(ArrayRef<uint8_t> Records);

public:
  GOFFDumper(const object::GOFFObjectFile &Obj);
  Error dump();
  GOFFYAML::Object &getYAMLObj();
};

GOFFDumper::GOFFDumper(const object::GOFFObjectFile &Obj) : Obj(Obj) {}

Error GOFFDumper::dumpHeader(ArrayRef<uint8_t> Records) {
  DataExtractor Data(Records, false, 0);
  DataExtractor::Cursor C(4);
  YAMLObj.Header.TargetEnvironment = Data.getU32(C);
  YAMLObj.Header.TargetOperatingSystem = Data.getU32(C);
  Data.skip(C, 2);
  YAMLObj.Header.CCSID = Data.getU16(C);
  YAMLObj.Header.CharacterSetName = getFixedLengthEBCDICString(Data, C, 16);
  YAMLObj.Header.LanguageProductIdentifier =
      getFixedLengthEBCDICString(Data, C, 16);
  YAMLObj.Header.ArchitectureLevel = Data.getU32(C);
  uint16_t PropertiesLength = Data.getU16(C);
  Data.skip(C, 6);
  if (PropertiesLength) {
    YAMLObj.Header.InternalCCSID = Data.getU16(C);
    PropertiesLength -= 2;
  }
  if (PropertiesLength) {
    YAMLObj.Header.TargetSoftwareEnvironment = Data.getU8(C);
    PropertiesLength -= 1;
  }
  return C.takeError();
}

Error GOFFDumper::dumpExternalSymbol(ArrayRef<uint8_t> Records) {
  // TODO Implement.
  return Error::success();
}

Error GOFFDumper::dumpText(ArrayRef<uint8_t> Records) {
  // TODO Implement.
  return Error::success();
}

Error GOFFDumper::dumpRelocationDirectory(ArrayRef<uint8_t> Records) {
  // TODO Implement.
  return Error::success();
}

Error GOFFDumper::dumpDeferredLength(ArrayRef<uint8_t> Records) {
  // TODO Implement.
  return Error::success();
}

Error GOFFDumper::dumpEnd(ArrayRef<uint8_t> Records) {
  // TODO Implement.
  return Error::success();
}

Error GOFFDumper::dump() {
  Error Err = Error::success();
  for (auto &Rec : Obj.records(&Err)) {
    if (Err)
      return Err;
    switch (Rec.getRecordType()) {
    case GOFF::RT_ESD:
      if (auto Err = dumpExternalSymbol(Rec.getContents()))
        return Err;
      break;
    case GOFF::RT_TXT:
      if (auto Err = dumpText(Rec.getContents()))
        return Err;
      break;
    case GOFF::RT_RLD:
      if (auto Err = dumpRelocationDirectory(Rec.getContents()))
        return Err;
      break;
    case GOFF::RT_LEN:
      if (auto Err = dumpDeferredLength(Rec.getContents()))
        return Err;
      break;
    case GOFF::RT_END:
      if (auto Err = dumpEnd(Rec.getContents()))
        return Err;
      break;
    case GOFF::RT_HDR:
      if (auto Err = dumpHeader(Rec.getContents()))
        return Err;
      break;
    }
  }
  return Err;
}

GOFFYAML::Object &GOFFDumper::getYAMLObj() { return YAMLObj; }

Error goff2yaml(raw_ostream &Out, const llvm::object::GOFFObjectFile &Obj) {
  GOFFDumper Dumper(Obj);

  if (auto Err = Dumper.dump())
    return Err;

  yaml::Output Yout(Out);
  Yout << Dumper.getYAMLObj();

  return Error::success();
}
