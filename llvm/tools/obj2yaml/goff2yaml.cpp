//===------ goff2yaml.cpp - obj2yaml conversion tool ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "obj2yaml.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Object/GOFF.h"
#include "llvm/Object/GOFFObjectFile.h"
#include "llvm/ObjectYAML/ObjectYAML.h"
#include "llvm/Support/ConvertEBCDIC.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"

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

  Error dumpHeader(ArrayRef<uint8_t> Data);
  Error dumpExternalSymbol(ArrayRef<uint8_t> Data);
  Error dumpText(ArrayRef<uint8_t> Data);
  Error dumpRelocationDirectory(ArrayRef<uint8_t> Data);
  Error dumpDeferredLength(ArrayRef<uint8_t> Data);
  Error dumpEnd(ArrayRef<uint8_t> Data);

public:
  GOFFDumper(const object::GOFFObjectFile &Obj);
  Error dump();
  GOFFYAML::Object &getYAMLObj();
};

GOFFDumper::GOFFDumper(const object::GOFFObjectFile &Obj) : Obj(Obj) {}

Error GOFFDumper::dumpHeader(ArrayRef<uint8_t> Data) {
  DataExtractor DE(Data, false, 0);
  DataExtractor::Cursor C(0);

  // Flattened data contains: PTV header (bytes 0-2) + bytes 3-60 (prefix) + data from byte 60 onwards
  // HDR data starts at byte 4 in the original record
  C.seek(4); // Skip PTV header and record type
  YAMLObj.Header.TargetEnvironment = DE.getU32(C);
  YAMLObj.Header.TargetOperatingSystem = DE.getU32(C);
  DE.skip(C, 2);
  YAMLObj.Header.CCSID = DE.getU16(C);
  YAMLObj.Header.CharacterSetName = getFixedLengthEBCDICString(DE, C, 16);
  YAMLObj.Header.LanguageProductIdentifier =
      getFixedLengthEBCDICString(DE, C, 16);
  YAMLObj.Header.ArchitectureLevel = DE.getU32(C);
  uint16_t PropertiesLength = DE.getU16(C);
  DE.skip(C, 6);
  if (PropertiesLength) {
    YAMLObj.Header.InternalCCSID = DE.getU16(C);
    PropertiesLength -= 2;
  }
  if (PropertiesLength) {
    YAMLObj.Header.TargetSoftwareEnvironment = DE.getU8(C);
    PropertiesLength -= 1;
  }
  if (!C)
    return C.takeError();
  return Error::success();
}

Error GOFFDumper::dumpExternalSymbol(ArrayRef<uint8_t> Data) {
  // TODO: Implement dumping ESD records
  return Error::success();
}

Error GOFFDumper::dumpText(ArrayRef<uint8_t> Data) {
  // TODO: Implement dumping TXT records
  return Error::success();
}

Error GOFFDumper::dumpRelocationDirectory(ArrayRef<uint8_t> Data) {
  // TODO: Implement dumping RLD records
  return Error::success();
}

Error GOFFDumper::dumpDeferredLength(ArrayRef<uint8_t> Records) {
  // TODO: Implement if/when GOFF LEN records are emitted by current producers
  // or covered by handcrafted-object tests.
  return Error::success();
}

Error GOFFDumper::dumpEnd(ArrayRef<uint8_t> Records) {
  // TODO: implement dumping END records
  return Error::success();
}

Error GOFFDumper::dump() {
  Error Err = Error::success();

  // Use the pre-flattened data structure instead of iterating through records
  const auto& FlattenedData = Obj.getFlattenedData();

  for (const auto& [RecordType, Data] : FlattenedData) {
    switch (RecordType) {
    case GOFF::RT_HDR:
      if (auto Err = dumpHeader(Data))
        return Err;
      break;
    case GOFF::RT_ESD:
      if (auto Err = dumpExternalSymbol(Data))
        return Err;
      break;
    case GOFF::RT_TXT:
      if (auto Err = dumpText(Data))
        return Err;
      break;
    case GOFF::RT_RLD:
      if (auto Err = dumpRelocationDirectory(Data))
        return Err;
      break;
    case GOFF::RT_LEN:
      if (auto Err = dumpDeferredLength(Data))
        return Err;
      break;
    case GOFF::RT_END:
      if (auto Err = dumpEnd(Data))
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