//===------ goff2yaml.cpp - obj2yaml conversion tool ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "obj2yaml.h"
#include "llvm/Object/GOFF.h"
#include "llvm/Object/GOFFObjectFile.h"
#include "llvm/ObjectYAML/ObjectYAML.h"
#include "llvm/Support/ConvertEBCDIC.h"

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
  DataExtractor DE(Data, false);
  DataExtractor::Cursor C(0);

  // Flattened data contains: PTV header (bytes 0-2) + bytes 3-60 (prefix) +
  // data from byte 60 onwards HDR data starts at byte 4 in the original record
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
  GOFFYAML::ESDRecord Sym;
  // Flattened data contains PTV header (bytes 0-2) + bytes 3-72 (prefix) + name
  // data Use DataExtractor to read fields with correct endianness (big-endian
  // for GOFF)
  DataExtractor DE(Data, false); // false = big-endian
  DataExtractor::Cursor C(0);

  // Skip PTV header (bytes 0-2)
  C.seek(3);

  // ESD fields starting from byte 3:
  // Byte 3: Symbol Type
  Sym.SymbolType = DE.getU8(C);

  // Bytes 4-7: ESD ID
  Sym.ESDID = DE.getU32(C);

  // Bytes 8-11: Parent ESD ID
  uint32_t ParentEsdId = DE.getU32(C);
  if (ParentEsdId)
    Sym.ParentESDID = ParentEsdId;

  // Skip bytes 12-15
  DE.skip(C, 4);

  // Bytes 16-19: Offset
  uint32_t Offset = DE.getU32(C);
  if (Offset)
    Sym.Offset = Offset;

  // Skip bytes 20-23
  DE.skip(C, 4);

  // Bytes 24-27: Length
  uint32_t Length = DE.getU32(C);
  if (Length)
    Sym.Length = Length;

  // Skip to byte 40
  C.seek(40);
  uint8_t NameSpace = DE.getU8(C);
  if (NameSpace)
    Sym.NameSpace = NameSpace;

  // Byte 41: Flags
  uint8_t Flags = DE.getU8(C);
  bool FillBytePresent = (Flags & 0x80) != 0;
  if (FillBytePresent) {
    uint8_t FillByteValue = DE.getU8(C);
    Sym.FillByteValue = FillByteValue;
  } else {
    DE.skip(C, 1);
  }

  // Skip 1 byte, then read ADA ESD ID (bytes 44-47)
  DE.skip(C, 1);
  uint32_t ADAEsdId = DE.getU32(C);
  if (ADAEsdId)
    Sym.ADAESDID = ADAEsdId;

  // Bytes 48-51: Sort Priority
  uint32_t SortPriority = DE.getU32(C);
  if (SortPriority)
    Sym.SortPriority = SortPriority;

  // Skip to behavioral attributes at byte 60
  C.seek(60);
  uint8_t Amode = DE.getU8(C);
  if (Amode)
    Sym.Amode = Amode;

  uint8_t Rmode = DE.getU8(C);
  if (Rmode)
    Sym.Rmode = Rmode;

  uint8_t TextStyleAndBinding = DE.getU8(C);
  uint8_t TextStyle = (TextStyleAndBinding >> 4) & 0x0F;
  if (TextStyle)
    Sym.TextStyle = TextStyle;
  uint8_t BindingAlgorithm = TextStyleAndBinding & 0x0F;
  if (BindingAlgorithm)
    Sym.BindingAlgorithm = BindingAlgorithm;

  uint8_t TaskingAndExec = DE.getU8(C);
  uint8_t TaskingBehavior = (TaskingAndExec >> 5) & 0x07;
  if (TaskingBehavior)
    Sym.TaskingBehavior = TaskingBehavior;
  bool ReadOnly = (TaskingAndExec & 0x08) != 0;
  if (ReadOnly)
    Sym.ReadOnly = ReadOnly;
  uint8_t Executable = TaskingAndExec & 0x07;
  if (Executable)
    Sym.Executable = Executable;

  uint8_t BindingFlags = DE.getU8(C);
  uint8_t BindingStrength = BindingFlags & 0x0F;
  if (BindingStrength)
    Sym.BindingStrength = BindingStrength;

  uint8_t LoadingAndScope = DE.getU8(C);
  uint8_t LoadingBehavior = (LoadingAndScope >> 6) & 0x03;
  if (LoadingBehavior)
    Sym.LoadingBehavior = LoadingBehavior;
  bool IndirectReference = (LoadingAndScope & 0x10) != 0;
  if (IndirectReference)
    Sym.IndirectReference = IndirectReference;
  uint8_t BindingScope = LoadingAndScope & 0x0F;
  if (BindingScope)
    Sym.BindingScope = BindingScope;

  uint8_t LinkageAndAlign = DE.getU8(C);
  uint8_t LinkageType = (LinkageAndAlign >> 5) & 0x01;
  if (LinkageType)
    Sym.LinkageType = LinkageType;
  uint8_t Alignment = LinkageAndAlign & 0x1F;
  if (Alignment)
    Sym.Alignment = Alignment;

  // Skip 3 bytes to get to name length at byte 70
  DE.skip(C, 3);
  uint16_t NameLength = DE.getU16(C);
  if (NameLength) {
    // Name data starts at byte 72
    size_t NameOffset = 72;
    if (Data.size() > NameOffset) {
      ArrayRef<uint8_t> NameData = Data.slice(
          NameOffset, std::min((size_t)NameLength, Data.size() - NameOffset));
      StringRef NameStr(reinterpret_cast<const char *>(NameData.data()),
                        NameData.size());
      SmallString<256> UTF8Name;
      ConverterEBCDIC::convertToUTF8(NameStr, UTF8Name);
      Sym.Name = std::string(UTF8Name.str());
    }
  }

  if (!C)
    return C.takeError();

  YAMLObj.ESDRecords.push_back(std::move(Sym));
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
  const auto &FlattenedData = Obj.getFlattenedData();

  for (const auto &[RecordType, Data] : FlattenedData) {
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
