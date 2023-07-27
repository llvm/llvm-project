//===- BTFParser.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// BTFParser reads/interprets .BTF and .BTF.ext ELF sections.
// Refer to BTFParser.h for API description.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/BTF/BTFParser.h"
#include "llvm/Support/Errc.h"

#define DEBUG_TYPE "debug-info-btf-parser"

using namespace llvm;
using object::ObjectFile;
using object::SectionedAddress;
using object::SectionRef;

const char BTFSectionName[] = ".BTF";
const char BTFExtSectionName[] = ".BTF.ext";

// Utility class with API similar to raw_ostream but can be cast
// to Error, e.g.:
//
// Error foo(...) {
//   ...
//   if (Error E = bar(...))
//     return Err("error while foo(): ") << E;
//   ...
// }
//
namespace {
class Err {
  std::string Buffer;
  raw_string_ostream Stream;

public:
  Err(const char *InitialMsg) : Buffer(InitialMsg), Stream(Buffer) {}
  Err(const char *SectionName, DataExtractor::Cursor &C)
      : Buffer(), Stream(Buffer) {
    *this << "error while reading " << SectionName
          << " section: " << C.takeError();
  };

  template <typename T> Err &operator<<(T Val) {
    Stream << Val;
    return *this;
  }

  Err &write_hex(unsigned long long Val) {
    Stream.write_hex(Val);
    return *this;
  }

  Err &operator<<(Error Val) {
    handleAllErrors(std::move(Val),
                    [=](ErrorInfoBase &Info) { Stream << Info.message(); });
    return *this;
  }

  operator Error() const {
    return make_error<StringError>(Buffer, errc::invalid_argument);
  }
};
} // anonymous namespace

// ParseContext wraps information that is only necessary while parsing
// ObjectFile and can be discarded once parsing is done.
// Used by BTFParser::parse* auxiliary functions.
struct BTFParser::ParseContext {
  const ObjectFile &Obj;
  // Map from ELF section name to SectionRef
  DenseMap<StringRef, SectionRef> Sections;

public:
  ParseContext(const ObjectFile &Obj) : Obj(Obj) {}

  Expected<DataExtractor> makeExtractor(SectionRef Sec) {
    Expected<StringRef> Contents = Sec.getContents();
    if (!Contents)
      return Contents.takeError();
    return DataExtractor(Contents.get(), Obj.isLittleEndian(),
                         Obj.getBytesInAddress());
  }

  std::optional<SectionRef> findSection(StringRef Name) const {
    auto It = Sections.find(Name);
    if (It != Sections.end())
      return It->second;
    return std::nullopt;
  }
};

Error BTFParser::parseBTF(ParseContext &Ctx, SectionRef BTF) {
  Expected<DataExtractor> MaybeExtractor = Ctx.makeExtractor(BTF);
  if (!MaybeExtractor)
    return MaybeExtractor.takeError();

  DataExtractor &Extractor = MaybeExtractor.get();
  DataExtractor::Cursor C = DataExtractor::Cursor(0);
  uint16_t Magic = Extractor.getU16(C);
  if (!C)
    return Err(".BTF", C);
  if (Magic != BTF::MAGIC)
    return Err("invalid .BTF magic: ").write_hex(Magic);
  uint8_t Version = Extractor.getU8(C);
  if (!C)
    return Err(".BTF", C);
  if (Version != 1)
    return Err("unsupported .BTF version: ") << (unsigned)Version;
  (void)Extractor.getU8(C); // flags
  uint32_t HdrLen = Extractor.getU32(C);
  if (!C)
    return Err(".BTF", C);
  if (HdrLen < 8)
    return Err("unexpected .BTF header length: ") << HdrLen;
  (void)Extractor.getU32(C); // type_off
  (void)Extractor.getU32(C); // type_len
  uint32_t StrOff = Extractor.getU32(C);
  uint32_t StrLen = Extractor.getU32(C);
  uint32_t StrStart = HdrLen + StrOff;
  uint32_t StrEnd = StrStart + StrLen;
  if (!C)
    return Err(".BTF", C);
  if (Extractor.getData().size() < StrEnd)
    return Err("invalid .BTF section size, expecting at-least ")
           << StrEnd << " bytes";

  StringsTable = Extractor.getData().substr(StrStart, StrLen);
  return Error::success();
}

Error BTFParser::parseBTFExt(ParseContext &Ctx, SectionRef BTFExt) {
  Expected<DataExtractor> MaybeExtractor = Ctx.makeExtractor(BTFExt);
  if (!MaybeExtractor)
    return MaybeExtractor.takeError();

  DataExtractor &Extractor = MaybeExtractor.get();
  DataExtractor::Cursor C = DataExtractor::Cursor(0);
  uint16_t Magic = Extractor.getU16(C);
  if (!C)
    return Err(".BTF.ext", C);
  if (Magic != BTF::MAGIC)
    return Err("invalid .BTF.ext magic: ").write_hex(Magic);
  uint8_t Version = Extractor.getU8(C);
  if (!C)
    return Err(".BTF", C);
  if (Version != 1)
    return Err("unsupported .BTF.ext version: ") << (unsigned)Version;
  (void)Extractor.getU8(C); // flags
  uint32_t HdrLen = Extractor.getU32(C);
  if (!C)
    return Err(".BTF.ext", C);
  if (HdrLen < 8)
    return Err("unexpected .BTF.ext header length: ") << HdrLen;
  (void)Extractor.getU32(C); // func_info_off
  (void)Extractor.getU32(C); // func_info_len
  uint32_t LineInfoOff = Extractor.getU32(C);
  uint32_t LineInfoLen = Extractor.getU32(C);
  if (!C)
    return Err(".BTF.ext", C);
  uint32_t LineInfoStart = HdrLen + LineInfoOff;
  uint32_t LineInfoEnd = LineInfoStart + LineInfoLen;
  if (Error E = parseLineInfo(Ctx, Extractor, LineInfoStart, LineInfoEnd))
    return E;

  return Error::success();
}

Error BTFParser::parseLineInfo(ParseContext &Ctx, DataExtractor &Extractor,
                               uint64_t LineInfoStart, uint64_t LineInfoEnd) {
  DataExtractor::Cursor C = DataExtractor::Cursor(LineInfoStart);
  uint32_t RecSize = Extractor.getU32(C);
  if (!C)
    return Err(".BTF.ext", C);
  if (RecSize < 16)
    return Err("unexpected .BTF.ext line info record length: ") << RecSize;

  while (C && C.tell() < LineInfoEnd) {
    uint32_t SecNameOff = Extractor.getU32(C);
    uint32_t NumInfo = Extractor.getU32(C);
    StringRef SecName = findString(SecNameOff);
    std::optional<SectionRef> Sec = Ctx.findSection(SecName);
    if (!C)
      return Err(".BTF.ext", C);
    if (!Sec)
      return Err("") << "can't find section '" << SecName
                     << "' while parsing .BTF.ext line info";
    BTFLinesVector &Lines = SectionLines[Sec->getIndex()];
    for (uint32_t I = 0; C && I < NumInfo; ++I) {
      uint64_t RecStart = C.tell();
      uint32_t InsnOff = Extractor.getU32(C);
      uint32_t FileNameOff = Extractor.getU32(C);
      uint32_t LineOff = Extractor.getU32(C);
      uint32_t LineCol = Extractor.getU32(C);
      if (!C)
        return Err(".BTF.ext", C);
      Lines.push_back({InsnOff, FileNameOff, LineOff, LineCol});
      C.seek(RecStart + RecSize);
    }
    llvm::stable_sort(Lines,
                      [](const BTF::BPFLineInfo &L, const BTF::BPFLineInfo &R) {
                        return L.InsnOffset < R.InsnOffset;
                      });
  }
  if (!C)
    return Err(".BTF.ext", C);

  return Error::success();
}

Error BTFParser::parse(const ObjectFile &Obj) {
  StringsTable = StringRef();
  SectionLines.clear();

  ParseContext Ctx(Obj);
  std::optional<SectionRef> BTF;
  std::optional<SectionRef> BTFExt;
  for (SectionRef Sec : Obj.sections()) {
    Expected<StringRef> MaybeName = Sec.getName();
    if (!MaybeName)
      return Err("error while reading section name: ") << MaybeName.takeError();
    Ctx.Sections[*MaybeName] = Sec;
    if (*MaybeName == BTFSectionName)
      BTF = Sec;
    if (*MaybeName == BTFExtSectionName)
      BTFExt = Sec;
  }
  if (!BTF)
    return Err("can't find .BTF section");
  if (!BTFExt)
    return Err("can't find .BTF.ext section");
  if (Error E = parseBTF(Ctx, *BTF))
    return E;
  if (Error E = parseBTFExt(Ctx, *BTFExt))
    return E;

  return Error::success();
}

bool BTFParser::hasBTFSections(const ObjectFile &Obj) {
  bool HasBTF = false;
  bool HasBTFExt = false;
  for (SectionRef Sec : Obj.sections()) {
    Expected<StringRef> Name = Sec.getName();
    if (Error E = Name.takeError()) {
      logAllUnhandledErrors(std::move(E), errs());
      continue;
    }
    HasBTF |= *Name == BTFSectionName;
    HasBTFExt |= *Name == BTFExtSectionName;
    if (HasBTF && HasBTFExt)
      return true;
  }
  return false;
}

StringRef BTFParser::findString(uint32_t Offset) const {
  return StringsTable.slice(Offset, StringsTable.find(0, Offset));
}

const BTF::BPFLineInfo *
BTFParser::findLineInfo(SectionedAddress Address) const {
  auto MaybeSecInfo = SectionLines.find(Address.SectionIndex);
  if (MaybeSecInfo == SectionLines.end())
    return nullptr;

  const BTFLinesVector &SecInfo = MaybeSecInfo->second;
  const uint64_t TargetOffset = Address.Address;
  BTFLinesVector::const_iterator LineInfo =
      llvm::partition_point(SecInfo, [=](const BTF::BPFLineInfo &Line) {
        return Line.InsnOffset < TargetOffset;
      });
  if (LineInfo == SecInfo.end() || LineInfo->InsnOffset != Address.Address)
    return nullptr;

  return LineInfo;
}
