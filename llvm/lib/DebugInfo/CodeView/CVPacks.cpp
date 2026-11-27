//===- CVPacks.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/CVPacks.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/DebugInfo/CodeView/CodeViewError.h"
#include "llvm/DebugInfo/CodeView/RecordSerialization.h"
#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/DebugInfo/CodeView/TypeIndexDiscovery.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LEB128.h"
#include <algorithm>
#include <limits>

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::codeview::cvpack;
using namespace llvm::support::endian;

static Error malformedRecord(const Twine &Message) {
  return make_error<CodeViewError>(Message);
}

/// Append Value in little-endian order.
template <typename T>
static void appendLE(std::vector<uint8_t> &Bytes, T Value) {
  uint8_t Buffer[sizeof(T)];
  support::endian::write<T, llvm::endianness::little>(Buffer, Value);
  Bytes.insert(Bytes.end(), Buffer, Buffer + sizeof(T));
}

static void appendBytes(std::vector<uint8_t> &Bytes, ArrayRef<uint8_t> Data) {
  Bytes.insert(Bytes.end(), Data.begin(), Data.end());
}

static void appendAlignedZeros(std::vector<uint8_t> &Bytes) {
  while ((Bytes.size() & 3) != 0)
    Bytes.push_back(0);
}

static Expected<ArrayRef<uint8_t>> getRecordBody(CVType Record,
                                                 TypeLeafKind Kind) {
  ArrayRef<uint8_t> Bytes = Record.data();
  if (Bytes.size() < sizeof(RecordPrefix) || Record.kind() != Kind)
    return malformedRecord("unexpected or truncated CodeView pack record");
  if (read16le(Bytes.data()) != Bytes.size() - sizeof(uint16_t) ||
      read16le(Bytes.data()) > MaxRecordLength)
    return malformedRecord("invalid CodeView pack record length");

  ArrayRef<uint8_t> Payload = Bytes.drop_front(sizeof(RecordPrefix));
  if (Payload.size() < 8)
    return malformedRecord("truncated CodeView pack record header");
  if (read16le(Payload.data()) != FormatVersion)
    return malformedRecord("unsupported CodeView pack format version");
  if (read16le(Payload.data() + 2) != 0)
    return malformedRecord("unsupported CodeView pack record flags");

  uint32_t BodySize = read32le(Payload.data() + 4);
  if (BodySize > Payload.size() - 8)
    return malformedRecord("truncated CodeView pack record body");
  size_t PaddingSize = Payload.size() - 8 - BodySize;
  if (PaddingSize > 3)
    return malformedRecord("invalid CodeView pack record padding");
  for (size_t I = 0; I != PaddingSize; ++I) {
    uint8_t Expected = static_cast<uint8_t>(LF_PAD0 + PaddingSize - I);
    if (Payload[8 + BodySize + I] != Expected)
      return malformedRecord("invalid CodeView pack record padding");
  }
  return Payload.slice(8, BodySize);
}

static Expected<std::vector<uint8_t>> makeRecord(uint16_t Kind,
                                                 ArrayRef<uint8_t> Body) {
  size_t Size = sizeof(RecordPrefix) + 8 + Body.size();
  size_t Padding = (4 - (Size % 4)) % 4;
  Size += Padding;
  if (Size > MaxRecordLength)
    return malformedRecord("CodeView pack record exceeds the size limit");

  std::vector<uint8_t> Record;
  Record.reserve(Size);
  appendLE<uint16_t>(Record, 0);
  appendLE<uint16_t>(Record, Kind);
  appendLE<uint16_t>(Record, FormatVersion);
  appendLE<uint16_t>(Record, 0);
  appendLE<uint32_t>(Record, static_cast<uint32_t>(Body.size()));
  Record.insert(Record.end(), Body.begin(), Body.end());
  for (size_t I = Padding; I != 0; --I)
    Record.push_back(static_cast<uint8_t>(LF_PAD0 + I));
  support::endian::write16le(Record.data(),
                             static_cast<uint16_t>(Record.size() - 2));
  return Record;
}

static void appendULEB128(std::vector<uint8_t> &Bytes, uint64_t Value) {
  uint8_t Buffer[16];
  Bytes.insert(Bytes.end(), Buffer, Buffer + encodeULEB128(Value, Buffer));
}

static void appendZigZagULEB128(std::vector<uint8_t> &Bytes, int64_t Value) {
  uint64_t Encoded = Value < 0 ? (static_cast<uint64_t>(-(Value + 1)) * 2 + 1)
                               : static_cast<uint64_t>(Value) * 2;
  appendULEB128(Bytes, Encoded);
}

static Expected<uint64_t> readULEB128(ArrayRef<uint8_t> Bytes, size_t &Offset) {
  unsigned Size = 0;
  const char *Problem = nullptr;
  uint64_t Value =
      decodeULEB128(Bytes.data() + Offset, &Size, Bytes.end(), &Problem);
  if (Problem)
    return malformedRecord(Twine("invalid CodeView pack ULEB128 value: ") +
                           Problem);
  if (Size != getULEB128Size(Value))
    return malformedRecord("noncanonical CodeView pack ULEB128 value");
  Offset += Size;
  return Value;
}

static Expected<int64_t> readZigZagULEB128(ArrayRef<uint8_t> Bytes,
                                           size_t &Offset) {
  auto Encoded = readULEB128(Bytes, Offset);
  if (!Encoded)
    return Encoded.takeError();
  uint64_t Value = *Encoded;
  int64_t Magnitude = static_cast<int64_t>(Value >> 1);
  return (Value & 1) != 0 ? -Magnitude - 1 : Magnitude;
}

template <typename RecordT>
static Expected<RangeMap> decodeMap(ArrayRef<RecordT> Records,
                                    TypeLeafKind TypeKind, SymbolKind SymKind) {
  RangeMap Map;
  for (RecordT Record : Records) {
    ArrayRef<uint8_t> Body;
    if constexpr (std::is_same_v<RecordT, CVType>) {
      auto Result = getRecordBody(Record, TypeKind);
      if (!Result)
        return Result.takeError();
      Body = *Result;
    } else {
      CVType AsType(Record.data());
      if (Record.kind() != SymKind)
        return malformedRecord("unexpected CodeView pack symbol map record");
      auto Result = getRecordBody(AsType, static_cast<TypeLeafKind>(SymKind));
      if (!Result)
        return Result.takeError();
      Body = *Result;
    }
    if (Body.empty())
      return malformedRecord("empty CodeView pack map chunk");
    size_t Offset = 0;
    while (Offset != Body.size()) {
      auto Skip = readULEB128(Body, Offset);
      if (!Skip)
        return Skip.takeError();
      auto Slot = readULEB128(Body, Offset);
      if (!Slot)
        return Slot.takeError();
      auto Count = readULEB128(Body, Offset);
      if (!Count)
        return Count.takeError();
      auto FirstDelta = readZigZagULEB128(Body, Offset);
      if (!FirstDelta)
        return FirstDelta.takeError();
      if (*Slot > std::numeric_limits<uint32_t>::max() || *Count == 0 ||
          *Count > std::numeric_limits<uint32_t>::max())
        return malformedRecord("invalid slot or count in CodeView pack map");
      Map.Entries.push_back({*Skip, static_cast<uint32_t>(*Slot),
                             static_cast<uint32_t>(*Count), *FirstDelta});
    }
  }
  return Map;
}

static Expected<std::vector<std::vector<uint8_t>>>
encodeMap(const RangeMap &Map, uint16_t Kind) {
  std::vector<std::vector<uint8_t>> Records;
  std::vector<uint8_t> Body;
  for (const RangeMapEntry &Entry : Map.Entries) {
    if (Entry.Count == 0)
      return malformedRecord("CodeView pack map ranges must be nonempty");
    std::vector<uint8_t> Encoded;
    appendULEB128(Encoded, Entry.Skip);
    appendULEB128(Encoded, Entry.Slot);
    appendULEB128(Encoded, Entry.Count);
    appendZigZagULEB128(Encoded, Entry.FirstDelta);
    if (Encoded.size() > MaxMapBodySize)
      return malformedRecord(
          "CodeView pack map entry exceeds the record size limit");
    if (!Body.empty() && Body.size() + Encoded.size() > MaxMapBodySize) {
      auto Record = makeRecord(Kind, Body);
      if (!Record)
        return Record.takeError();
      Records.push_back(std::move(*Record));
      Body.clear();
    }
    Body.insert(Body.end(), Encoded.begin(), Encoded.end());
  }
  if (!Body.empty()) {
    auto Record = makeRecord(Kind, Body);
    if (!Record)
      return Record.takeError();
    Records.push_back(std::move(*Record));
  }
  return Records;
}

static bool isKnownEditKind(OriginEditKind Kind) {
  return Kind == OriginEditKind::TypeStream ||
         Kind == OriginEditKind::SymbolStream ||
         Kind == OriginEditKind::DebugH || Kind == OriginEditKind::Directive ||
         Kind == OriginEditKind::AddedDirectiveSection;
}

static Error validateOriginEdits(ArrayRef<OriginEdit> Edits) {
  uint32_t PreviousSection = 0;
  uint64_t PreviousEnd = 0;
  for (const OriginEdit &Edit : Edits) {
    if (Edit.SectionIndex == 0 || !isKnownEditKind(Edit.Kind))
      return malformedRecord(
          "invalid section or kind in CodeView pack origin edit");
    if (Edit.OriginalOffset >
        std::numeric_limits<uint64_t>::max() - Edit.OriginalLength)
      return malformedRecord("CodeView pack origin edit range overflows");
    if (Edit.SectionIndex < PreviousSection ||
        (Edit.SectionIndex == PreviousSection &&
         Edit.OriginalOffset < PreviousEnd))
      return malformedRecord(
          "CodeView pack origin edits are unordered or overlap");

    uint64_t FragmentEnd = 0;
    for (const RestorationFragment &Fragment : Edit.RestorationFragments) {
      if (Fragment.Bytes.empty() || Fragment.Offset < FragmentEnd ||
          Fragment.Offset > Edit.OriginalLength ||
          Fragment.Bytes.size() > Edit.OriginalLength - Fragment.Offset)
        return malformedRecord(
            "invalid restoration fragment in CodeView pack origin");
      FragmentEnd = uint64_t(Fragment.Offset) + Fragment.Bytes.size();
    }
    PreviousSection = Edit.SectionIndex;
    PreviousEnd = Edit.OriginalOffset + Edit.OriginalLength;
  }
  return Error::success();
}

static Expected<uint32_t> restorationDataSize(const OriginEdit &Edit) {
  if (Edit.RestorationFragments.empty())
    return uint32_t(0);
  uint64_t Size = 4;
  for (const RestorationFragment &Fragment : Edit.RestorationFragments) {
    Size += 8 + Fragment.Bytes.size();
    if (Size > std::numeric_limits<uint32_t>::max())
      return malformedRecord(
          "CodeView pack origin restoration data is too large");
  }
  return static_cast<uint32_t>(Size);
}

Error llvm::codeview::cvpack::forEachTypeIndex(
    ArrayRef<uint8_t> Record, bool IsSymbol,
    function_ref<Error(uint32_t Offset, TiRefKind Kind)> Callback) {
  SmallVector<TiReference, 8> Refs;
  if (IsSymbol) {
    if (!discoverTypeIndicesInSymbol(Record, Refs))
      return malformedRecord("unknown symbol record kind 0x" +
                             Twine::utohexstr(read16le(Record.data() + 2)));
  } else {
    discoverTypeIndices(Record, Refs);
  }
  for (const TiReference &Ref : Refs) {
    uint32_t Offset = sizeof(RecordPrefix) + Ref.Offset;
    for (uint32_t I = 0; I != Ref.Count; ++I, Offset += sizeof(uint32_t)) {
      if (Offset + sizeof(uint32_t) > Record.size())
        return malformedRecord("invalid type-index offset in CodeView record");
      if (Error E = Callback(Offset, Ref.Kind))
        return E;
    }
  }
  return Error::success();
}

Expected<PackId> llvm::codeview::cvpack::decodePackList(CVType Record) {
  auto BodyOrErr = getRecordBody(Record, TypeLeafKind::LF_CVPACK_LIST);
  if (!BodyOrErr)
    return BodyOrErr.takeError();
  ArrayRef<uint8_t> Body = *BodyOrErr;
  if (Body.size() < 4)
    return malformedRecord("truncated CodeView pack list");
  if (read32le(Body.data()) != 1)
    return malformedRecord("CodeView pack list must hold one pack");
  PackId Id;
  if (Body.size() != 4 + Id.size())
    return malformedRecord("invalid CodeView pack list length");
  std::copy_n(Body.begin() + 4, Id.size(), Id.begin());
  return Id;
}

Expected<Origin> llvm::codeview::cvpack::decodeOrigin(CVType Record) {
  auto BodyOrErr = getRecordBody(Record, TypeLeafKind::LF_CVPACK_ORIGIN);
  if (!BodyOrErr)
    return BodyOrErr.takeError();
  ArrayRef<uint8_t> Body = *BodyOrErr;
  if (Body.size() < 48)
    return malformedRecord("truncated CodeView pack origin");

  Origin Data;
  Data.OriginalFileSize = read64le(Body.data());
  std::copy_n(Body.begin() + 8, Data.OriginalFileHash.size(),
              Data.OriginalFileHash.begin());
  uint32_t DebugHSize = read32le(Body.data() + 40);
  uint32_t EditCount = read32le(Body.data() + 44);
  if (DebugHSize > Body.size() - 48)
    return malformedRecord("truncated .debug$H header in CodeView pack origin");
  size_t Offset = 48;
  Data.DebugHHeader.assign(Body.begin() + Offset,
                           Body.begin() + Offset + DebugHSize);
  Offset += DebugHSize;
  if (EditCount > (Body.size() - Offset) / 28)
    return malformedRecord("invalid edit count in CodeView pack origin");
  Data.Edits.reserve(EditCount);
  for (uint32_t I = 0; I != EditCount; ++I) {
    if (Body.size() - Offset < 28)
      return malformedRecord("truncated edit in CodeView pack origin");
    OriginEdit Edit;
    Edit.SectionIndex = read32le(Body.data() + Offset);
    Edit.Kind = static_cast<OriginEditKind>(read16le(Body.data() + Offset + 4));
    if (read16le(Body.data() + Offset + 6) != 0)
      return malformedRecord("reserved field set in CodeView pack origin edit");
    Edit.OriginalOffset = read64le(Body.data() + Offset + 8);
    Edit.OriginalLength = read32le(Body.data() + Offset + 16);
    Edit.PackedLength = read32le(Body.data() + Offset + 20);
    uint32_t RestorationSize = read32le(Body.data() + Offset + 24);
    Offset += 28;
    if (RestorationSize > Body.size() - Offset)
      return malformedRecord(
          "truncated restoration data in CodeView pack origin");

    if (RestorationSize != 0) {
      ArrayRef<uint8_t> Restoration = Body.slice(Offset, RestorationSize);
      if (Restoration.size() < 4)
        return malformedRecord("truncated restoration fragment list");
      uint32_t FragmentCount = read32le(Restoration.data());
      if (FragmentCount > (Restoration.size() - 4) / 8)
        return malformedRecord("invalid restoration fragment count");
      size_t FragmentOffset = 4;
      Edit.RestorationFragments.reserve(FragmentCount);
      for (uint32_t J = 0; J != FragmentCount; ++J) {
        if (Restoration.size() - FragmentOffset < 8)
          return malformedRecord(
              "truncated CodeView pack restoration fragment");
        uint32_t DataOffset = read32le(Restoration.data() + FragmentOffset);
        uint32_t DataSize = read32le(Restoration.data() + FragmentOffset + 4);
        FragmentOffset += 8;
        if (DataSize > Restoration.size() - FragmentOffset)
          return malformedRecord("truncated CodeView pack restoration bytes");
        RestorationFragment Fragment;
        Fragment.Offset = DataOffset;
        Fragment.Bytes.assign(Restoration.begin() + FragmentOffset,
                              Restoration.begin() + FragmentOffset + DataSize);
        Edit.RestorationFragments.push_back(std::move(Fragment));
        FragmentOffset += DataSize;
      }
      if (FragmentOffset != Restoration.size())
        return malformedRecord(
            "extra bytes in CodeView pack restoration fragments");
    }
    Offset += RestorationSize;
    Data.Edits.push_back(std::move(Edit));
  }
  if (Offset != Body.size())
    return malformedRecord("extra bytes in CodeView pack origin");
  if (Error E = validateOriginEdits(Data.Edits))
    return std::move(E);
  return Data;
}

Expected<RangeMap>
llvm::codeview::cvpack::decodeTypeMap(ArrayRef<CVType> Records) {
  return decodeMap(Records, TypeLeafKind::LF_CVPACK_MAP,
                   SymbolKind::S_CVPACK_MAP);
}

Expected<RangeMap>
llvm::codeview::cvpack::decodeSymbolMap(ArrayRef<CVSymbol> Records) {
  return decodeMap(Records, TypeLeafKind::LF_CVPACK_MAP,
                   SymbolKind::S_CVPACK_MAP);
}

Expected<std::vector<uint8_t>>
llvm::codeview::cvpack::encodePackList(const PackId &Id) {
  std::vector<uint8_t> Body;
  appendLE<uint32_t>(Body, 1);
  appendBytes(Body, Id);
  return makeRecord(TypeLeafKind::LF_CVPACK_LIST, Body);
}

Expected<std::vector<uint8_t>>
llvm::codeview::cvpack::encodeOrigin(const Origin &Data) {
  if (Data.DebugHHeader.size() > std::numeric_limits<uint32_t>::max() ||
      Data.Edits.size() > std::numeric_limits<uint32_t>::max())
    return malformedRecord("CodeView pack origin contains too many entries");
  if (Error E = validateOriginEdits(Data.Edits))
    return std::move(E);

  std::vector<uint8_t> Body;
  appendLE<uint64_t>(Body, Data.OriginalFileSize);
  appendBytes(Body, Data.OriginalFileHash);
  appendLE<uint32_t>(Body, static_cast<uint32_t>(Data.DebugHHeader.size()));
  appendLE<uint32_t>(Body, static_cast<uint32_t>(Data.Edits.size()));
  appendBytes(Body, Data.DebugHHeader);
  for (const OriginEdit &Edit : Data.Edits) {
    auto RestorationSize = restorationDataSize(Edit);
    if (!RestorationSize)
      return RestorationSize.takeError();
    appendLE<uint32_t>(Body, Edit.SectionIndex);
    appendLE<uint16_t>(Body, static_cast<uint16_t>(Edit.Kind));
    appendLE<uint16_t>(Body, 0);
    appendLE<uint64_t>(Body, Edit.OriginalOffset);
    appendLE<uint32_t>(Body, Edit.OriginalLength);
    appendLE<uint32_t>(Body, Edit.PackedLength);
    appendLE<uint32_t>(Body, *RestorationSize);
    if (*RestorationSize == 0)
      continue;
    appendLE<uint32_t>(Body,
                       static_cast<uint32_t>(Edit.RestorationFragments.size()));
    for (const RestorationFragment &Fragment : Edit.RestorationFragments) {
      appendLE<uint32_t>(Body, Fragment.Offset);
      appendLE<uint32_t>(Body, static_cast<uint32_t>(Fragment.Bytes.size()));
      appendBytes(Body, Fragment.Bytes);
    }
  }
  return makeRecord(TypeLeafKind::LF_CVPACK_ORIGIN, Body);
}

std::optional<FunctionRef>
llvm::codeview::cvpack::decodeFunctionRef(ArrayRef<uint8_t> SectionData) {
  if (SectionData.size() != FunctionRefSectionSize ||
      read32le(SectionData.data()) != COFF::DEBUG_SECTION_MAGIC ||
      read32le(SectionData.data() + 4) != FunctionRefSubsectionKind ||
      read32le(SectionData.data() + 8) != 12)
    return std::nullopt;
  return FunctionRef{read32le(SectionData.data() + 12),
                     read32le(SectionData.data() + 16),
                     read32le(SectionData.data() + 20)};
}

std::vector<uint8_t>
llvm::codeview::cvpack::encodeFunctionRef(const FunctionRef &Ref) {
  std::vector<uint8_t> Data;
  appendLE<uint32_t>(Data, COFF::DEBUG_SECTION_MAGIC);
  appendLE<uint32_t>(Data, FunctionRefSubsectionKind);
  appendLE<uint32_t>(Data, 12);
  appendBytes(Data, encodeFunctionRefBody(Ref));
  return Data;
}

Expected<FunctionRef>
llvm::codeview::cvpack::decodeFunctionRefBody(ArrayRef<uint8_t> Body) {
  if (Body.size() != 12)
    return malformedRecord("invalid CodeView pack function reference");
  return FunctionRef{read32le(Body.data()), read32le(Body.data() + 4),
                     read32le(Body.data() + 8)};
}

std::vector<uint8_t>
llvm::codeview::cvpack::encodeFunctionRefBody(const FunctionRef &Ref) {
  std::vector<uint8_t> Body;
  appendLE<uint32_t>(Body, Ref.Slot);
  appendLE<uint32_t>(Body, Ref.Ordinal);
  appendLE<uint32_t>(Body, Ref.OriginalSize);
  return Body;
}

Expected<std::vector<PackFileEntry>>
llvm::codeview::cvpack::decodePackFiles(ArrayRef<uint8_t> Body) {
  if (Body.size() < 4)
    return malformedRecord("truncated CodeView pack files subsection");
  uint32_t Count = read32le(Body.data());
  std::vector<PackFileEntry> Files;
  size_t Pos = 4;
  for (uint32_t I = 0; I != Count; ++I) {
    if (Body.size() - Pos < 8)
      return malformedRecord("truncated CodeView pack file entry");
    uint32_t NameSize = read32le(Body.data() + Pos);
    uint8_t ChecksumSize = Body[Pos + 5];
    if (uint64_t(NameSize) + ChecksumSize > Body.size() - Pos - 8)
      return malformedRecord("truncated CodeView pack file entry");
    Files.push_back({toStringRef(Body.slice(Pos + 8, NameSize)), Body[Pos + 4],
                     Body.slice(Pos + 8 + NameSize, ChecksumSize)});
    Pos = std::min((Pos + 8 + NameSize + ChecksumSize + 3) & ~size_t(3),
                   Body.size());
  }
  if (Pos != Body.size())
    return malformedRecord("trailing data in CodeView pack files subsection");
  return Files;
}

Expected<std::vector<uint8_t>>
llvm::codeview::cvpack::encodePackFiles(ArrayRef<PackFileEntry> Files) {
  std::vector<uint8_t> Body;
  appendLE<uint32_t>(Body, Files.size());
  for (const PackFileEntry &File : Files) {
    if (File.Checksum.size() > 0xff)
      return malformedRecord("CodeView pack file checksum is too large");
    appendLE<uint32_t>(Body, File.Name.size());
    Body.push_back(File.ChecksumKind);
    Body.push_back(File.Checksum.size());
    appendLE<uint16_t>(Body, 0);
    appendBytes(Body, arrayRefFromStringRef(File.Name));
    appendBytes(Body, File.Checksum);
    appendAlignedZeros(Body);
  }
  return Body;
}

Expected<std::vector<ArrayRef<uint8_t>>>
llvm::codeview::cvpack::decodePackFunctions(ArrayRef<uint8_t> Body) {
  if (Body.size() < 4)
    return malformedRecord("truncated CodeView pack functions subsection");
  uint32_t Count = read32le(Body.data());
  std::vector<ArrayRef<uint8_t>> Functions;
  Functions.reserve(std::min<size_t>(Count, Body.size() / 4));
  size_t Pos = 4;
  for (uint32_t I = 0; I != Count; ++I) {
    if (Body.size() - Pos < 4 ||
        read32le(Body.data() + Pos) > Body.size() - Pos - 4)
      return malformedRecord("truncated CodeView pack function");
    uint32_t Size = read32le(Body.data() + Pos);
    Functions.push_back(Body.slice(Pos + 4, Size));
    Pos = std::min((Pos + 4 + Size + 3) & ~size_t(3), Body.size());
  }
  if (Pos != Body.size())
    return malformedRecord(
        "trailing data in CodeView pack functions subsection");
  return Functions;
}

void llvm::codeview::cvpack::encodePackFunctions(
    ArrayRef<ArrayRef<uint8_t>> Functions, std::vector<uint8_t> &Out) {
  appendLE<uint32_t>(Out, Functions.size());
  for (ArrayRef<uint8_t> Function : Functions) {
    appendLE<uint32_t>(Out, Function.size());
    appendBytes(Out, Function);
    appendAlignedZeros(Out);
  }
}

bool llvm::codeview::cvpack::isPackedTypeStream(ArrayRef<uint8_t> Section) {
  if (Section.size() < 8)
    return false;
  return read32le(Section.data()) == COFF::DEBUG_CVPACK_SECTION_MAGIC &&
         read16le(Section.data() + 6) == TypeLeafKind::LF_CVPACK_LIST;
}

Expected<std::vector<std::vector<uint8_t>>>
llvm::codeview::cvpack::encodeTypeMap(const RangeMap &Map) {
  return encodeMap(Map, TypeLeafKind::LF_CVPACK_MAP);
}

Expected<std::vector<std::vector<uint8_t>>>
llvm::codeview::cvpack::encodeSymbolMap(const RangeMap &Map) {
  return encodeMap(Map, SymbolKind::S_CVPACK_MAP);
}

Expected<std::vector<RangeMapSegment>>
llvm::codeview::cvpack::expandRangeMap(const RangeMap &Map, uint64_t PackEnd,
                                       uint64_t InlineCount,
                                       uint64_t FirstPackIndex) {
  uint64_t PreviousEnd = FirstPackIndex;
  std::vector<RangeMapSegment> Segments;
  uint64_t InlineUsed = 0;
  for (const RangeMapEntry &Entry : Map.Entries) {
    if (Entry.Slot != 0)
      return malformedRecord("CodeView pack map slot is out of range");
    if (InlineUsed > InlineCount || Entry.Skip > InlineCount - InlineUsed)
      return malformedRecord("CodeView pack map skips beyond inline records");
    if (Entry.Skip != 0)
      Segments.push_back({false, Entry.Skip, 0});
    InlineUsed += Entry.Skip;

    uint64_t First = PreviousEnd;
    if (Entry.FirstDelta >= 0) {
      uint64_t Delta = static_cast<uint64_t>(Entry.FirstDelta);
      if (First > std::numeric_limits<uint64_t>::max() - Delta)
        return malformedRecord("CodeView pack map index overflows");
      First += Delta;
    } else {
      uint64_t Delta = static_cast<uint64_t>(-(Entry.FirstDelta + 1)) + 1;
      if (First < Delta)
        return malformedRecord("CodeView pack map index underflows");
      First -= Delta;
    }
    if (First < FirstPackIndex || First > PackEnd ||
        Entry.Count > PackEnd - First)
      return malformedRecord("CodeView pack map range exceeds its pack");
    Segments.push_back({true, Entry.Count, First});
    PreviousEnd = First + Entry.Count;
  }

  if (InlineUsed > InlineCount)
    return malformedRecord(
        "CodeView pack map consumes too many inline records");
  uint64_t TrailingInline = InlineCount - InlineUsed;
  if (TrailingInline != 0)
    Segments.push_back({false, TrailingInline, 0});
  return Segments;
}
