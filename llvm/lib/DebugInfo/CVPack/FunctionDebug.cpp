//===- FunctionDebug.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Moves the debug info of COMDAT functions between an object and a pack.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CVPack/FunctionDebug.h"
#include "CVPackUtils.h"
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/DebugInfo/CodeView/CVPacks.h"
#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/DebugInfo/CodeView/TypeIndexDiscovery.h"
#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::codeview::cvpack;
using namespace llvm::support::endian;

/// Read a compressed annotation integer; nullopt at the end or for an invalid
/// first byte.
static std::optional<uint32_t> readAnnotation(ArrayRef<uint8_t> Data,
                                              size_t &Pos) {
  if (Pos >= Data.size())
    return std::nullopt;
  uint8_t B0 = Data[Pos];
  if ((B0 & 0x80) == 0) {
    Pos += 1;
    return B0;
  }
  if ((B0 & 0xc0) == 0x80) {
    if (Data.size() - Pos < 2)
      return std::nullopt;
    uint32_t V = (uint32_t(B0 & 0x3f) << 8) | Data[Pos + 1];
    Pos += 2;
    return V;
  }
  if ((B0 & 0xe0) == 0xc0) {
    if (Data.size() - Pos < 4)
      return std::nullopt;
    uint32_t V = (uint32_t(B0 & 0x1f) << 24) | (uint32_t(Data[Pos + 1]) << 16) |
                 (uint32_t(Data[Pos + 2]) << 8) | Data[Pos + 3];
    Pos += 4;
    return V;
  }
  return std::nullopt;
}

static Error appendAnnotation(std::vector<uint8_t> &Out, uint32_t V,
                              FileOperandEncoding Encoding) {
  if (V > 0x1fffffff)
    return malformedRecord("annotation operand is too large");
  if (Encoding == FileOperandEncoding::Minimal && V <= 0x7f) {
    Out.push_back(V);
  } else if (Encoding == FileOperandEncoding::Minimal && V <= 0x3fff) {
    Out.push_back(0x80 | (V >> 8));
    Out.push_back(V & 0xff);
  } else {
    Out.push_back(0xc0 | (V >> 24));
    Out.push_back((V >> 16) & 0xff);
    Out.push_back((V >> 8) & 0xff);
    Out.push_back(V & 0xff);
  }
  return Error::success();
}

/// Whether an inline site's annotations change the file.
static bool changesFile(ArrayRef<uint8_t> Annotations) {
  for (size_t Pos = 0; Pos < Annotations.size();) {
    std::optional<uint32_t> Op = readAnnotation(Annotations, Pos);
    if (!Op || *Op == uint32_t(BinaryAnnotationsOpCode::Invalid))
      return false;
    if (*Op == uint32_t(BinaryAnnotationsOpCode::ChangeFile))
      return true;
    unsigned Operands =
        *Op == uint32_t(BinaryAnnotationsOpCode::ChangeCodeLengthAndCodeOffset)
            ? 2
            : 1;
    for (unsigned I = 0; I != Operands; ++I)
      if (!readAnnotation(Annotations, Pos))
        return false;
  }
  return false;
}

/// Append the annotations with every file operand mapped and re-encoded; the
/// padding after the last annotation is dropped.
/// Base is the offset of the annotations for MapFile.
static Error appendMappedAnnotations(
    ArrayRef<uint8_t> Annotations, uint32_t Base, std::vector<uint8_t> &Out,
    function_ref<Expected<uint32_t>(uint32_t, uint32_t)> MapFile,
    FileOperandEncoding Encoding) {
  for (size_t Pos = 0; Pos < Annotations.size();) {
    size_t Start = Pos;
    std::optional<uint32_t> Op = readAnnotation(Annotations, Pos);
    if (!Op)
      return malformedRecord("invalid inline-site annotation");
    if (*Op == uint32_t(BinaryAnnotationsOpCode::Invalid))
      break;
    if (*Op == uint32_t(BinaryAnnotationsOpCode::ChangeFile)) {
      uint32_t Operand = Base + Pos;
      std::optional<uint32_t> File = readAnnotation(Annotations, Pos);
      if (!File)
        return malformedRecord("invalid inline-site annotation");
      auto Mapped = MapFile(*File, Operand);
      if (!Mapped)
        return Mapped.takeError();
      appendBytes(Out, Annotations.slice(Start, 1));
      if (Error E = appendAnnotation(Out, *Mapped, Encoding))
        return E;
      continue;
    }
    unsigned Operands =
        *Op == uint32_t(BinaryAnnotationsOpCode::ChangeCodeLengthAndCodeOffset)
            ? 2
            : 1;
    for (unsigned I = 0; I != Operands; ++I)
      if (!readAnnotation(Annotations, Pos))
        return malformedRecord("invalid inline-site annotation");
    appendBytes(Out, Annotations.slice(Start, Pos - Start));
  }
  return Error::success();
}

Error llvm::codeview::cvpack::transformFunctionDebug(
    ArrayRef<uint8_t> Data, std::vector<uint8_t> &Out,
    function_ref<Expected<uint32_t>(uint32_t, uint32_t)> MapType,
    function_ref<Expected<uint32_t>(uint32_t, uint32_t)> MapFile,
    FileOperandEncoding Encoding) {
  auto mapAt = [&](size_t OutOffset, uint32_t Offset, bool IsFile) -> Error {
    uint32_t Value = support::endian::read32le(Out.data() + OutOffset);
    if (!IsFile && Value < TypeIndex::FirstNonSimpleIndex)
      return Error::success();
    auto Mapped = IsFile ? MapFile(Value, Offset) : MapType(Value, Offset);
    if (!Mapped)
      return Mapped.takeError();
    support::endian::write32le(Out.data() + OutOffset, *Mapped);
    return Error::success();
  };
  size_t Offset = 0;
  while (Offset != Data.size()) {
    if (Data.size() - Offset < 8)
      return malformedRecord("truncated function debug subsection");
    uint32_t Kind = read32le(Data.data() + Offset);
    uint32_t Size = read32le(Data.data() + Offset + 4);
    size_t Begin = Offset + 8;
    if (Size > Data.size() - Begin)
      return malformedRecord("truncated function debug subsection");
    size_t End = Begin + Size;
    size_t OutHeader = Out.size();
    appendLE<uint32_t>(Out, Kind);
    appendLE<uint32_t>(Out, 0);
    if (Kind == uint32_t(DebugSubsectionKind::Symbols)) {
      for (size_t Pos = Begin; Pos != End;) {
        if (End - Pos < 4)
          return malformedRecord("truncated function debug symbol");
        size_t RecordSize = size_t(read16le(Data.data() + Pos)) + 2;
        if (RecordSize < 4 || RecordSize > End - Pos)
          return malformedRecord("truncated function debug symbol");
        ArrayRef<uint8_t> Record = Data.slice(Pos, RecordSize);
        SymbolKind SymKind =
            static_cast<SymbolKind>(read16le(Data.data() + Pos + 2));
        if (SymKind == SymbolKind::S_FILESTATIC ||
            SymKind == SymbolKind::S_DEFRANGE ||
            SymKind == SymbolKind::S_CVPACK_MAP)
          return malformedRecord("function debug symbol cannot be packed");
        size_t OutRecord = Out.size();
        size_t Annotations =
            16 + (SymKind == SymbolKind::S_INLINESITE2 ? 4 : 0);
        bool Rebuild = (SymKind == SymbolKind::S_INLINESITE ||
                        SymKind == SymbolKind::S_INLINESITE2) &&
                       RecordSize >= Annotations &&
                       changesFile(Record.drop_front(Annotations));
        if (Rebuild) {
          appendBytes(Out, Record.take_front(Annotations));
          if (Error E = appendMappedAnnotations(Record.drop_front(Annotations),
                                                Pos + Annotations, Out, MapFile,
                                                Encoding))
            return E;
          appendAlignedZeros(Out);
          size_t NewSize = Out.size() - OutRecord;
          if (NewSize - 2 > 0xffff)
            return malformedRecord("inline site is too large");
          write16le(Out.data() + OutRecord, static_cast<uint16_t>(NewSize - 2));
        } else {
          appendBytes(Out, Record);
        }
        // A rebuilt inline site keeps its type indices before its annotations.
        if (Error E = forEachTypeIndex(
                Record, true, [&](uint32_t Field, TiRefKind) -> Error {
                  if (Rebuild && Field + 4 > Annotations)
                    return malformedRecord(
                        "invalid type-index offset in symbol");
                  return mapAt(OutRecord + Field, Pos + Field, false);
                }))
          return E;
        Pos += RecordSize;
      }
    } else if (Kind == uint32_t(DebugSubsectionKind::Lines)) {
      if (Size < 12)
        return malformedRecord("truncated lines subsection");
      size_t OutBegin = Out.size();
      appendBytes(Out, Data.slice(Begin, Size));
      bool Columns = read16le(Data.data() + Begin + 6) & LF_HaveColumns;
      for (size_t Pos = Begin + 12; Pos != End;) {
        if (End - Pos < 12)
          return malformedRecord("truncated line block");
        uint64_t BlockSize =
            12 + uint64_t(read32le(Data.data() + Pos + 4)) * (Columns ? 12 : 8);
        if (read32le(Data.data() + Pos + 8) != BlockSize ||
            BlockSize > End - Pos)
          return malformedRecord("invalid line block size");
        if (Error E = mapAt(OutBegin + (Pos - Begin), Pos, true))
          return E;
        Pos += BlockSize;
      }
    } else {
      return malformedRecord("function debug subsection cannot be packed");
    }
    write32le(Out.data() + OutHeader + 4,
              static_cast<uint32_t>(Out.size() - OutHeader - 8));
    size_t Next = std::min((End + 3) & ~size_t(3), Data.size());
    // Keep the original padding bytes, which are part of the section.
    appendBytes(Out, Data.slice(End, Next - End));
    Offset = Next;
  }
  return Error::success();
}

std::string llvm::codeview::cvpack::fileIdentity(const PackFileEntry &File) {
  std::string Identity;
  Identity.push_back(char(File.ChecksumKind));
  Identity.push_back(char(File.Checksum.size()));
  Identity.append(File.Checksum.begin(), File.Checksum.end());
  Identity.append(File.Name.begin(), File.Name.end());
  return Identity;
}

Expected<ObjectFileTable>
ObjectFileTable::create(ArrayRef<ArrayRef<uint8_t>> DebugSSections) {
  std::optional<ArrayRef<uint8_t>> Strings, Checksums;
  for (ArrayRef<uint8_t> Section : DebugSSections) {
    if (Section.size() < 4 ||
        read32le(Section.data()) != COFF::DEBUG_SECTION_MAGIC)
      continue;
    // The last subsection need not be padded: Offset can pass the end.
    for (size_t Offset = 4; Offset + 8 <= Section.size();) {
      uint32_t Kind = read32le(Section.data() + Offset);
      uint32_t Size = read32le(Section.data() + Offset + 4);
      if (Size > Section.size() - Offset - 8)
        break;
      std::optional<ArrayRef<uint8_t>> *Slot = nullptr;
      if (Kind == uint32_t(DebugSubsectionKind::StringTable))
        Slot = &Strings;
      else if (Kind == uint32_t(DebugSubsectionKind::FileChecksums))
        Slot = &Checksums;
      if (Slot) {
        if (*Slot)
          return malformedRecord("object has several file checksum tables");
        *Slot = Section.slice(Offset + 8, Size);
      }
      Offset += 8 + ((size_t(Size) + 3) & ~size_t(3));
    }
  }
  ObjectFileTable Table;
  if (!Checksums)
    return Table;
  if (!Strings)
    return malformedRecord("object has file checksums but no string table");
  for (size_t Offset = 0; Offset != Checksums->size();) {
    if (Checksums->size() - Offset < 6)
      return malformedRecord("truncated file checksum entry");
    uint32_t NameOffset = read32le(Checksums->data() + Offset);
    uint8_t Size = (*Checksums)[Offset + 4];
    uint8_t Kind = (*Checksums)[Offset + 5];
    if (Size > Checksums->size() - Offset - 6 || NameOffset >= Strings->size())
      return malformedRecord("invalid file checksum entry");
    StringRef Rest = toStringRef(Strings->drop_front(NameOffset));
    size_t NameEnd = Rest.find('\0');
    if (NameEnd == StringRef::npos)
      return malformedRecord("unterminated file name");
    PackFileEntry Entry{Rest.take_front(NameEnd), Kind,
                        Checksums->slice(Offset + 6, Size)};
    Table.ByIdentity.try_emplace(fileIdentity(Entry), uint32_t(Offset));
    Table.Entries.emplace_back(uint32_t(Offset), Entry);
    Offset = std::min((Offset + 6 + Size + 3) & ~size_t(3), Checksums->size());
  }
  return Table;
}

std::optional<uint32_t>
ObjectFileTable::lookup(const PackFileEntry &File) const {
  auto It = ByIdentity.find(fileIdentity(File));
  if (It == ByIdentity.end())
    return std::nullopt;
  return It->second;
}

std::optional<PackFileEntry> ObjectFileTable::entryAt(uint32_t Offset) const {
  auto It = llvm::lower_bound(Entries, Offset,
                              [](const std::pair<uint32_t, PackFileEntry> &E,
                                 uint32_t V) { return E.first < V; });
  if (It == Entries.end() || It->first != Offset)
    return std::nullopt;
  return It->second;
}
