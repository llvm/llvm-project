//===- PackFile.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Reads and writes CodeView pack files, and finds them by content id.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CVPack/PackFile.h"
#include "COFFSectionTable.h"
#include "CVPackUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/DebugInfo/CodeView/CVPacks.h"
#include "llvm/DebugInfo/CodeView/RecordSerialization.h"
#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/DebugInfo/CodeView/TypeHashing.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include <map>
#include <mutex>

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::codeview::cvpack;
using namespace llvm::support::endian;

namespace {
struct COFFSectionView {
  StringRef Name;
  ArrayRef<uint8_t> Contents;
};
} // namespace

static Expected<std::vector<COFFSectionView>>
getPackSections(ArrayRef<uint8_t> FileData) {
  auto Table = COFFSectionTable::create(FileData);
  if (!Table)
    return Table.takeError();
  const COFFLayout &Layout = Table->layout();
  if (Layout.IsBigObj)
    return malformedRecord("CodeView pack COFF file is a bigobj");
  const auto &FileHeader =
      *reinterpret_cast<const object::coff_file_header *>(FileData.data());
  if (FileHeader.Machine == COFF::IMAGE_FILE_MACHINE_UNKNOWN)
    return malformedRecord("CodeView pack COFF file has no machine");
  if (Table->size() != 2 && Table->size() != 3)
    return malformedRecord(
        "CodeView pack must have two or three COFF sections");
  if (Layout.SymbolTable != 0 || Table->object().getRawNumberOfSymbols() != 0)
    return malformedRecord("CodeView pack COFF file has a symbol table");
  if (FileHeader.SizeOfOptionalHeader != 0)
    return malformedRecord("CodeView pack COFF file has an optional header");
  size_t SectionTableEnd =
      Table->headerOffset(Table->size()) + sizeof(object::coff_section);

  std::vector<COFFSectionView> Sections;
  Sections.reserve(Table->size());
  std::vector<std::pair<uint64_t, uint64_t>> RawRanges;
  for (uint32_t I = 1; I <= Table->size(); ++I) {
    const object::coff_section &Header = Table->header(I);
    StringRef Name = Table->name(I);
    if (Name != ".debug$T" && Name != ".debug$S" && Name != ".debug$H")
      return malformedRecord(
          "CodeView pack COFF file has a non-CodeView section");
    for (const COFFSectionView &Existing : Sections)
      if (Existing.Name == Name)
        return malformedRecord(
            "duplicate CodeView section in CodeView pack COFF file");
    uint32_t RawSize = Header.SizeOfRawData;
    uint32_t RawOffset = Header.PointerToRawData;
    uint32_t Characteristics = Header.Characteristics;
    uint32_t RequiredCharacteristics =
        COFF::IMAGE_SCN_CNT_INITIALIZED_DATA | COFF::IMAGE_SCN_MEM_DISCARDABLE |
        COFF::IMAGE_SCN_MEM_READ | COFF::IMAGE_SCN_ALIGN_4BYTES;
    uint32_t ForbiddenCharacteristics =
        COFF::IMAGE_SCN_CNT_CODE | COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA |
        COFF::IMAGE_SCN_MEM_EXECUTE | COFF::IMAGE_SCN_MEM_WRITE;
    if ((Characteristics & RequiredCharacteristics) !=
            RequiredCharacteristics ||
        (Characteristics & ForbiddenCharacteristics) != 0)
      return malformedRecord(
          "invalid characteristics on CodeView pack section");
    if (Header.PointerToRelocations != 0 || Header.PointerToLinenumbers != 0 ||
        Header.NumberOfRelocations != 0 || Header.NumberOfLinenumbers != 0)
      return malformedRecord(
          "CodeView pack COFF section has relocations or line numbers");
    if (RawSize != 0 && (RawOffset < SectionTableEnd || (RawOffset & 3) != 0 ||
                         !containsRange(FileData, RawOffset, RawSize)))
      return malformedRecord(
          "invalid raw-data range in CodeView pack COFF section");
    if (RawSize != 0) {
      uint64_t RawEnd = uint64_t(RawOffset) + RawSize;
      for (const auto &Range : RawRanges)
        if (RawOffset < Range.second && Range.first < RawEnd)
          return malformedRecord(
              "overlapping raw sections in CodeView pack COFF file");
      RawRanges.emplace_back(RawOffset, RawEnd);
    }
    Sections.push_back({Name, FileData.slice(RawOffset, RawSize)});
  }
  return Sections;
}

static Error validateRawCodeViewRecords(ArrayRef<ArrayRef<uint8_t>> Records,
                                        bool IsSymbol) {
  for (ArrayRef<uint8_t> Bytes : Records) {
    if (Error E = validateRecord(Bytes))
      return E;
    uint16_t Kind = read16le(Bytes.data() + 2);
    if (IsSymbol) {
      if (Kind != SymbolKind::S_UDT && Kind != SymbolKind::S_CONSTANT)
        return malformedRecord(
            "CodeView pack contains an unsupported symbol record");
    } else if (Kind == TypeLeafKind::LF_CVPACK_LIST ||
               Kind == TypeLeafKind::LF_CVPACK_ORIGIN ||
               Kind == TypeLeafKind::LF_CVPACK_MAP) {
      return malformedRecord(
          "CodeView pack inline records contain a leading record");
    }
  }
  return Error::success();
}

Expected<PackFileView>
llvm::codeview::cvpack::readPackFile(ArrayRef<uint8_t> FileData,
                                     std::optional<PackId> ExpectedContentId,
                                     bool VerifyContentId) {
  auto SectionsOrErr = getPackSections(FileData);
  if (!SectionsOrErr)
    return SectionsOrErr.takeError();

  const COFFSectionView *TypeSection = nullptr;
  const COFFSectionView *SymbolSection = nullptr;
  const COFFSectionView *HashSection = nullptr;
  for (const COFFSectionView &Section : *SectionsOrErr) {
    if (Section.Name == ".debug$T")
      TypeSection = &Section;
    else if (Section.Name == ".debug$S")
      SymbolSection = &Section;
    else if (Section.Name == ".debug$H")
      HashSection = &Section;
  }
  if (!TypeSection || TypeSection->Contents.size() < 4 ||
      read32le(TypeSection->Contents.data()) != COFF::DEBUG_SECTION_MAGIC)
    return malformedRecord("CodeView pack is missing a valid .debug$T stream");
  if (!SymbolSection)
    return malformedRecord("CodeView pack is missing its .debug$S section");

  PackFileView Pack;
  Pack.FileData = FileData;
  Pack.Machine =
      reinterpret_cast<const object::coff_file_header *>(FileData.data())
          ->Machine;
  if (ExpectedContentId && !VerifyContentId)
    Pack.ContentId = *ExpectedContentId;
  else
    Pack.ContentId = BLAKE3::hash(FileData);
  if (ExpectedContentId && Pack.ContentId != *ExpectedContentId)
    return malformedRecord("CodeView pack content id does not match its file");

  std::vector<CVType> Records;
  // Records average well over 32 bytes; reserving avoids regrowing the vector
  // over millions of records.
  Records.reserve(TypeSection->Contents.size() / 32);
  if (Error E = forEachCodeViewRecord<CVType>(
          TypeSection->Contents.drop_front(sizeof(uint32_t)),
          [&](CVType Record) -> Error {
            if (Error Err = validateRecord(Record.data()))
              return Err;
            Records.push_back(Record);
            return Error::success();
          }))
    return std::move(E);

  Pack.Types.reserve(Records.size());
  for (CVType Record : Records) {
    TypeLeafKind Kind = Record.kind();
    if (Kind == TypeLeafKind::LF_CVPACK_LIST ||
        Kind == TypeLeafKind::LF_CVPACK_ORIGIN ||
        Kind == TypeLeafKind::LF_CVPACK_MAP)
      return malformedRecord("CodeView pack cannot contain a pack record");
    Pack.Types.push_back(Record);
  }

  if (HashSection) {
    ArrayRef<uint8_t> H = HashSection->Contents;
    if (H.size() < 8 ||
        read32le(H.data()) != COFF::DEBUG_HASHES_SECTION_MAGIC ||
        read16le(H.data() + 4) != 0 ||
        read16le(H.data() + 6) != 2 /* BLAKE3 */ ||
        H.size() - 8 != 8 * uint64_t(Pack.Types.size()))
      return malformedRecord("invalid .debug$H section in CodeView pack");
    Pack.GlobalHashes = H.drop_front(8);
  }

  if (SymbolSection) {
    ArrayRef<uint8_t> Bytes = SymbolSection->Contents;
    if (Bytes.size() < 4 || read32le(Bytes.data()) != COFF::DEBUG_SECTION_MAGIC)
      return malformedRecord("CodeView pack has a malformed .debug$S stream");
    size_t Offset = 4;
    unsigned SubsectionCount = 0;
    while (Offset != Bytes.size()) {
      if (Bytes.size() - Offset < 8)
        return malformedRecord("truncated CodeView pack symbol subsection");
      uint32_t Kind = read32le(Bytes.data() + Offset);
      uint32_t SubsectionSize = read32le(Bytes.data() + Offset + 4);
      size_t DataOffset = Offset + 8;
      if (SubsectionSize > Bytes.size() - DataOffset)
        return malformedRecord(
            "truncated CodeView pack symbol subsection data");
      ArrayRef<uint8_t> Data = Bytes.slice(DataOffset, SubsectionSize);
      ++SubsectionCount;
      if (Kind == PackFilesSubsectionKind && SubsectionCount == 2) {
        auto Files = decodePackFiles(Data);
        if (!Files)
          return Files.takeError();
        Pack.Files = std::move(*Files);
      } else if (Kind == PackFunctionsSubsectionKind &&
                 (SubsectionCount == 2 ||
                  (SubsectionCount == 3 && !Pack.Files.empty()))) {
        auto Functions = decodePackFunctions(Data);
        if (!Functions)
          return Functions.takeError();
        Pack.Functions = std::move(*Functions);
      } else if (Kind != uint32_t(DebugSubsectionKind::Symbols) ||
                 SubsectionCount != 1) {
        return malformedRecord(
            "CodeView pack must have one symbols subsection");
      }
      if (Kind == uint32_t(DebugSubsectionKind::Symbols)) {
        if (Error E = forEachCodeViewRecord<CVSymbol>(
                Data, [&](CVSymbol Symbol) -> Error {
                  if (Error Err = validateRecord(Symbol.data()))
                    return Err;
                  if (Symbol.kind() != SymbolKind::S_UDT &&
                      Symbol.kind() != SymbolKind::S_CONSTANT)
                    return malformedRecord(
                        "CodeView pack contains an unsupported symbol record");
                  Pack.Symbols.push_back(Symbol);
                  return Error::success();
                }))
          return std::move(E);
      }
      size_t SubsectionEnd = DataOffset + SubsectionSize;
      size_t AlignedEnd = (SubsectionEnd + 3) & ~size_t(3);
      if (AlignedEnd > Bytes.size())
        return malformedRecord(
            "invalid CodeView pack symbol subsection padding");
      Offset = AlignedEnd;
    }
  }
  return Pack;
}

Expected<PackWriteResult>
llvm::codeview::cvpack::writePackFile(const PackContents &Contents) {
  if (Error E = validateRawCodeViewRecords(Contents.Types, false))
    return std::move(E);
  if (Error E = validateRawCodeViewRecords(Contents.Symbols, true))
    return std::move(E);

  std::vector<uint8_t> TypeData;
  appendLE<uint32_t>(TypeData, COFF::DEBUG_SECTION_MAGIC);
  for (ArrayRef<uint8_t> Record : Contents.Types)
    TypeData.insert(TypeData.end(), Record.begin(), Record.end());

  std::vector<uint8_t> SymbolData;
  appendLE<uint32_t>(SymbolData, COFF::DEBUG_SECTION_MAGIC);
  appendLE<uint32_t>(SymbolData, uint32_t(DebugSubsectionKind::Symbols));
  appendLE<uint32_t>(SymbolData, 0);
  if (!Contents.Symbols.empty()) {
    uint64_t SymbolBytes = 0;
    for (ArrayRef<uint8_t> Record : Contents.Symbols)
      SymbolBytes += Record.size();
    if (SymbolBytes > std::numeric_limits<uint32_t>::max())
      return malformedRecord("CodeView pack symbol subsection is too large");
    write32le(SymbolData.data() + 8, static_cast<uint32_t>(SymbolBytes));
    for (ArrayRef<uint8_t> Record : Contents.Symbols)
      SymbolData.insert(SymbolData.end(), Record.begin(), Record.end());
    appendAlignedZeros(SymbolData);
  }
  // Function debug info: a files subsection and a functions subsection, each
  // a count followed by 4-byte aligned entries.
  auto writeSubsection = [&](uint32_t Kind, auto &&Body) -> Error {
    size_t Header = SymbolData.size();
    appendLE<uint32_t>(SymbolData, Kind);
    appendLE<uint32_t>(SymbolData, 0);
    Body();
    uint64_t Size = SymbolData.size() - Header - 8;
    if (Size > std::numeric_limits<uint32_t>::max())
      return malformedRecord("CodeView pack function subsection is too large");
    write32le(SymbolData.data() + Header + 4, static_cast<uint32_t>(Size));
    appendAlignedZeros(SymbolData);
    return Error::success();
  };
  if (!Contents.Files.empty()) {
    auto Body = encodePackFiles(Contents.Files);
    if (!Body)
      return Body.takeError();
    if (Error E = writeSubsection(PackFilesSubsectionKind,
                                  [&] { appendBytes(SymbolData, *Body); }))
      return E;
  }
  if (!Contents.Functions.empty()) {
    // Written in place: the functions are most of a pack's bytes.
    if (Error E = writeSubsection(PackFunctionsSubsectionKind, [&] {
          encodePackFunctions(Contents.Functions, SymbolData);
        }))
      return E;
  }

  if (TypeData.size() > std::numeric_limits<uint32_t>::max() ||
      SymbolData.size() > std::numeric_limits<uint32_t>::max())
    return malformedRecord("CodeView pack COFF section is too large");
  std::vector<uint8_t> HashData;
  if (Contents.WriteGlobalHashes) {
    std::vector<CVType> Types(Contents.Types.begin(), Contents.Types.end());
    appendLE<uint32_t>(HashData, COFF::DEBUG_HASHES_SECTION_MAGIC);
    appendLE<uint16_t>(HashData, 0);
    appendLE<uint16_t>(HashData, 2); // BLAKE3
    for (const GloballyHashedType &Hash : GloballyHashedType::hashTypes(Types))
      appendBytes(HashData, Hash.Hash);
  }
  uint16_t SectionCount = HashData.empty() ? 2 : 3;
  std::vector<uint8_t> FileData(sizeof(object::coff_file_header) +
                                    SectionCount * sizeof(object::coff_section),
                                0);
  auto &FileHeader = headerAt<object::coff_file_header>(FileData, 0);
  FileHeader.Machine = Contents.Machine;
  FileHeader.NumberOfSections = SectionCount;

  auto writeSection = [&](uint32_t Index, StringRef Name,
                          ArrayRef<uint8_t> Data) -> Error {
    size_t HeaderOffset = sizeof(object::coff_file_header) +
                          (Index - 1) * sizeof(object::coff_section);
    std::copy(Name.begin(), Name.end(), FileData.begin() + HeaderOffset);
    appendAlignedZeros(FileData);
    if (FileData.size() > std::numeric_limits<uint32_t>::max())
      return malformedRecord("CodeView pack COFF file is too large");
    auto &Header = headerAt<object::coff_section>(FileData, HeaderOffset);
    Header.SizeOfRawData = static_cast<uint32_t>(Data.size());
    Header.PointerToRawData = static_cast<uint32_t>(FileData.size());
    Header.Characteristics =
        COFF::IMAGE_SCN_CNT_INITIALIZED_DATA | COFF::IMAGE_SCN_MEM_DISCARDABLE |
        COFF::IMAGE_SCN_MEM_READ | COFF::IMAGE_SCN_ALIGN_4BYTES;
    FileData.insert(FileData.end(), Data.begin(), Data.end());
    return Error::success();
  };

  if (Error E = writeSection(1, ".debug$T", TypeData))
    return std::move(E);
  if (Error E = writeSection(2, ".debug$S", SymbolData))
    return std::move(E);
  if (!HashData.empty())
    if (Error E = writeSection(3, ".debug$H", HashData))
      return std::move(E);
  if (FileData.size() > std::numeric_limits<uint32_t>::max())
    return malformedRecord("CodeView pack COFF file is too large");

  PackWriteResult Result;
  Result.ContentId = BLAKE3::hash(FileData);
  Result.FileData = std::move(FileData);
  auto Verification = readPackFile(Result.FileData, Result.ContentId);
  if (!Verification)
    return Verification.takeError();
  return Result;
}

struct CVPackResolver::Impl {
  std::vector<std::string> Directories;
  std::mutex Mutex;
  std::map<PackId, std::unique_ptr<ResolvedCVPack>> Packs;

  Expected<ResolvedCVPack *> get(const PackId &Id, bool VerifyContentId,
                                 ArrayRef<StringRef> ObjectDirectories) {
    auto Existing = Packs.find(Id);
    if (Existing != Packs.end())
      return Existing->second.get();
    std::string Name = toHex(Id, /*LowerCase=*/true) + PackFileExtension.str();
    std::unique_ptr<MemoryBuffer> Buffer;
    // Each directory once, in the order searched.
    SmallVector<StringRef, 16> Searched;
    auto search = [&](StringRef Directory) {
      if (Buffer || is_contained(Searched, Directory))
        return;
      Searched.push_back(Directory);
      SmallString<256> Path(Directory);
      sys::path::append(Path, Name);
      if (auto File = MemoryBuffer::getFile(Path, false, false))
        Buffer = std::move(*File);
    };
    for (StringRef Directory : ObjectDirectories)
      search(Directory);
    for (const std::string &Directory : Directories)
      search(Directory);
    if (!Buffer) {
      std::string SearchPaths;
      for (StringRef Directory : Searched) {
        if (!SearchPaths.empty())
          SearchPaths += ", ";
        SearchPaths += Directory.empty() ? "." : Directory.str();
      }
      return malformedRecord(
          "missing CodeView pack " + Name + "; searched: " +
          (SearchPaths.empty() ? std::string("no directory") : SearchPaths));
    }
    auto Pack = std::make_unique<ResolvedCVPack>();
    Pack->Buffer = std::move(Buffer);
    auto View = readPackFile(arrayRefFromStringRef(Pack->Buffer->getBuffer()),
                             Id, VerifyContentId);
    if (!View)
      return View.takeError();
    Pack->View = std::move(*View);
    ResolvedCVPack *Raw = Pack.get();
    Packs.emplace(Id, std::move(Pack));
    return Raw;
  }
};

CVPackResolver::CVPackResolver(ArrayRef<StringRef> Directories)
    : P(std::make_unique<Impl>()) {
  for (StringRef Directory : Directories)
    P->Directories.push_back(Directory.str());
}

CVPackResolver::~CVPackResolver() = default;

Expected<const ResolvedCVPack *>
CVPackResolver::get(const PackId &Id, bool VerifyContentId,
                    ArrayRef<StringRef> ObjectDirectories) {
  std::lock_guard<std::mutex> Lock(P->Mutex);
  return P->get(Id, VerifyContentId, ObjectDirectories);
}
