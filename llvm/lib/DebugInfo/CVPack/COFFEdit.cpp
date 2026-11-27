//===- COFFEdit.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Rewrites the sections of a COFF object in place: the headers are read through
// object::COFFObjectFile, and every file offset is moved by hand, so that the
// bytes that are not edited stay exactly as they were.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CVPack/COFFEdit.h"
#include "COFFSectionTable.h"
#include "CVPackUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/Endian.h"
#include <array>
#include <cstddef>
#include <limits>

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::codeview::cvpack;
using namespace llvm::support::endian;

/// Every file offset field of a COFF object: the raw data, relocations and
/// line numbers of each section, and the symbol table. Zero means "none".
static std::vector<size_t> fileOffsetFields(const COFFLayout &Layout) {
  std::vector<size_t> Fields;
  for (uint32_t I = 0; I != Layout.SectionCount; ++I) {
    size_t Header =
        Layout.SectionTableOffset + size_t(I) * sizeof(object::coff_section);
    for (size_t Field : {offsetof(object::coff_section, PointerToRawData),
                         offsetof(object::coff_section, PointerToRelocations),
                         offsetof(object::coff_section, PointerToLinenumbers)})
      Fields.push_back(Header + Field);
  }
  Fields.push_back(Layout.SymbolTablePointerOffset);
  return Fields;
}

Expected<COFFSectionTable> COFFSectionTable::create(ArrayRef<uint8_t> Object) {
  auto ObjOrErr = object::COFFObjectFile::create(
      MemoryBufferRef(toStringRef(Object), "COFF object"));
  if (!ObjOrErr)
    return ObjOrErr.takeError();
  COFFSectionTable Table;
  Table.Obj = std::move(*ObjOrErr);
  const object::COFFObjectFile &Obj = *Table.Obj;
  // The file header of an image is not at the start of the file.
  if (Obj.getDOSHeader())
    return malformedRecord("not a COFF object file");

  COFFLayout &Layout = Table.Layout;
  Layout.IsBigObj = Obj.getCOFFBigObjHeader() != nullptr;
  Layout.Machine = Obj.getMachine();
  Layout.SectionCount = Obj.getNumberOfSections();
  Layout.SymbolTable = Obj.getPointerToSymbolTable();
  if (Layout.IsBigObj) {
    Layout.SectionTableOffset = sizeof(object::coff_bigobj_file_header);
    Layout.SymbolTablePointerOffset =
        offsetof(object::coff_bigobj_file_header, PointerToSymbolTable);
  } else {
    Layout.SectionTableOffset = sizeof(object::coff_file_header) +
                                Obj.getCOFFHeader()->SizeOfOptionalHeader;
    Layout.SymbolTablePointerOffset =
        offsetof(object::coff_file_header, PointerToSymbolTable);
  }
  return std::move(Table);
}

size_t COFFSectionTable::headerOffset(uint32_t Index) const {
  assert(Index != 0 && Index <= Layout.SectionCount && "no such section");
  return Layout.SectionTableOffset +
         size_t(Index - 1) * sizeof(object::coff_section);
}

const object::coff_section &COFFSectionTable::header(uint32_t Index) const {
  assert(Index != 0 && Index <= Layout.SectionCount && "no such section");
  return *cantFail(Obj->getSection(Index));
}

StringRef COFFSectionTable::name(uint32_t Index) const {
  Expected<StringRef> Name = Obj->getSectionName(&header(Index));
  if (Name)
    return *Name;
  consumeError(Name.takeError());
  return StringRef();
}

bool COFFSectionTable::isComdat(uint32_t Index) const {
  return header(Index).Characteristics & COFF::IMAGE_SCN_LNK_COMDAT;
}

std::optional<ArrayRef<uint8_t>>
COFFSectionTable::contents(uint32_t Index) const {
  const object::coff_section &Header = header(Index);
  if (Header.SizeOfRawData == 0)
    return ArrayRef<uint8_t>();
  if (Header.PointerToRawData == 0)
    return std::nullopt;
  ArrayRef<uint8_t> Contents;
  if (Error E = Obj->getSectionContents(&Header, Contents)) {
    consumeError(std::move(E));
    return std::nullopt;
  }
  return Contents;
}

std::optional<ArrayRef<object::coff_relocation>>
COFFSectionTable::relocations(uint32_t Index) const {
  const object::coff_section &Header = header(Index);
  // A count that overflows into the first entry is not supported.
  if (Header.Characteristics & COFF::IMAGE_SCN_LNK_NRELOC_OVFL)
    return std::nullopt;
  ArrayRef<object::coff_relocation> Relocations = Obj->getRelocations(&Header);
  // COFFObjectFile gives no relocations for a table outside the object.
  if (Relocations.size() != Header.NumberOfRelocations)
    return std::nullopt;
  return Relocations;
}

Expected<std::vector<uint8_t>> llvm::codeview::cvpack::replaceCOFFSections(
    ArrayRef<uint8_t> Object, ArrayRef<SectionReplacement> Replacements) {
  auto Table = COFFSectionTable::create(Object);
  if (!Table)
    return Table.takeError();
  const COFFLayout &Layout = Table->layout();
  size_t SectionTableEnd = Layout.SectionTableOffset +
                           Layout.SectionCount * sizeof(object::coff_section);

  // The replaced raw-data ranges in file order. One pass copies the file with
  // the new contents; every file offset then moves by the size change of the
  // ranges that end before it.
  struct Item {
    const SectionReplacement *R;
    uint32_t RawOffset;
    uint32_t RawSize;
  };
  std::vector<Item> Items;
  Items.reserve(Replacements.size());
  for (const SectionReplacement &R : Replacements) {
    if (R.SectionIndex == 0 || R.SectionIndex > Layout.SectionCount)
      return malformedRecord("section replacement index is out of range");
    const object::coff_section &Header = Table->header(R.SectionIndex);
    uint32_t RawSize = Header.SizeOfRawData;
    uint32_t RawOffset = Header.PointerToRawData;
    if (RawOffset == 0)
      return malformedRecord("cannot replace a section without raw data");
    if (RawOffset < SectionTableEnd ||
        !containsRange(Object, RawOffset, RawSize))
      return malformedRecord("invalid COFF section raw-data range");
    if (R.Data.size() > std::numeric_limits<uint32_t>::max())
      return malformedRecord("replacement section is too large");
    Items.push_back({&R, RawOffset, RawSize});
  }
  llvm::sort(Items, [](const Item &A, const Item &B) {
    return std::make_pair(A.RawOffset, A.R->SectionIndex) <
           std::make_pair(B.RawOffset, B.R->SectionIndex);
  });
  std::vector<uint64_t> Ends;
  std::vector<int64_t> Deltas; // cumulative, through each item
  Ends.reserve(Items.size());
  Deltas.reserve(Items.size());
  int64_t Delta = 0;
  for (size_t I = 0; I != Items.size(); ++I) {
    const Item &It = Items[I];
    if (I && (Items[I - 1].R->SectionIndex == It.R->SectionIndex ||
              Ends.back() > It.RawOffset))
      return malformedRecord("overlapping section replacements");
    Ends.push_back(uint64_t(It.RawOffset) + It.RawSize);
    Delta += int64_t(It.R->Data.size()) - int64_t(It.RawSize);
    Deltas.push_back(Delta);
  }

  std::vector<uint8_t> Result;
  Result.reserve(std::max<int64_t>(int64_t(Object.size()) + Delta, 0));
  size_t Cursor = 0;
  for (const Item &It : Items) {
    Result.insert(Result.end(), Object.begin() + Cursor,
                  Object.begin() + It.RawOffset);
    Result.insert(Result.end(), It.R->Data.begin(), It.R->Data.end());
    Cursor = It.RawOffset + It.RawSize;
  }
  Result.insert(Result.end(), Object.begin() + Cursor, Object.end());

  auto shiftPointer = [&](size_t Offset) -> Error {
    uint32_t Pointer = read32le(Object.data() + Offset);
    if (Pointer == 0)
      return Error::success();
    size_t Before = llvm::upper_bound(Ends, uint64_t(Pointer)) - Ends.begin();
    if (Before == 0)
      return Error::success();
    int64_t Shifted = int64_t(Pointer) + Deltas[Before - 1];
    if (Shifted < 0 || Shifted > std::numeric_limits<uint32_t>::max())
      return malformedRecord("COFF file offset overflows");
    write32le(Result.data() + Offset, static_cast<uint32_t>(Shifted));
    return Error::success();
  };
  for (size_t Field : fileOffsetFields(Layout))
    if (Error E = shiftPointer(Field))
      return std::move(E);

  for (const Item &It : Items) {
    const SectionReplacement *R = It.R;
    auto &Header = headerAt<object::coff_section>(
        Result, Table->headerOffset(R->SectionIndex));
    Header.SizeOfRawData = static_cast<uint32_t>(R->Data.size());

    // Move this section's relocations through the kept spans.
    uint32_t RelocOffset = Header.PointerToRelocations;
    uint32_t RelocCount = Header.NumberOfRelocations;
    uint32_t Characteristics = Header.Characteristics;
    size_t FirstReloc = 0;
    if ((Characteristics & COFF::IMAGE_SCN_LNK_NRELOC_OVFL) &&
        RelocCount == 0xffff) {
      if (!containsRange(Result, RelocOffset, COFF::RelocationSize))
        return malformedRecord("truncated COFF relocation table");
      RelocCount = read32le(Result.data() + RelocOffset);
      FirstReloc = 1;
    }
    if (RelocCount == 0)
      continue;
    if (!containsRange(Result, RelocOffset,
                       uint64_t(RelocCount) * COFF::RelocationSize))
      return malformedRecord("truncated COFF relocation table");
    if (R->Spans.empty())
      return malformedRecord("relocated section rewritten without spans");
    for (size_t I = FirstReloc; I != RelocCount; ++I) {
      size_t Entry = RelocOffset + I * COFF::RelocationSize;
      uint32_t Address = read32le(Result.data() + Entry);
      auto SpanIt = llvm::upper_bound(
          R->Spans, Address,
          [](uint32_t A, const OffsetSpan &S) { return A < S.OldOffset; });
      if (SpanIt == R->Spans.begin())
        return malformedRecord("relocation lies outside the kept spans");
      --SpanIt;
      if (Address - SpanIt->OldOffset >= SpanIt->Size ||
          SpanIt->Size - (Address - SpanIt->OldOffset) < 2)
        return malformedRecord("relocation lies outside the kept spans");
      write32le(Result.data() + Entry,
                SpanIt->NewOffset + (Address - SpanIt->OldOffset));
    }
  }
  return Result;
}

static Error setSectionCount(std::vector<uint8_t> &Object,
                             const COFFLayout &Layout, uint64_t Count) {
  if (Layout.IsBigObj) {
    if (Count > uint64_t(std::numeric_limits<int32_t>::max()))
      return malformedRecord("too many COFF sections");
    headerAt<object::coff_bigobj_file_header>(Object, 0).NumberOfSections =
        static_cast<uint32_t>(Count);
  } else {
    if (Count > COFF::MaxNumberOfSections16)
      return malformedRecord("too many COFF sections");
    headerAt<object::coff_file_header>(Object, 0).NumberOfSections =
        static_cast<uint16_t>(Count);
  }
  return Error::success();
}

Expected<std::vector<uint8_t>> llvm::codeview::cvpack::appendCOFFSection(
    ArrayRef<uint8_t> Object, StringRef Name, uint32_t Characteristics,
    ArrayRef<uint8_t> Data) {
  if (Name.size() > COFF::NameSize)
    return malformedRecord("appended COFF section name is too long");
  auto Table = COFFSectionTable::create(Object);
  if (!Table)
    return Table.takeError();
  const COFFLayout &Layout = Table->layout();
  size_t TableEnd = Layout.SectionTableOffset +
                    size_t(Layout.SectionCount) * COFF::SectionSize;
  size_t DataAt = Layout.SymbolTable ? Layout.SymbolTable : Object.size();
  if (DataAt < TableEnd || DataAt > Object.size())
    return malformedRecord("invalid COFF symbol table offset");
  if (uint64_t(Object.size()) + COFF::SectionSize + Data.size() >
      std::numeric_limits<uint32_t>::max())
    return malformedRecord("COFF object is too large");

  // The new header goes after the last one and the data before the symbol
  // table: offsets past each insertion point move by its size.
  auto shifted = [&](uint32_t Offset) -> uint32_t {
    if (Offset == 0)
      return 0;
    uint64_t New = Offset;
    if (Offset >= TableEnd)
      New += COFF::SectionSize;
    if (Offset >= DataAt)
      New += Data.size();
    return static_cast<uint32_t>(New);
  };
  std::vector<uint8_t> Result;
  Result.reserve(Object.size() + COFF::SectionSize + Data.size());
  appendBytes(Result, Object.take_front(TableEnd));
  std::array<uint8_t, COFF::SectionSize> Header{};
  std::copy(Name.begin(), Name.end(), Header.begin());
  Result.insert(Result.end(), Header.begin(), Header.end());
  appendBytes(Result, Object.slice(TableEnd, DataAt - TableEnd));
  appendBytes(Result, Data);
  appendBytes(Result, Object.drop_front(DataAt));
  for (size_t Field : fileOffsetFields(Layout))
    write32le(Result.data() +
                  (Field < TableEnd ? Field : Field + COFF::SectionSize),
              shifted(read32le(Object.data() + Field)));
  auto &Added = headerAt<object::coff_section>(Result, TableEnd);
  Added.SizeOfRawData = static_cast<uint32_t>(Data.size());
  Added.PointerToRawData =
      Data.empty() ? 0 : static_cast<uint32_t>(DataAt + COFF::SectionSize);
  Added.Characteristics = Characteristics;
  if (Error E =
          setSectionCount(Result, Layout, uint64_t(Layout.SectionCount) + 1))
    return std::move(E);
  return Result;
}

Expected<std::vector<uint8_t>>
llvm::codeview::cvpack::removeLastCOFFSection(ArrayRef<uint8_t> Object) {
  auto Table = COFFSectionTable::create(Object);
  if (!Table)
    return Table.takeError();
  const COFFLayout &Layout = Table->layout();
  if (Layout.SectionCount == 0)
    return malformedRecord("COFF object has no section to remove");
  const object::coff_section &Last = Table->header(Layout.SectionCount);
  size_t Header = Table->headerOffset(Layout.SectionCount);
  uint32_t RawSize = Last.SizeOfRawData;
  uint32_t RawOffset = Last.PointerToRawData;
  if (Last.PointerToRelocations || Last.PointerToLinenumbers ||
      Last.NumberOfRelocations || Last.NumberOfLinenumbers)
    return malformedRecord("removed COFF section has relocations");
  if (RawOffset == 0)
    RawSize = 0;
  size_t TableEnd = Header + COFF::SectionSize;
  if ((RawSize && RawOffset < TableEnd) ||
      !containsRange(Object, RawOffset, RawSize))
    return malformedRecord("invalid COFF section raw-data range");
  size_t RawEnd = size_t(RawOffset) + RawSize;

  auto shifted = [&](uint32_t Offset) -> Expected<uint32_t> {
    if (Offset == 0)
      return 0;
    if ((Offset >= Header && Offset < TableEnd) ||
        (RawSize && Offset >= RawOffset && Offset < RawEnd))
      return malformedRecord("COFF offset points into the removed section");
    uint64_t New = Offset;
    if (Offset >= TableEnd)
      New -= COFF::SectionSize;
    if (RawSize && Offset >= RawEnd)
      New -= RawSize;
    return static_cast<uint32_t>(New);
  };
  std::vector<uint8_t> Result;
  Result.reserve(Object.size());
  appendBytes(Result, Object.take_front(Header));
  if (RawSize) {
    appendBytes(Result, Object.slice(TableEnd, RawOffset - TableEnd));
    appendBytes(Result, Object.drop_front(RawEnd));
  } else {
    appendBytes(Result, Object.drop_front(TableEnd));
  }
  COFFLayout Remaining = Layout;
  --Remaining.SectionCount;
  for (size_t Field : fileOffsetFields(Remaining)) {
    auto New = shifted(read32le(Object.data() + Field));
    if (!New)
      return New.takeError();
    write32le(Result.data() + Field, *New);
  }
  if (Error E = setSectionCount(Result, Layout, Remaining.SectionCount))
    return std::move(E);
  return Result;
}
