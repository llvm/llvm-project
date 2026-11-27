//===- Materialize.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Restores the original bytes of a packed object.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CVPack/Materialize.h"
#include "COFFSectionTable.h"
#include "CVPackUtils.h"
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/DebugInfo/CVPack/COFFEdit.h"
#include "llvm/DebugInfo/CVPack/FunctionDebug.h"
#include "llvm/DebugInfo/CVPack/IndexRemap.h"
#include "llvm/DebugInfo/CVPack/PackFile.h"
#include "llvm/DebugInfo/CodeView/CVPacks.h"
#include "llvm/DebugInfo/CodeView/RecordSerialization.h"
#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/DebugInfo/CodeView/TypeHashing.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::codeview::cvpack;
using namespace llvm::support::endian;

Expected<std::vector<uint8_t>> llvm::codeview::cvpack::materializeObject(
    ArrayRef<uint8_t> Object, CVPackResolver &Resolver,
    ArrayRef<StringRef> ObjectDirectories) {
  auto TableOrErr = COFFSectionTable::create(Object);
  if (!TableOrErr)
    return TableOrErr.takeError();
  const COFFSectionTable &SectionTable = *TableOrErr;
  const COFFLayout &Layout = SectionTable.layout();
  auto sectionContents = [&](uint32_t Index) -> Expected<ArrayRef<uint8_t>> {
    if (Index == 0 || Index > Layout.SectionCount)
      return malformedRecord(
          "CodeView pack origin edit names a missing section");
    std::optional<ArrayRef<uint8_t>> Contents = SectionTable.contents(Index);
    if (!Contents)
      return malformedRecord("invalid COFF section raw-data range");
    return *Contents;
  };

  uint32_t TypeSectionIndex = 0;
  for (uint32_t I = 1; I <= Layout.SectionCount; ++I) {
    if (SectionTable.name(I) != ".debug$T")
      continue;
    if (TypeSectionIndex != 0)
      return malformedRecord("multiple .debug$T sections in packed object");
    TypeSectionIndex = I;
  }
  if (TypeSectionIndex == 0)
    return malformedRecord("object has no CodeView pack stream");
  auto PackedTypesOrErr = sectionContents(TypeSectionIndex);
  if (!PackedTypesOrErr)
    return PackedTypesOrErr.takeError();
  ArrayRef<uint8_t> PackedTypes = *PackedTypesOrErr;
  if (!isPackedTypeStream(PackedTypes))
    return malformedRecord("object has no CodeView pack stream");

  std::vector<CVType> Records;
  if (Error E = forEachCodeViewRecord<CVType>(
          PackedTypes.drop_front(sizeof(uint32_t)),
          [&](CVType Record) -> Error {
            if (Error Err = validateRecord(Record.data()))
              return Err;
            Records.push_back(Record);
            return Error::success();
          }))
    return std::move(E);
  if (Records.empty() || Records[0].kind() != TypeLeafKind::LF_CVPACK_LIST)
    return malformedRecord("packed type stream has no leading pack list");
  auto Id = decodePackList(Records[0]);
  if (!Id)
    return Id.takeError();
  if (Records.size() < 2 || Records[1].kind() != TypeLeafKind::LF_CVPACK_ORIGIN)
    return malformedRecord("packed object has no origin record");
  auto OriginData = decodeOrigin(Records[1]);
  if (!OriginData)
    return OriginData.takeError();

  auto PackOrErr =
      Resolver.get(*Id, /*VerifyContentId=*/true, ObjectDirectories);
  if (!PackOrErr)
    return PackOrErr.takeError();
  const PackFileView &Pack = (*PackOrErr)->View;

  std::vector<CVType> MapRecords;
  std::vector<CVType> InlineRecords;
  for (size_t I = 2; I != Records.size(); ++I) {
    TypeLeafKind Kind = Records[I].kind();
    if (Kind == TypeLeafKind::LF_CVPACK_MAP && InlineRecords.empty()) {
      MapRecords.push_back(Records[I]);
      continue;
    }
    if (Kind == TypeLeafKind::LF_CVPACK_LIST ||
        Kind == TypeLeafKind::LF_CVPACK_ORIGIN ||
        Kind == TypeLeafKind::LF_CVPACK_MAP)
      return malformedRecord("misplaced CodeView pack bookkeeping record");
    InlineRecords.push_back(Records[I]);
  }
  auto Map = decodeTypeMap(MapRecords);
  if (!Map)
    return Map.takeError();
  auto Segments =
      expandRangeMap(*Map, TypeIndex::FirstNonSimpleIndex + Pack.Types.size(),
                     InlineRecords.size(), TypeIndex::FirstNonSimpleIndex);
  if (!Segments)
    return Segments.takeError();

  // Map each pack index used by this object to its local index.
  IndexRemap IndexMap;
  uint64_t LocalIndex = TypeIndex::FirstNonSimpleIndex;
  for (const RangeMapSegment &Segment : *Segments) {
    if (Segment.IsPackReference)
      IndexMap.add(Segment.FirstIndex, Segment.Count, LocalIndex);
    LocalIndex += Segment.Count;
  }
  if (Error E = IndexMap.finalize())
    return std::move(E);

  // The restored type stream, built in place: each pack record is copied and
  // renumbered into the object's index space.
  std::vector<uint8_t> OriginalTypes;
  OriginalTypes.reserve(PackedTypes.size() * 4);
  appendLE<uint32_t>(OriginalTypes, COFF::DEBUG_SECTION_MAGIC);
  std::vector<uint32_t> RecordOffsets;
  RecordOffsets.reserve(LocalIndex - TypeIndex::FirstNonSimpleIndex);
  size_t InlineIndex = 0;
  for (const RangeMapSegment &Segment : *Segments) {
    for (uint64_t I = 0; I != Segment.Count; ++I) {
      size_t Offset = OriginalTypes.size();
      RecordOffsets.push_back(static_cast<uint32_t>(Offset));
      if (!Segment.IsPackReference) {
        appendBytes(OriginalTypes, InlineRecords[InlineIndex++].RecordData);
        continue;
      }
      ArrayRef<uint8_t> Source =
          Pack.Types[Segment.FirstIndex + I - TypeIndex::FirstNonSimpleIndex]
              .RecordData;
      appendBytes(OriginalTypes, Source);
      MutableArrayRef<uint8_t> Copy(OriginalTypes.data() + Offset,
                                    Source.size());
      if (Error E = remapTypeIndices(Copy, false, IndexMap))
        return std::move(E);
    }
  }
  auto recordAt = [&](size_t I) {
    size_t End = I + 1 == RecordOffsets.size() ? OriginalTypes.size()
                                               : RecordOffsets[I + 1];
    return ArrayRef<uint8_t>(OriginalTypes)
        .slice(RecordOffsets[I], End - RecordOffsets[I]);
  };

  std::vector<SectionReplacement> Replacements;
  std::map<uint32_t, std::vector<const OriginEdit *>> SymbolEdits;
  bool RemoveLastSection = false;
  auto sectionNamed = [&](uint32_t Index, StringRef Name) {
    return SectionTable.name(Index) == Name;
  };
  ArrayRef<uint8_t> Directive = arrayRefFromStringRef(PackedObjectDirective);
  for (const OriginEdit &Edit : OriginData->Edits) {
    if (!Edit.RestorationFragments.empty() &&
        Edit.Kind != OriginEditKind::DebugH)
      return malformedRecord("unsupported CodeView pack restoration fragments");
    auto Contents = sectionContents(Edit.SectionIndex);
    if (!Contents)
      return Contents.takeError();
    switch (Edit.Kind) {
    case OriginEditKind::TypeStream: {
      if (Edit.SectionIndex != TypeSectionIndex || Edit.OriginalOffset != 0 ||
          Edit.PackedLength != Contents->size())
        return malformedRecord("invalid type-stream origin edit");
      SectionReplacement R;
      R.SectionIndex = Edit.SectionIndex;
      R.Data = OriginalTypes;
      if (R.Data.size() != Edit.OriginalLength)
        return malformedRecord("materialized type stream has the wrong size");
      Replacements.push_back(std::move(R));
      break;
    }
    case OriginEditKind::DebugH: {
      if (Edit.OriginalOffset != 0 || Edit.PackedLength != Contents->size() ||
          ArrayRef<uint8_t>(OriginData->DebugHHeader) != *Contents)
        return malformedRecord("invalid .debug$H origin edit");
      std::vector<CVType> Types;
      Types.reserve(RecordOffsets.size());
      for (size_t I = 0; I != RecordOffsets.size(); ++I)
        Types.emplace_back(recordAt(I));
      SectionReplacement R;
      R.SectionIndex = Edit.SectionIndex;
      appendBytes(R.Data, *Contents);
      for (const GloballyHashedType &Hash :
           GloballyHashedType::hashTypes(Types))
        appendBytes(R.Data, Hash.Hash);
      if (R.Data.size() != Edit.OriginalLength)
        return malformedRecord("materialized .debug$H has the wrong size");
      // Stored hashes that differ from the recomputed ones.
      for (const RestorationFragment &Fragment : Edit.RestorationFragments)
        std::copy(Fragment.Bytes.begin(), Fragment.Bytes.end(),
                  R.Data.begin() + Fragment.Offset);
      Replacements.push_back(std::move(R));
      break;
    }
    case OriginEditKind::SymbolStream:
      SymbolEdits[Edit.SectionIndex].push_back(&Edit);
      break;
    case OriginEditKind::Directive: {
      // Remove the inserted directive, moving any relocations through spans.
      if (!sectionNamed(Edit.SectionIndex, ".drectve") ||
          Edit.OriginalLength != 0 || Edit.PackedLength != Directive.size() ||
          !containsRange(*Contents, Edit.OriginalOffset, Edit.PackedLength) ||
          Contents->slice(Edit.OriginalOffset, Edit.PackedLength) != Directive)
        return malformedRecord("invalid directive origin edit");
      uint32_t At = static_cast<uint32_t>(Edit.OriginalOffset);
      uint32_t Rest = Contents->size() - At - Edit.PackedLength;
      SectionReplacement R;
      R.SectionIndex = Edit.SectionIndex;
      appendBytes(R.Data, Contents->take_front(At));
      appendBytes(R.Data, Contents->drop_front(At + Edit.PackedLength));
      if (At)
        R.Spans.push_back({0, 0, At});
      if (Rest)
        R.Spans.push_back({At + Edit.PackedLength, At, Rest});
      Replacements.push_back(std::move(R));
      break;
    }
    case OriginEditKind::AddedDirectiveSection:
      if (Edit.SectionIndex != Layout.SectionCount ||
          !sectionNamed(Edit.SectionIndex, ".drectve") ||
          Edit.OriginalOffset != 0 || Edit.OriginalLength != 0 ||
          Edit.PackedLength != Directive.size() || *Contents != Directive)
        return malformedRecord("invalid added directive section origin edit");
      RemoveLastSection = true;
      break;
    }
  }

  for (auto &[SectionIndex, Edits] : SymbolEdits) {
    auto PackedOrErr = sectionContents(SectionIndex);
    if (!PackedOrErr)
      return PackedOrErr.takeError();
    ArrayRef<uint8_t> Packed = *PackedOrErr;
    SectionReplacement R;
    R.SectionIndex = SectionIndex;
    auto keep = [&](size_t From, size_t Size) {
      if (Size != 0)
        R.Spans.push_back({static_cast<uint32_t>(From),
                           static_cast<uint32_t>(R.Data.size()),
                           static_cast<uint32_t>(Size)});
      appendBytes(R.Data, Packed.slice(From, Size));
    };
    size_t Cursor = 0;
    int64_t Delta = 0;
    for (const OriginEdit *Edit : Edits) {
      int64_t PackedOffset = int64_t(Edit->OriginalOffset) + Delta;
      if (PackedOffset < int64_t(Cursor) || Edit->PackedLength < 8 ||
          !containsRange(Packed, PackedOffset, Edit->PackedLength))
        return malformedRecord("invalid symbol-stream origin edit");
      keep(Cursor, PackedOffset - Cursor);
      ArrayRef<uint8_t> Subsection =
          Packed.slice(PackedOffset, Edit->PackedLength);
      if (read32le(Subsection.data()) !=
              uint32_t(DebugSubsectionKind::Symbols) ||
          read32le(Subsection.data() + 4) != Subsection.size() - 8)
        return malformedRecord("symbol-stream edit is not a symbol subsection");
      std::vector<CVSymbol> Symbols;
      if (Error E = forEachCodeViewRecord<CVSymbol>(
              Subsection.drop_front(8), [&](CVSymbol Symbol) -> Error {
                Symbols.push_back(Symbol);
                return Error::success();
              }))
        return std::move(E);
      size_t MapCount = 0;
      while (MapCount != Symbols.size() &&
             Symbols[MapCount].kind() == SymbolKind::S_CVPACK_MAP)
        ++MapCount;
      auto SymbolMap =
          decodeSymbolMap(ArrayRef<CVSymbol>(Symbols).take_front(MapCount));
      if (!SymbolMap)
        return SymbolMap.takeError();
      auto SymbolSegments = expandRangeMap(*SymbolMap, Pack.Symbols.size(),
                                           Symbols.size() - MapCount, 0);
      if (!SymbolSegments)
        return SymbolSegments.takeError();

      size_t HeaderOffset = R.Data.size();
      appendBytes(R.Data, Subsection.take_front(8));
      size_t Next = MapCount;
      for (const RangeMapSegment &Segment : *SymbolSegments) {
        for (uint64_t I = 0; I != Segment.Count; ++I) {
          if (!Segment.IsPackReference) {
            CVSymbol Symbol = Symbols[Next++];
            keep(Symbol.data().data() - Packed.data(), Symbol.length());
            continue;
          }
          ArrayRef<uint8_t> Source =
              Pack.Symbols[Segment.FirstIndex + I].data();
          std::vector<uint8_t> Copy(Source.begin(), Source.end());
          if (Error E = remapTypeIndices(Copy, true, IndexMap))
            return std::move(E);
          appendBytes(R.Data, Copy);
        }
      }
      write32le(R.Data.data() + HeaderOffset + 4,
                static_cast<uint32_t>(R.Data.size() - HeaderOffset - 8));
      if (R.Data.size() - HeaderOffset != Edit->OriginalLength)
        return malformedRecord(
            "materialized symbol subsection has the wrong size");
      Cursor = PackedOffset + Edit->PackedLength;
      Delta += int64_t(Edit->PackedLength) - int64_t(Edit->OriginalLength);
    }
    keep(Cursor, Packed.size() - Cursor);
    Replacements.push_back(std::move(R));
  }

  // COMDAT .debug$S sections whose content moved to a pack. Their relocations
  // were kept at their original offsets.
  std::optional<ObjectFileTable> Files;
  auto sectionIs = [&](uint32_t I, bool Comdat) {
    return SectionTable.name(I) == ".debug$S" &&
           SectionTable.isComdat(I) == Comdat;
  };
  for (uint32_t I = 1; I <= Layout.SectionCount; ++I) {
    if (!sectionIs(I, true))
      continue;
    auto Contents = sectionContents(I);
    if (!Contents)
      return Contents.takeError();
    std::optional<FunctionRef> Ref = decodeFunctionRef(*Contents);
    if (!Ref)
      continue;
    if (!Files) {
      std::vector<ArrayRef<uint8_t>> Sections;
      for (uint32_t J = 1; J <= Layout.SectionCount; ++J)
        if (sectionIs(J, false)) {
          auto Data = sectionContents(J);
          if (!Data)
            return Data.takeError();
          Sections.push_back(*Data);
        }
      auto Table = ObjectFileTable::create(Sections);
      if (!Table)
        return Table.takeError();
      Files = std::move(*Table);
    }
    if (Ref->Slot != 0 || Ref->Ordinal >= Pack.Functions.size())
      return malformedRecord("function reference names a missing pack entry");
    ArrayRef<uint8_t> Blob = Pack.Functions[Ref->Ordinal];
    SectionReplacement R;
    R.SectionIndex = I;
    R.Data.reserve(Ref->OriginalSize);
    appendLE<uint32_t>(R.Data, COFF::DEBUG_SECTION_MAGIC);
    if (Error E = transformFunctionDebug(
            Blob, R.Data,
            [&](uint32_t Index, uint32_t) -> Expected<uint32_t> {
              if (std::optional<uint32_t> Local = IndexMap.lookup(Index))
                return *Local;
              return malformedRecord(
                  "function debug info references a pack type absent from "
                  "the map");
            },
            [&](uint32_t File, uint32_t) -> Expected<uint32_t> {
              std::optional<uint32_t> Offset;
              if (File < Pack.Files.size())
                Offset = Files->lookup(Pack.Files[File]);
              if (!Offset)
                return malformedRecord("pack file is absent from the object");
              return *Offset;
            },
            FileOperandEncoding::Minimal))
      return std::move(E);
    if (R.Data.size() != Ref->OriginalSize)
      return malformedRecord("function reference has the wrong size");
    R.Spans.push_back({0, 0, Ref->OriginalSize});
    Replacements.push_back(std::move(R));
  }

  auto Result = replaceCOFFSections(Object, Replacements);
  if (!Result)
    return Result.takeError();
  if (RemoveLastSection) {
    Result = removeLastCOFFSection(*Result);
    if (!Result)
      return Result.takeError();
  }
  if (Result->size() != OriginData->OriginalFileSize ||
      BLAKE3::hash(*Result) != OriginData->OriginalFileHash)
    return malformedRecord(
        "materialized object does not match its origin hash");
  return Result;
}
