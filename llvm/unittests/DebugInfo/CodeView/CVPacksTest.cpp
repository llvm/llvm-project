//===- llvm/unittest/DebugInfo/CodeView/CVPacksTest.cpp -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/CVPacks.h"

#include "llvm/BinaryFormat/COFF.h"
#include "llvm/DebugInfo/CodeView/SymbolDumper.h"
#include "llvm/DebugInfo/CodeView/TypeDumpVisitor.h"
#include "llvm/DebugInfo/CodeView/TypeTableCollection.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ScopedPrinter.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::codeview::cvpack;

TEST(CVPacksTest, PackListRoundTrip) {
  PackId Id{};
  Id[0] = 0x12;
  Id[31] = 0x34;
  auto Encoded = encodePackList(Id);
  ASSERT_TRUE(static_cast<bool>(Encoded));
  auto Decoded = decodePackList(CVType(*Encoded));
  ASSERT_TRUE(static_cast<bool>(Decoded));
  EXPECT_EQ(*Decoded, Id);
}

TEST(CVPacksTest, PackListHoldsOnePack) {
  auto Encoded = encodePackList(PackId{});
  ASSERT_TRUE(static_cast<bool>(Encoded));
  // The count follows the record prefix and the 8-byte header.
  std::vector<uint8_t> Two = *Encoded;
  Two[12] = 2;
  auto Decoded = decodePackList(CVType(Two));
  EXPECT_FALSE(static_cast<bool>(Decoded));
  consumeError(Decoded.takeError());
}

TEST(CVPacksTest, TypeMapRoundTripAndExpansion) {
  RangeMap Map;
  Map.Entries = {{1, 0, 2, 0}, {0, 0, 1, 0}, {2, 0, 1, 3}};

  auto Encoded = encodeTypeMap(Map);
  ASSERT_TRUE(static_cast<bool>(Encoded));
  std::vector<CVType> Records;
  for (const std::vector<uint8_t> &Record : *Encoded)
    Records.emplace_back(Record);
  auto Decoded = decodeTypeMap(Records);
  ASSERT_TRUE(static_cast<bool>(Decoded));
  ASSERT_EQ(Decoded->Entries.size(), Map.Entries.size());
  for (size_t I = 0; I != Map.Entries.size(); ++I) {
    EXPECT_EQ(Decoded->Entries[I].Skip, Map.Entries[I].Skip);
    EXPECT_EQ(Decoded->Entries[I].Slot, Map.Entries[I].Slot);
    EXPECT_EQ(Decoded->Entries[I].Count, Map.Entries[I].Count);
    EXPECT_EQ(Decoded->Entries[I].FirstDelta, Map.Entries[I].FirstDelta);
  }

  auto Segments = expandRangeMap(*Decoded, 0x1007, 3, 0x1000);
  ASSERT_TRUE(static_cast<bool>(Segments));
  ASSERT_EQ(Segments->size(), 5u);
  EXPECT_FALSE((*Segments)[0].IsPackReference);
  EXPECT_EQ((*Segments)[0].Count, 1u);
  EXPECT_TRUE((*Segments)[1].IsPackReference);
  EXPECT_EQ((*Segments)[1].FirstIndex, 0x1000u);
  EXPECT_EQ((*Segments)[1].Count, 2u);
  EXPECT_TRUE((*Segments)[2].IsPackReference);
  EXPECT_EQ((*Segments)[2].FirstIndex, 0x1002u);
  EXPECT_FALSE((*Segments)[3].IsPackReference);
  EXPECT_EQ((*Segments)[3].Count, 2u);
  EXPECT_TRUE((*Segments)[4].IsPackReference);
  EXPECT_EQ((*Segments)[4].FirstIndex, 0x1006u);
}

TEST(CVPacksTest, SymbolMapRoundTrip) {
  RangeMap Map;
  Map.Entries = {{0, 2, 7, 0}, {1, 0, 3, -2}};

  auto Encoded = encodeSymbolMap(Map);
  ASSERT_TRUE(static_cast<bool>(Encoded));
  std::vector<CVSymbol> Records;
  for (const std::vector<uint8_t> &Record : *Encoded)
    Records.emplace_back(Record);
  auto Decoded = decodeSymbolMap(Records);
  ASSERT_TRUE(static_cast<bool>(Decoded));
  ASSERT_EQ(Decoded->Entries.size(), 2u);
  EXPECT_EQ(Decoded->Entries[0].Slot, 2u);
  EXPECT_EQ(Decoded->Entries[0].Count, 7u);
  EXPECT_EQ(Decoded->Entries[1].FirstDelta, -2);
}

TEST(CVPacksTest, MapIsSplitAtEntryBoundaries) {
  RangeMap Map;
  Map.Entries.resize(MaxMapBodySize / 4 + 2, {0, 0, 1, 0});
  auto Encoded = encodeTypeMap(Map);
  ASSERT_TRUE(static_cast<bool>(Encoded));
  ASSERT_EQ(Encoded->size(), 2u);

  std::vector<CVType> Records;
  for (const std::vector<uint8_t> &Record : *Encoded)
    Records.emplace_back(Record);
  auto Decoded = decodeTypeMap(Records);
  ASSERT_TRUE(static_cast<bool>(Decoded));
  EXPECT_EQ(Decoded->Entries.size(), Map.Entries.size());
}

TEST(CVPacksTest, OriginRoundTrip) {
  Origin Data;
  Data.OriginalFileSize = 1234;
  Data.OriginalFileHash[0] = 0x7a;
  Data.DebugHHeader = {1, 2, 3, 4};

  OriginEdit Edit;
  // Bigobj files can have more than 65535 sections.
  Edit.SectionIndex = 0x12345;
  Edit.Kind = OriginEditKind::SymbolStream;
  Edit.OriginalOffset = 8;
  Edit.OriginalLength = 12;
  Edit.PackedLength = 5;
  Edit.RestorationFragments.push_back({2, {0xaa, 0xbb}});
  Data.Edits.push_back(Edit);

  auto Encoded = encodeOrigin(Data);
  ASSERT_TRUE(static_cast<bool>(Encoded));
  auto Decoded = decodeOrigin(CVType(*Encoded));
  ASSERT_TRUE(static_cast<bool>(Decoded));
  EXPECT_EQ(Decoded->OriginalFileSize, Data.OriginalFileSize);
  EXPECT_EQ(Decoded->OriginalFileHash, Data.OriginalFileHash);
  EXPECT_EQ(Decoded->DebugHHeader, Data.DebugHHeader);
  ASSERT_EQ(Decoded->Edits.size(), 1u);
  EXPECT_EQ(Decoded->Edits[0].SectionIndex, 0x12345u);
  EXPECT_EQ(Decoded->Edits[0].Kind, OriginEditKind::SymbolStream);
  ASSERT_EQ(Decoded->Edits[0].RestorationFragments.size(), 1u);
  EXPECT_EQ(Decoded->Edits[0].RestorationFragments[0].Offset, 2u);
  EXPECT_EQ(Decoded->Edits[0].RestorationFragments[0].Bytes,
            (std::vector<uint8_t>{0xaa, 0xbb}));
}

TEST(CVPacksTest, MapExpansionRejectsInvalidRanges) {
  // A range past the end of the pack.
  RangeMap Map;
  Map.Entries.push_back({0, 0, 2, 0});
  auto Segments = expandRangeMap(Map, 0x1001, 0, 0x1000);
  EXPECT_FALSE(static_cast<bool>(Segments));
  consumeError(Segments.takeError());
  // A slot other than 0.
  Map.Entries[0] = {0, 1, 1, 0};
  Segments = expandRangeMap(Map, 0x1001, 0, 0x1000);
  EXPECT_FALSE(static_cast<bool>(Segments));
  consumeError(Segments.takeError());
}

TEST(CVPacksTest, TypeDumpVisitorPrintsPackRecords) {
  PackId Id{};
  Id[0] = 0xab;
  auto Encoded = encodePackList(Id);
  ASSERT_TRUE(static_cast<bool>(Encoded));

  ArrayRef<ArrayRef<uint8_t>> NoTypes;
  TypeTableCollection Types(NoTypes);
  std::string Text;
  raw_string_ostream OS(Text);
  ScopedPrinter Printer(OS);
  TypeDumpVisitor Visitor(Types, &Printer, false);
  CVType Record(*Encoded);
  ASSERT_FALSE(static_cast<bool>(
      Visitor.visitTypeBegin(Record, TypeIndex::fromArrayIndex(0))));
  ASSERT_FALSE(static_cast<bool>(Visitor.visitUnknownType(Record)));
  ASSERT_FALSE(static_cast<bool>(Visitor.visitTypeEnd(Record)));
  OS.flush();

  EXPECT_NE(Text.find("LF_CVPACK_LIST"), std::string::npos);
  EXPECT_NE(Text.find("PackId"), std::string::npos);
}

TEST(CVPacksTest, SymbolDumperPrintsSymbolPackMap) {
  RangeMap Map;
  Map.Entries.push_back({0, 0, 1, 0});
  auto Encoded = encodeSymbolMap(Map);
  ASSERT_TRUE(static_cast<bool>(Encoded));

  ArrayRef<ArrayRef<uint8_t>> NoTypes;
  TypeTableCollection Types(NoTypes);
  std::string Text;
  raw_string_ostream OS(Text);
  ScopedPrinter Printer(OS);
  CVSymbolDumper Dumper(Printer, Types, CodeViewContainer::ObjectFile,
                        std::unique_ptr<SymbolDumpDelegate>(), CPUType::X64,
                        false);
  CVSymbol Record(Encoded->front());
  ASSERT_FALSE(static_cast<bool>(Dumper.dump(Record)));
  OS.flush();

  EXPECT_NE(Text.find("S_CVPACK_MAP"), std::string::npos);
  EXPECT_NE(Text.find("RangeCount: 1"), std::string::npos);
  EXPECT_NE(Text.find("FirstDelta: 0"), std::string::npos);
}

static void put16(std::vector<uint8_t> &B, size_t O, uint16_t V) {
  B[O] = V & 0xff;
  B[O + 1] = V >> 8;
}
static void put32(std::vector<uint8_t> &B, size_t O, uint32_t V) {
  for (unsigned I = 0; I != 4; ++I)
    B[O + I] = (V >> (8 * I)) & 0xff;
}

TEST(CVPacksTest, PackedTypeStreamsAreRecognized) {
  std::vector<uint8_t> Stream(12, 0);
  put32(Stream, 0, COFF::DEBUG_CVPACK_SECTION_MAGIC);
  put16(Stream, 4, 6);
  put16(Stream, 6, TypeLeafKind::LF_CVPACK_LIST);
  EXPECT_TRUE(isPackedTypeStream(Stream));
  // The C13 signature is not that of a packed object.
  put32(Stream, 0, COFF::DEBUG_SECTION_MAGIC);
  EXPECT_FALSE(isPackedTypeStream(Stream));
  // An ordinary type stream.
  put16(Stream, 6, TypeLeafKind::LF_POINTER);
  EXPECT_FALSE(isPackedTypeStream(Stream));
  put32(Stream, 0, 0x12345678);
  put16(Stream, 6, TypeLeafKind::LF_CVPACK_LIST);
  EXPECT_FALSE(isPackedTypeStream(Stream));
}

TEST(CVPacksTest, FunctionRefRoundTrip) {
  FunctionRef Ref{2, 7, 212};
  std::optional<FunctionRef> Decoded =
      decodeFunctionRef(encodeFunctionRef(Ref));
  ASSERT_TRUE(Decoded.has_value());
  EXPECT_EQ(Decoded->Slot, 2u);
  EXPECT_EQ(Decoded->Ordinal, 7u);
  EXPECT_EQ(Decoded->OriginalSize, 212u);
}
