//===- llvm/unittest/DebugInfo/CVPack/CVPackTest.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/COFF.h"
#include "llvm/DebugInfo/CVPack/COFFEdit.h"
#include "llvm/DebugInfo/CVPack/FunctionDebug.h"
#include "llvm/DebugInfo/CVPack/PackFile.h"
#include "llvm/DebugInfo/CVPack/PackIndex.h"
#include "llvm/DebugInfo/CodeView/CVPacks.h"
#include "llvm/DebugInfo/CodeView/TypeHashing.h"
#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::codeview::cvpack;

static void put16(std::vector<uint8_t> &B, size_t O, uint16_t V) {
  B[O] = V & 0xff;
  B[O + 1] = V >> 8;
}
static void put32(std::vector<uint8_t> &B, size_t O, uint32_t V) {
  for (unsigned I = 0; I != 4; ++I)
    B[O + I] = (V >> (8 * I)) & 0xff;
}
static uint32_t get32(ArrayRef<uint8_t> B, size_t O) {
  return B[O] | (B[O + 1] << 8) | (B[O + 2] << 16) | (uint32_t(B[O + 3]) << 24);
}

static void add32(std::vector<uint8_t> &B, uint32_t V) {
  B.resize(B.size() + 4);
  put32(B, B.size() - 4, V);
}

// A symbols subsection with one S_LOCAL, then a lines subsection with one
// block: the type index is at offset 12 and the file at offset 40.
static std::vector<uint8_t> functionDebug() {
  std::vector<uint8_t> D;
  for (uint32_t V : {0xf1u, 12u}) // symbols subsection
    add32(D, V);
  for (uint8_t B :
       {0x0a, 0x00, 0x3e, 0x11, 0x00, 0x10, 0x00, 0x00, 0x01, 0x00, 0x78, 0x00})
    D.push_back(B);
  for (uint32_t V : {0xf2u, 32u, 0u, 0u, 1u, 0x18u, 1u, 20u, 0u, 0x80000001u})
    add32(D, V);
  return D;
}

TEST(CVPackTest, PackCOFFWriterAndReaderRoundTrip) {
  PackContents Contents;
  Contents.Machine = COFF::IMAGE_FILE_MACHINE_ARM64;
  const std::vector<uint8_t> TypeRecord = {0x0a, 0x00, 0x01, 0x10, 0x74, 0x00,
                                           0x00, 0x00, 0x00, 0x00, 0xf2, 0xf1};
  const std::vector<uint8_t> SymbolRecord = {
      0x0a, 0x00, 0x08, 0x11, 0x00, 0x10, 0x00, 0x00, 0x61, 0x00, 0xf2, 0xf1};
  Contents.Types.push_back(TypeRecord);
  Contents.Symbols.push_back(SymbolRecord);

  auto Written = writePackFile(Contents);
  ASSERT_TRUE(static_cast<bool>(Written));
  auto Read = readPackFile(Written->FileData, Written->ContentId);
  ASSERT_TRUE(static_cast<bool>(Read));
  EXPECT_EQ(Read->ContentId, Written->ContentId);
  EXPECT_EQ(Read->Machine, COFF::IMAGE_FILE_MACHINE_ARM64);
  ASSERT_EQ(Read->Types.size(), 1u);
  ASSERT_EQ(Read->Symbols.size(), 1u);

  PackId WrongId = Written->ContentId;
  WrongId[0] ^= 0xff;
  auto WrongIdentity = readPackFile(Written->FileData, WrongId);
  EXPECT_FALSE(static_cast<bool>(WrongIdentity));
  consumeError(WrongIdentity.takeError());

  // A pack refers to no other pack.
  auto List = encodePackList(PackId{});
  ASSERT_TRUE(static_cast<bool>(List));
  Contents.Types.insert(Contents.Types.begin(), *List);
  auto Dependent = writePackFile(Contents);
  EXPECT_FALSE(static_cast<bool>(Dependent));
  consumeError(Dependent.takeError());
}

TEST(CVPackTest, EmptyPackRoundTrip) {
  auto Written = writePackFile(PackContents{});
  ASSERT_TRUE(static_cast<bool>(Written));
  auto Read = readPackFile(Written->FileData, Written->ContentId);
  ASSERT_TRUE(static_cast<bool>(Read));
  EXPECT_TRUE(Read->Types.empty());
  EXPECT_TRUE(Read->Symbols.empty());
}

TEST(CVPackTest, RecordKeysNormalizeReferencedTypeIndices) {
  const std::vector<uint8_t> First = {0x0a, 0x00, 0x02, 0x10, 0x00, 0x10,
                                      0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  const std::vector<uint8_t> Second = {0x0a, 0x00, 0x02, 0x10, 0x01, 0x10,
                                       0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  const std::array<uint8_t, 8> ReferentHash = {0x34, 0x12, 0x56, 0x78,
                                               0x9a, 0xbc, 0xde, 0xf0};
  std::vector<GloballyHashedType> TypeHashes = {
      GloballyHashedType(ReferentHash), GloballyHashedType(ReferentHash)};

  auto FirstKey = getTypeRecordKey(First, TypeHashes, {});
  ASSERT_TRUE(static_cast<bool>(FirstKey));
  auto SecondKey = getTypeRecordKey(Second, TypeHashes, {});
  ASSERT_TRUE(static_cast<bool>(SecondKey));
  EXPECT_EQ(*FirstKey, *SecondKey);
}

TEST(CVPackTest, RecordKeyRejectsUnresolvedReferences) {
  const std::vector<uint8_t> Record = {0x0a, 0x00, 0x02, 0x10, 0x01, 0x10,
                                       0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  auto Key = getTypeRecordKey(Record, {}, {});
  EXPECT_FALSE(static_cast<bool>(Key));
  consumeError(Key.takeError());
}

TEST(CVPackTest, BigObjSectionIsAppendedAndRemovedExactly) {
  // A bigobj header and one section header.
  std::vector<uint8_t> Object(56 + 40, 0);
  put16(Object, 2, 0xffff);
  put16(Object, 4, 2);
  put16(Object, 6, 0x8664);
  // The magic contains bytes above 0x7f, which a signed comparison misses.
  std::copy(std::begin(COFF::BigObjMagic), std::end(COFF::BigObjMagic),
            Object.begin() + 12);
  put32(Object, 44, 1);

  const uint8_t Data[] = {1, 2, 3, 4};
  auto Added =
      appendCOFFSection(Object, ".drectve", COFF::IMAGE_SCN_LNK_INFO, Data);
  ASSERT_TRUE(static_cast<bool>(Added)) << toString(Added.takeError());
  ArrayRef<uint8_t> New = *Added;
  EXPECT_EQ(New.size(), Object.size() + 40 + sizeof(Data));
  EXPECT_EQ(get32(New, 44), 2u);                 // section count
  EXPECT_EQ(get32(New, 56 + 40 + 16), 4u);       // new section size
  EXPECT_EQ(get32(New, 56 + 40 + 20), 56 + 80u); // its data follows the table

  auto Removed = removeLastCOFFSection(*Added);
  ASSERT_TRUE(static_cast<bool>(Removed)) << toString(Removed.takeError());
  EXPECT_EQ(ArrayRef<uint8_t>(*Removed), ArrayRef<uint8_t>(Object));
}

TEST(CVPackTest, ReplaceSectionsShiftsOffsetsAndRelocations) {
  // Header, two section headers, section 1 (16 bytes) with two relocations,
  // section 2 (8 bytes), then a symbol table pointer target.
  const size_t Sec1 = 20 + 80, Relocs = Sec1 + 16, Sec2 = Relocs + 20,
               Symbols = Sec2 + 8;
  std::vector<uint8_t> Object(Symbols + 18, 0);
  put16(Object, 0, 0x8664);
  put16(Object, 2, 2);
  put32(Object, 8, Symbols);
  put32(Object, 12, 1);
  put32(Object, 20 + 16, 16);
  put32(Object, 20 + 20, Sec1);
  put32(Object, 20 + 24, Relocs);
  put16(Object, 20 + 32, 2);
  put32(Object, 60 + 16, 8);
  put32(Object, 60 + 20, Sec2);
  for (size_t I = 0; I != 16; ++I)
    Object[Sec1 + I] = I;
  put32(Object, Relocs, 4);       // in the first kept span
  put32(Object, Relocs + 10, 12); // in the second kept span

  // Drop bytes 4..11 of section 1 and keep 0..3 and 12..15.
  SectionReplacement R;
  R.SectionIndex = 1;
  R.Data = {0, 1, 2, 3, 12, 13, 14, 15};
  R.Spans = {{0, 0, 4}, {12, 4, 4}};
  // The relocation at offset 4 now falls in a dropped range.
  auto Rejected = replaceCOFFSections(Object, {R});
  EXPECT_FALSE(static_cast<bool>(Rejected));
  consumeError(Rejected.takeError());

  put32(Object, Relocs, 2);
  auto Result = replaceCOFFSections(Object, {R});
  ASSERT_TRUE(static_cast<bool>(Result));
  ArrayRef<uint8_t> New = *Result;
  EXPECT_EQ(New.size(), Object.size() - 8);
  EXPECT_EQ(get32(New, 20 + 16), 8u);         // section 1 size
  EXPECT_EQ(get32(New, 20 + 24), Relocs - 8); // its relocations moved
  EXPECT_EQ(get32(New, 60 + 20), Sec2 - 8);   // section 2 moved
  EXPECT_EQ(get32(New, 8), Symbols - 8);      // symbol table moved
  EXPECT_EQ(get32(New, Relocs - 8), 2u);      // unchanged offset
  EXPECT_EQ(get32(New, Relocs - 8 + 10), 4u); // 12 -> 4
}

TEST(CVPackTest, AppendedSectionIsRemovedExactly) {
  // Header, one section header, its 8 bytes of data with one relocation, then
  // a symbol table and string table.
  const size_t Sec1 = 20 + 40, Relocs = Sec1 + 8, Symbols = Relocs + 10;
  std::vector<uint8_t> Object(Symbols + 18 + 4, 0);
  put16(Object, 0, 0x8664);
  put16(Object, 2, 1);
  put32(Object, 8, Symbols);
  put32(Object, 12, 1);
  put32(Object, 20 + 16, 8);
  put32(Object, 20 + 20, Sec1);
  put32(Object, 20 + 24, Relocs);
  put16(Object, 20 + 32, 1);
  put32(Object, Symbols + 18, 4); // empty string table

  const char Text[] = " /INCLUDE:x";
  ArrayRef<uint8_t> Data(reinterpret_cast<const uint8_t *>(Text),
                         sizeof(Text) - 1);
  auto Added = appendCOFFSection(
      Object, ".drectve", COFF::IMAGE_SCN_LNK_INFO | COFF::IMAGE_SCN_LNK_REMOVE,
      Data);
  ASSERT_TRUE(static_cast<bool>(Added));
  ArrayRef<uint8_t> New = *Added;
  EXPECT_EQ(New.size(), Object.size() + 40 + Data.size());
  EXPECT_EQ(get32(New, 2) & 0xffff, 2u);                // section count
  EXPECT_EQ(get32(New, 20 + 20), Sec1 + 40);            // data moved
  EXPECT_EQ(get32(New, 20 + 24), Relocs + 40);          // relocations moved
  EXPECT_EQ(get32(New, 60 + 16), Data.size());          // new section size
  EXPECT_EQ(get32(New, 60 + 20), Symbols + 40);         // before the symbols
  EXPECT_EQ(get32(New, 8), Symbols + 40 + Data.size()); // symbols moved
  EXPECT_EQ(New.slice(Symbols + 40, Data.size()), Data);
  EXPECT_EQ(std::string(reinterpret_cast<const char *>(New.data() + 60), 8),
            std::string(".drectve"));

  auto Removed = removeLastCOFFSection(*Added);
  ASSERT_TRUE(static_cast<bool>(Removed));
  EXPECT_EQ(ArrayRef<uint8_t>(*Removed), ArrayRef<uint8_t>(Object));

  // A section with relocations is not removed.
  auto Refused = removeLastCOFFSection(Object);
  EXPECT_FALSE(static_cast<bool>(Refused));
  consumeError(Refused.takeError());
}

TEST(CVPackTest, FunctionDebugFieldsAreRewritten) {
  std::vector<uint8_t> D = functionDebug();
  std::vector<uint32_t> TypeFields, FileFields;
  std::vector<uint8_t> Out;
  ASSERT_FALSE(static_cast<bool>(transformFunctionDebug(
      D, Out,
      [&](uint32_t I, uint32_t Offset) -> Expected<uint32_t> {
        TypeFields.push_back(Offset);
        return I + 5;
      },
      [&](uint32_t F, uint32_t Offset) -> Expected<uint32_t> {
        FileFields.push_back(Offset);
        return F / 0x18;
      },
      FileOperandEncoding::Minimal)));
  EXPECT_EQ(TypeFields, std::vector<uint32_t>{12});
  EXPECT_EQ(FileFields, std::vector<uint32_t>{40});
  ASSERT_EQ(Out.size(), D.size());
  EXPECT_EQ(get32(Out, 12), 0x1005u);
  EXPECT_EQ(get32(Out, 40), 1u);
}

TEST(CVPackTest, FunctionDebugRejectsUnsupportedContent) {
  auto identity = [](uint32_t V, uint32_t) -> Expected<uint32_t> { return V; };
  auto rejected = [&](std::vector<uint8_t> D) {
    std::vector<uint8_t> Out;
    Error E = transformFunctionDebug(D, Out, identity, identity,
                                     FileOperandEncoding::Minimal);
    bool Failed = static_cast<bool>(E);
    consumeError(std::move(E));
    return Failed;
  };
  std::vector<uint8_t> D = functionDebug();
  EXPECT_FALSE(rejected(D));
  // A file checksums subsection.
  std::vector<uint8_t> Checksums = D;
  put32(Checksums, 20, 0xf4);
  EXPECT_TRUE(rejected(Checksums));
  // S_FILESTATIC references the string table.
  std::vector<uint8_t> FileStatic = D;
  FileStatic[10] = 0x53;
  FileStatic[11] = 0x11;
  EXPECT_TRUE(rejected(FileStatic));
  // An inline site whose annotations change the file.
  std::vector<uint8_t> Site;
  for (uint32_t V : {0xf1u, 20u})
    add32(Site, V);
  for (uint8_t B : {0x12, 0x00, 0x4d, 0x11, 0,    0,    0,    0,    0,    0,
                    0,    0,    0x00, 0x10, 0x00, 0x00, 0x05, 0x18, 0x00, 0x00})
    Site.push_back(B);
  EXPECT_FALSE(rejected(Site));

  // Its file operand is rewritten in 4 bytes for a pack, and restored in the
  // fewest bytes.
  std::vector<uint8_t> Packed, Restored;
  ASSERT_FALSE(static_cast<bool>(transformFunctionDebug(
      ArrayRef<uint8_t>(Site).drop_front(0), Packed, identity,
      [](uint32_t F, uint32_t) -> Expected<uint32_t> {
        return F == 0x18 ? 2 : 99;
      },
      FileOperandEncoding::Fixed)));
  ASSERT_EQ(Packed.size(), Site.size() + 4);
  EXPECT_EQ(get32(Packed, 4), 24u); // subsection size
  EXPECT_EQ(Packed[8], 22u);        // record length
  EXPECT_EQ(Packed[24], 0x05);
  EXPECT_EQ(get32(Packed, 25), 0x020000c0u);
  ASSERT_FALSE(static_cast<bool>(transformFunctionDebug(
      Packed, Restored, identity,
      [](uint32_t F, uint32_t) -> Expected<uint32_t> {
        return F == 2 ? 0x18 : 99;
      },
      FileOperandEncoding::Minimal)));
  EXPECT_EQ(Restored, Site);
}

TEST(CVPackTest, PackFunctionsRoundTrip) {
  PackContents Contents;
  const std::vector<uint8_t> Checksum = {1, 2, 3, 4};
  Contents.Files.push_back({"c:/src/a.h", 1, Checksum});
  Contents.Files.push_back({"b.h", 0, {}});
  std::vector<uint8_t> F1 = functionDebug();
  std::vector<uint8_t> F2 = {0xf1, 0, 0, 0, 0, 0, 0, 0};
  Contents.Functions = {F1, F2};
  auto Written = writePackFile(Contents);
  ASSERT_TRUE(static_cast<bool>(Written));
  auto Read = readPackFile(Written->FileData, Written->ContentId);
  ASSERT_TRUE(static_cast<bool>(Read));
  ASSERT_EQ(Read->Files.size(), 2u);
  EXPECT_EQ(Read->Files[0].Name, "c:/src/a.h");
  EXPECT_EQ(Read->Files[0].ChecksumKind, 1u);
  EXPECT_EQ(ArrayRef<uint8_t>(Read->Files[0].Checksum), ArrayRef(Checksum));
  EXPECT_EQ(Read->Files[1].Name, "b.h");
  ASSERT_EQ(Read->Functions.size(), 2u);
  EXPECT_EQ(Read->Functions[0], ArrayRef(F1));
  EXPECT_EQ(Read->Functions[1], ArrayRef(F2));

  FunctionRef Ref{2, 7, 212};
  std::optional<FunctionRef> Decoded =
      decodeFunctionRef(encodeFunctionRef(Ref));
  ASSERT_TRUE(Decoded.has_value());
  EXPECT_EQ(Decoded->Slot, 2u);
  EXPECT_EQ(Decoded->Ordinal, 7u);
  EXPECT_EQ(Decoded->OriginalSize, 212u);
}

TEST(CVPackTest, ObjectFileTableMapsFilesToChecksumOffsets) {
  // Strings "\0a.h\0b.h\0"; checksums: a.h (MD5-like, 4 bytes) then b.h twice.
  std::vector<uint8_t> S;
  add32(S, 4);
  for (uint32_t V : {0xf3u, 9u})
    add32(S, V);
  for (char C : StringRef("\0a.h\0b.h\0", 9))
    S.push_back(C);
  while (S.size() % 4)
    S.push_back(0);
  for (uint32_t V : {0xf4u, 28u})
    add32(S, V);
  for (uint8_t B : {1, 0, 0, 0, 4, 1, 9, 9, 9, 9, 0, 0, // a.h at 0
                    5, 0, 0, 0, 0, 0, 0, 0,             // b.h at 12
                    5, 0, 0, 0, 0, 0, 0, 0})            // b.h at 20
    S.push_back(B);
  auto Table = ObjectFileTable::create({ArrayRef<uint8_t>(S)});
  ASSERT_TRUE(static_cast<bool>(Table));
  ASSERT_EQ(Table->entries().size(), 3u);
  std::optional<PackFileEntry> B = Table->entryAt(20);
  ASSERT_TRUE(B.has_value());
  EXPECT_EQ(B->Name, "b.h");
  // The first entry with the same identity wins.
  EXPECT_EQ(Table->lookup(*B), std::optional<uint32_t>(12));
  std::optional<PackFileEntry> A = Table->entryAt(0);
  ASSERT_TRUE(A.has_value());
  EXPECT_EQ(A->Checksum.size(), 4u);
  EXPECT_EQ(Table->lookup(*A), std::optional<uint32_t>(0));
  EXPECT_FALSE(Table->entryAt(4).has_value());

  auto Twice =
      ObjectFileTable::create({ArrayRef<uint8_t>(S), ArrayRef<uint8_t>(S)});
  EXPECT_FALSE(static_cast<bool>(Twice));
  consumeError(Twice.takeError());
}

TEST(CVPackTest, ObjectFileTableStopsAtAnUnpaddedLastSubsection) {
  // A strings subsection of 9 bytes ends the section without padding. The
  // section is a slice of a larger buffer, whose next bytes look like a second
  // strings subsection: they must not be read.
  std::vector<uint8_t> Buffer;
  add32(Buffer, 4);
  for (uint32_t V : {0xf3u, 9u})
    add32(Buffer, V);
  for (char C : StringRef("\0a.h\0b.h\0", 9))
    Buffer.push_back(C);
  const size_t SectionSize = Buffer.size();
  while (Buffer.size() % 4)
    Buffer.push_back(0);
  for (uint32_t V : {0xf3u, 0u})
    add32(Buffer, V);
  auto Table = ObjectFileTable::create(
      {ArrayRef<uint8_t>(Buffer).take_front(SectionSize)});
  ASSERT_TRUE(static_cast<bool>(Table)) << toString(Table.takeError());
  EXPECT_TRUE(Table->entries().empty());
}

TEST(CVPackTest, PackGlobalHashesRoundTrip) {
  // LF_POINTER to int, then LF_MODIFIER of that pointer.
  const std::vector<uint8_t> Pointer = {0x0a, 0x00, 0x02, 0x10, 0x74, 0x00,
                                        0x00, 0x00, 0x0c, 0x00, 0x01, 0x00};
  const std::vector<uint8_t> Modifier = {0x0a, 0x00, 0x01, 0x10, 0x00, 0x10,
                                         0x00, 0x00, 0x01, 0x00, 0xf2, 0xf1};
  PackContents Contents;
  Contents.Types = {Pointer, Modifier};
  Contents.WriteGlobalHashes = true;
  auto Written = writePackFile(Contents);
  ASSERT_TRUE(static_cast<bool>(Written));
  auto Read = readPackFile(Written->FileData, Written->ContentId);
  ASSERT_TRUE(static_cast<bool>(Read));
  ASSERT_EQ(Read->GlobalHashes.size(), 16u);
  std::vector<CVType> Types = {CVType(Pointer), CVType(Modifier)};
  std::vector<GloballyHashedType> Hashes = GloballyHashedType::hashTypes(Types);
  EXPECT_EQ(Read->GlobalHashes.take_front(8),
            ArrayRef<uint8_t>(Hashes[0].Hash));
  EXPECT_EQ(Read->GlobalHashes.drop_front(8),
            ArrayRef<uint8_t>(Hashes[1].Hash));

  // Without the request, no hashes are written.
  Contents.WriteGlobalHashes = false;
  auto Plain = writePackFile(Contents);
  ASSERT_TRUE(static_cast<bool>(Plain));
  auto ReadPlain = readPackFile(Plain->FileData, Plain->ContentId);
  ASSERT_TRUE(static_cast<bool>(ReadPlain));
  EXPECT_TRUE(ReadPlain->GlobalHashes.empty());
}

TEST(CVPackTest, PackIndexRoundTripAndLookups) {
  auto key = [](uint8_t First, uint8_t Last) {
    RecordKey K{};
    K[0] = First;
    K[15] = Last;
    return K;
  };
  PackIndex Index;
  Index.PackContentId[0] = 0xab;
  Index.PackFileSize = 1234;
  Index.TypeCount = 10;
  Index.SymbolCount = 2;
  Index.FunctionCount = 3;
  // Given out of order: the writer sorts.
  Index.Types = {{key(0x90, 1), 7}, {key(0x10, 2), 3}, {key(0x50, 3), 9}};
  Index.Symbols = {{key(0x20, 4), 1}};
  Index.Functions = {{key(0x30, 5), 2}, {key(0x31, 6), 99}};
  auto Data = writePackIndex(Index);
  ASSERT_TRUE(static_cast<bool>(Data));
  EXPECT_EQ(Data->size(), 72u + 20u * 6);

  auto View = PackIndexView::create(*Data);
  ASSERT_TRUE(static_cast<bool>(View));
  EXPECT_EQ(View->packContentId(), Index.PackContentId);
  EXPECT_EQ(View->packFileSize(), 1234u);
  EXPECT_EQ(View->typeCount(), 10u);
  auto find = [&](PackIndexView::Table T, ArrayRef<RecordKey> Keys) {
    std::vector<std::pair<size_t, uint32_t>> Found;
    View->findSorted(T, Keys,
                     [&](size_t K, uint32_t I) { Found.push_back({K, I}); });
    return Found;
  };
  using Found = std::vector<std::pair<size_t, uint32_t>>;
  std::vector<RecordKey> Sorted = {key(0x00, 0), key(0x10, 2), key(0x40, 0),
                                   key(0x50, 3), key(0x50, 4), key(0x90, 1),
                                   key(0xff, 0)};
  EXPECT_EQ(find(PackIndexView::Table::Types, Sorted),
            (Found{{1, 3}, {3, 9}, {5, 7}}));
  EXPECT_EQ(find(PackIndexView::Table::Symbols, {key(0x20, 4)}),
            (Found{{0, 1}}));
  // An index beyond the pack's function count is not returned.
  EXPECT_EQ(find(PackIndexView::Table::Functions, {key(0x30, 5), key(0x31, 6)}),
            (Found{{0, 2}}));

  // Duplicate keys are rejected; truncated or foreign data is not an index.
  Index.Types.push_back({key(0x10, 2), 4});
  auto Duplicate = writePackIndex(Index);
  EXPECT_FALSE(static_cast<bool>(Duplicate));
  consumeError(Duplicate.takeError());
  auto Truncated = PackIndexView::create(ArrayRef<uint8_t>(*Data).drop_back(1));
  EXPECT_FALSE(static_cast<bool>(Truncated));
  consumeError(Truncated.takeError());
  std::vector<uint8_t> Foreign(*Data);
  Foreign[0] = 'X';
  auto NotIndex = PackIndexView::create(Foreign);
  EXPECT_FALSE(static_cast<bool>(NotIndex));
  consumeError(NotIndex.takeError());
}
