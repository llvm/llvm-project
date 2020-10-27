//===- llvm/unittest/DebugInfo/CodeView/RandomAccessVisitorTest.cpp -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/AppendingTypeTableBuilder.h"
#include "llvm/DebugInfo/CodeView/CVTypeVisitor.h"
#include "llvm/DebugInfo/CodeView/LazyRandomTypeCollection.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/CodeView/TypeRecordMapping.h"
#include "llvm/DebugInfo/CodeView/TypeVisitorCallbacks.h"
#include "llvm/DebugInfo/PDB/Native/RawTypes.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/BinaryByteStream.h"
#include "llvm/Support/BinaryItemStream.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/Support/Error.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::pdb;

namespace llvm {
namespace codeview {
inline bool operator==(const ArrayRecord &R1, const ArrayRecord &R2) {
  if (R1.ElementType != R2.ElementType)
    return false;
  if (R1.IndexType != R2.IndexType)
    return false;
  if (R1.Name != R2.Name)
    return false;
  if (R1.Size != R2.Size)
    return false;
  return true;
}
inline bool operator!=(const ArrayRecord &R1, const ArrayRecord &R2) {
  return !(R1 == R2);
}

inline bool operator==(const CVType &R1, const CVType &R2) {
  if (R1.RecordData != R2.RecordData)
    return false;
  return true;
}
inline bool operator!=(const CVType &R1, const CVType &R2) {
  return !(R1 == R2);
}
}
}

namespace llvm {
template <> struct BinaryItemTraits<CVType> {
  static size_t length(const CVType &Item) { return Item.length(); }
  static ArrayRef<uint8_t> bytes(const CVType &Item) { return Item.data(); }
};
}

namespace {

class MockCallbacks : public TypeVisitorCallbacks {
public:
  Error visitTypeBegin(CVType &CVR, TypeIndex Index) override {
    Indices.push_back(Index);
    return Error::success();
  }
  Error visitKnownRecord(CVType &CVR, ArrayRecord &AR) override {
    VisitedRecords.push_back(AR);
    RawRecords.push_back(CVR);
    return Error::success();
  }

  uint32_t count() const {
    assert(Indices.size() == RawRecords.size());
    assert(Indices.size() == VisitedRecords.size());
    return Indices.size();
  }
  std::vector<TypeIndex> Indices;
  std::vector<CVType> RawRecords;
  std::vector<ArrayRecord> VisitedRecords;
};

class RandomAccessVisitorTest : public testing::Test {
public:
  RandomAccessVisitorTest() {}

  static void SetUpTestCase() {
    GlobalState = std::make_unique<GlobalTestState>();

    AppendingTypeTableBuilder Builder(GlobalState->Allocator);

    uint32_t Offset = 0;
    for (int I = 0; I < 11; ++I) {
      ArrayRecord AR(TypeRecordKind::Array);
      AR.ElementType = TypeIndex::Int32();
      AR.IndexType = TypeIndex::UInt32();
      AR.Size = I;
      std::string Name;
      raw_string_ostream Stream(Name);
      Stream << "Array [" << I << "]";
      AR.Name = GlobalState->Strings.save(Stream.str());
      GlobalState->Records.push_back(AR);
      GlobalState->Indices.push_back(Builder.writeLeafType(AR));

      CVType Type(Builder.records().back());
      GlobalState->TypeVector.push_back(Type);

      GlobalState->AllOffsets.push_back(
          {GlobalState->Indices.back(), ulittle32_t(Offset)});
      Offset += Type.length();
    }

    GlobalState->ItemStream.setItems(GlobalState->TypeVector);
    GlobalState->TypeArray = VarStreamArray<CVType>(GlobalState->ItemStream);
  }

  static void TearDownTestCase() { GlobalState.reset(); }

  void SetUp() override {
    TestState = std::make_unique<PerTestState>();
  }

  void TearDown() override { TestState.reset(); }

protected:
  bool ValidateDatabaseRecord(LazyRandomTypeCollection &Types, uint32_t Index) {
    TypeIndex TI = TypeIndex::fromArrayIndex(Index);
    if (!Types.contains(TI))
      return false;
    if (GlobalState->TypeVector[Index] != Types.getType(TI))
      return false;
    return true;
  }

  bool ValidateVisitedRecord(uint32_t VisitationOrder,
                             uint32_t GlobalArrayIndex) {
    TypeIndex TI = TypeIndex::fromArrayIndex(GlobalArrayIndex);
    if (TI != TestState->Callbacks.Indices[VisitationOrder])
      return false;

    if (GlobalState->TypeVector[TI.toArrayIndex()] !=
        TestState->Callbacks.RawRecords[VisitationOrder])
      return false;

    if (GlobalState->Records[TI.toArrayIndex()] !=
        TestState->Callbacks.VisitedRecords[VisitationOrder])
      return false;

    return true;
  }

  struct GlobalTestState {
    GlobalTestState() : Strings(Allocator), ItemStream(llvm::support::little) {}

    BumpPtrAllocator Allocator;
    StringSaver Strings;

    std::vector<ArrayRecord> Records;
    std::vector<TypeIndex> Indices;
    std::vector<TypeIndexOffset> AllOffsets;
    std::vector<CVType> TypeVector;
    BinaryItemStream<CVType> ItemStream;
    VarStreamArray<CVType> TypeArray;

    MutableBinaryByteStream Stream;
  };

  struct PerTestState {
    FixedStreamArray<TypeIndexOffset> Offsets;

    MockCallbacks Callbacks;
  };

  FixedStreamArray<TypeIndexOffset>
  createPartialOffsets(MutableBinaryByteStream &Storage,
                       std::initializer_list<uint32_t> Indices) {

    uint32_t Count = Indices.size();
    uint32_t Size = Count * sizeof(TypeIndexOffset);
    uint8_t *Buffer = GlobalState->Allocator.Allocate<uint8_t>(Size);
    MutableArrayRef<uint8_t> Bytes(Buffer, Size);
    Storage = MutableBinaryByteStream(Bytes, support::little);
    BinaryStreamWriter Writer(Storage);
    for (const auto I : Indices)
      consumeError(Writer.writeObject(GlobalState->AllOffsets[I]));

    BinaryStreamReader Reader(Storage);
    FixedStreamArray<TypeIndexOffset> Result;
    consumeError(Reader.readArray(Result, Count));
    return Result;
  }

  static std::unique_ptr<GlobalTestState> GlobalState;
  std::unique_ptr<PerTestState> TestState;
};

std::unique_ptr<RandomAccessVisitorTest::GlobalTestState>
    RandomAccessVisitorTest::GlobalState;
}

TEST_F(RandomAccessVisitorTest, MultipleVisits) {
  TestState->Offsets = createPartialOffsets(GlobalState->Stream, {0, 8});
  LazyRandomTypeCollection Types(GlobalState->TypeArray,
                                 GlobalState->TypeVector.size(),
                                 TestState->Offsets);

  std::vector<uint32_t> IndicesToVisit = {5, 5, 5};

  for (uint32_t I : IndicesToVisit) {
    TypeIndex TI = TypeIndex::fromArrayIndex(I);
    CVType T = Types.getType(TI);
    EXPECT_THAT_ERROR(codeview::visitTypeRecord(T, TI, TestState->Callbacks),
                      Succeeded());
  }

  // [0,8) should be present
  EXPECT_EQ(8u, Types.size());
  for (uint32_t I = 0; I < 8; ++I)
    EXPECT_TRUE(ValidateDatabaseRecord(Types, I));

  // 5, 5, 5
  EXPECT_EQ(3u, TestState->Callbacks.count());
  for (auto I : enumerate(IndicesToVisit))
    EXPECT_TRUE(ValidateVisitedRecord(I.index(), I.value()));
}

TEST_F(RandomAccessVisitorTest, DescendingWithinChunk) {
  // Visit multiple items from the same "chunk" in reverse order.  In this
  // example, it's 7 then 4 then 2.  At the end, all records from 0 to 7 should
  // be known by the database, but only 2, 4, and 7 should have been visited.
  TestState->Offsets = createPartialOffsets(GlobalState->Stream, {0, 8});

  std::vector<uint32_t> IndicesToVisit = {7, 4, 2};

  LazyRandomTypeCollection Types(GlobalState->TypeArray,
                                 GlobalState->TypeVector.size(),
                                 TestState->Offsets);
  for (uint32_t I : IndicesToVisit) {
    TypeIndex TI = TypeIndex::fromArrayIndex(I);
    CVType T = Types.getType(TI);
    EXPECT_THAT_ERROR(codeview::visitTypeRecord(T, TI, TestState->Callbacks),
                      Succeeded());
  }

  // [0, 7]
  EXPECT_EQ(8u, Types.size());
  for (uint32_t I = 0; I < 8; ++I)
    EXPECT_TRUE(ValidateDatabaseRecord(Types, I));

  // 2, 4, 7
  EXPECT_EQ(3u, TestState->Callbacks.count());
  for (auto I : enumerate(IndicesToVisit))
    EXPECT_TRUE(ValidateVisitedRecord(I.index(), I.value()));
}

TEST_F(RandomAccessVisitorTest, AscendingWithinChunk) {
  // * Visit multiple items from the same chunk in ascending order, ensuring
  //   that intermediate items are not visited.  In the below example, it's
  //   5 -> 6 -> 7 which come from the [4,8) chunk.
  TestState->Offsets = createPartialOffsets(GlobalState->Stream, {0, 8});

  std::vector<uint32_t> IndicesToVisit = {2, 4, 7};

  LazyRandomTypeCollection Types(GlobalState->TypeArray,
                                 GlobalState->TypeVector.size(),
                                 TestState->Offsets);
  for (uint32_t I : IndicesToVisit) {
    TypeIndex TI = TypeIndex::fromArrayIndex(I);
    CVType T = Types.getType(TI);
    EXPECT_THAT_ERROR(codeview::visitTypeRecord(T, TI, TestState->Callbacks),
                      Succeeded());
  }

  // [0, 7]
  EXPECT_EQ(8u, Types.size());
  for (uint32_t I = 0; I < 8; ++I)
    EXPECT_TRUE(ValidateDatabaseRecord(Types, I));

  // 2, 4, 7
  EXPECT_EQ(3u, TestState->Callbacks.count());
  for (auto &I : enumerate(IndicesToVisit))
    EXPECT_TRUE(ValidateVisitedRecord(I.index(), I.value()));
}

TEST_F(RandomAccessVisitorTest, StopPrematurelyInChunk) {
  // * Don't visit the last item in one chunk, ensuring that visitation stops
  //   at the record you specify, and the chunk is only partially visited.
  //   In the below example, this is tested by visiting 0 and 1 but not 2,
  //   all from the [0,3) chunk.
  TestState->Offsets = createPartialOffsets(GlobalState->Stream, {0, 8});

  std::vector<uint32_t> IndicesToVisit = {0, 1, 2};

  LazyRandomTypeCollection Types(GlobalState->TypeArray,
                                 GlobalState->TypeVector.size(),
                                 TestState->Offsets);

  for (uint32_t I : IndicesToVisit) {
    TypeIndex TI = TypeIndex::fromArrayIndex(I);
    CVType T = Types.getType(TI);
    EXPECT_THAT_ERROR(codeview::visitTypeRecord(T, TI, TestState->Callbacks),
                      Succeeded());
  }

  // [0, 8) should be visited.
  EXPECT_EQ(8u, Types.size());
  for (uint32_t I = 0; I < 8; ++I)
    EXPECT_TRUE(ValidateDatabaseRecord(Types, I));

  // [0, 2]
  EXPECT_EQ(3u, TestState->Callbacks.count());
  for (auto I : enumerate(IndicesToVisit))
    EXPECT_TRUE(ValidateVisitedRecord(I.index(), I.value()));
}

TEST_F(RandomAccessVisitorTest, InnerChunk) {
  // Test that when a request comes from a chunk in the middle of the partial
  // offsets array, that items from surrounding chunks are not visited or
  // added to the database.
  TestState->Offsets = createPartialOffsets(GlobalState->Stream, {0, 4, 9});

  std::vector<uint32_t> IndicesToVisit = {5, 7};

  LazyRandomTypeCollection Types(GlobalState->TypeArray,
                                 GlobalState->TypeVector.size(),
                                 TestState->Offsets);

  for (uint32_t I : IndicesToVisit) {
    TypeIndex TI = TypeIndex::fromArrayIndex(I);
    CVType T = Types.getType(TI);
    EXPECT_THAT_ERROR(codeview::visitTypeRecord(T, TI, TestState->Callbacks),
                      Succeeded());
  }

  // [4, 9)
  EXPECT_EQ(5u, Types.size());
  for (uint32_t I = 4; I < 9; ++I)
    EXPECT_TRUE(ValidateDatabaseRecord(Types, I));

  // 5, 7
  EXPECT_EQ(2u, TestState->Callbacks.count());
  for (auto &I : enumerate(IndicesToVisit))
    EXPECT_TRUE(ValidateVisitedRecord(I.index(), I.value()));
}

TEST_F(RandomAccessVisitorTest, CrossChunkName) {
  AppendingTypeTableBuilder Builder(GlobalState->Allocator);

  // TypeIndex 0
  ClassRecord Class(TypeRecordKind::Class);
  Class.Name = "FooClass";
  Class.Options = ClassOptions::None;
  Class.MemberCount = 0;
  Class.Size = 4U;
  Class.DerivationList = TypeIndex::fromArrayIndex(0);
  Class.FieldList = TypeIndex::fromArrayIndex(0);
  Class.VTableShape = TypeIndex::fromArrayIndex(0);
  TypeIndex IndexZero = Builder.writeLeafType(Class);

  // TypeIndex 1 refers to type index 0.
  ModifierRecord Modifier(TypeRecordKind::Modifier);
  Modifier.ModifiedType = TypeIndex::fromArrayIndex(0);
  Modifier.Modifiers = ModifierOptions::Const;
  TypeIndex IndexOne = Builder.writeLeafType(Modifier);

  // set up a type stream that refers to the above two serialized records.
  std::vector<CVType> TypeArray = {
      {Builder.records()[0]},
      {Builder.records()[1]},
  };
  BinaryItemStream<CVType> ItemStream(llvm::support::little);
  ItemStream.setItems(TypeArray);
  VarStreamArray<CVType> TypeStream(ItemStream);

  // Figure out the byte offset of the second item.
  auto ItemOneIter = TypeStream.begin();
  ++ItemOneIter;

  // Set up a partial offsets buffer that contains the first and second items
  // in separate chunks.
  std::vector<TypeIndexOffset> TIO;
  TIO.push_back({IndexZero, ulittle32_t(0u)});
  TIO.push_back({IndexOne, ulittle32_t(ItemOneIter.offset())});
  ArrayRef<uint8_t> Buffer(reinterpret_cast<const uint8_t *>(TIO.data()),
                           TIO.size() * sizeof(TypeIndexOffset));

  BinaryStreamReader Reader(Buffer, llvm::support::little);
  FixedStreamArray<TypeIndexOffset> PartialOffsets;
  ASSERT_THAT_ERROR(Reader.readArray(PartialOffsets, 2), Succeeded());

  LazyRandomTypeCollection Types(TypeStream, 2, PartialOffsets);

  StringRef Name = Types.getTypeName(IndexOne);
  EXPECT_EQ("const FooClass", Name);
}
