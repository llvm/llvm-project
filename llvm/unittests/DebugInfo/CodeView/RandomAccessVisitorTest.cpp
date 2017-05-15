//===- llvm/unittest/DebugInfo/CodeView/RandomAccessVisitorTest.cpp -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ErrorChecking.h"

#include "llvm/ADT/SmallBitVector.h"
#include "llvm/DebugInfo/CodeView/CVTypeVisitor.h"
#include "llvm/DebugInfo/CodeView/RandomAccessTypeVisitor.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/CodeView/TypeRecordMapping.h"
#include "llvm/DebugInfo/CodeView/TypeSerializer.h"
#include "llvm/DebugInfo/CodeView/TypeServerHandler.h"
#include "llvm/DebugInfo/CodeView/TypeTableBuilder.h"
#include "llvm/DebugInfo/CodeView/TypeVisitorCallbackPipeline.h"
#include "llvm/DebugInfo/CodeView/TypeVisitorCallbacks.h"
#include "llvm/DebugInfo/PDB/Native/RawTypes.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/BinaryItemStream.h"
#include "llvm/Support/Error.h"

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
  if (R1.Type != R2.Type)
    return false;
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
  virtual Error visitTypeBegin(CVType &CVR, TypeIndex Index) {
    Indices.push_back(Index);
    return Error::success();
  }
  virtual Error visitKnownRecord(CVType &CVR, ArrayRecord &AR) {
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
    GlobalState = llvm::make_unique<GlobalTestState>();

    TypeTableBuilder Builder(GlobalState->Allocator);

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
      GlobalState->Indices.push_back(Builder.writeKnownType(AR));

      CVType Type(TypeLeafKind::LF_ARRAY, Builder.records().back());
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
    TestState = llvm::make_unique<PerTestState>();

    TestState->Pipeline.addCallbackToPipeline(TestState->Deserializer);
    TestState->Pipeline.addCallbackToPipeline(TestState->Callbacks);
  }

  void TearDown() override { TestState.reset(); }

protected:
  bool ValidateDatabaseRecord(const RandomAccessTypeVisitor &Visitor,
                              uint32_t Index) {
    TypeIndex TI = TypeIndex::fromArrayIndex(Index);
    if (!Visitor.database().contains(TI))
      return false;
    if (GlobalState->TypeVector[Index] != Visitor.database().getTypeRecord(TI))
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

    TypeVisitorCallbackPipeline Pipeline;
    TypeDeserializer Deserializer;
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
  RandomAccessTypeVisitor Visitor(GlobalState->TypeArray,
                                  GlobalState->TypeVector.size(),
                                  TestState->Offsets);

  std::vector<uint32_t> IndicesToVisit = {5, 5, 5};

  for (uint32_t I : IndicesToVisit) {
    TypeIndex TI = TypeIndex::fromArrayIndex(I);
    EXPECT_NO_ERROR(Visitor.visitTypeIndex(TI, TestState->Pipeline));
  }

  // [0,8) should be present
  EXPECT_EQ(8u, Visitor.database().size());
  for (uint32_t I = 0; I < 8; ++I)
    EXPECT_TRUE(ValidateDatabaseRecord(Visitor, I));

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

  RandomAccessTypeVisitor Visitor(GlobalState->TypeArray,
                                  GlobalState->TypeVector.size(),
                                  TestState->Offsets);

  for (uint32_t I : IndicesToVisit) {
    TypeIndex TI = TypeIndex::fromArrayIndex(I);
    EXPECT_NO_ERROR(Visitor.visitTypeIndex(TI, TestState->Pipeline));
  }

  // [0, 7]
  EXPECT_EQ(8u, Visitor.database().size());
  for (uint32_t I = 0; I < 8; ++I)
    EXPECT_TRUE(ValidateDatabaseRecord(Visitor, I));

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

  RandomAccessTypeVisitor Visitor(GlobalState->TypeArray,
                                  GlobalState->TypeVector.size(),
                                  TestState->Offsets);

  for (uint32_t I : IndicesToVisit) {
    TypeIndex TI = TypeIndex::fromArrayIndex(I);
    EXPECT_NO_ERROR(Visitor.visitTypeIndex(TI, TestState->Pipeline));
  }

  // [0, 7]
  EXPECT_EQ(8u, Visitor.database().size());
  for (uint32_t I = 0; I < 8; ++I)
    EXPECT_TRUE(ValidateDatabaseRecord(Visitor, I));

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

  RandomAccessTypeVisitor Visitor(GlobalState->TypeArray,
                                  GlobalState->TypeVector.size(),
                                  TestState->Offsets);

  for (uint32_t I : IndicesToVisit) {
    TypeIndex TI = TypeIndex::fromArrayIndex(I);
    EXPECT_NO_ERROR(Visitor.visitTypeIndex(TI, TestState->Pipeline));
  }

  // [0, 8) should be visited.
  EXPECT_EQ(8u, Visitor.database().size());
  for (uint32_t I = 0; I < 8; ++I)
    EXPECT_TRUE(ValidateDatabaseRecord(Visitor, I));

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

  RandomAccessTypeVisitor Visitor(GlobalState->TypeArray,
                                  GlobalState->TypeVector.size(),
                                  TestState->Offsets);

  for (uint32_t I : IndicesToVisit) {
    TypeIndex TI = TypeIndex::fromArrayIndex(I);
    EXPECT_NO_ERROR(Visitor.visitTypeIndex(TI, TestState->Pipeline));
  }

  // [4, 9)
  EXPECT_EQ(5u, Visitor.database().size());
  for (uint32_t I = 4; I < 9; ++I)
    EXPECT_TRUE(ValidateDatabaseRecord(Visitor, I));

  // 5, 7
  EXPECT_EQ(2u, TestState->Callbacks.count());
  for (auto &I : enumerate(IndicesToVisit))
    EXPECT_TRUE(ValidateVisitedRecord(I.index(), I.value()));
}
