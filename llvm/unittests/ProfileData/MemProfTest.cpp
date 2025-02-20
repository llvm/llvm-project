//===- unittests/Support/MemProfTest.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/MemProf.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/Symbolize/SymbolizableModule.h"
#include "llvm/IR/Value.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/ProfileData/MemProfData.inc"
#include "llvm/ProfileData/MemProfReader.h"
#include "llvm/ProfileData/MemProfYAML.h"
#include "llvm/Support/raw_ostream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <initializer_list>

namespace llvm {
namespace memprof {
namespace {

using ::llvm::DIGlobal;
using ::llvm::DIInliningInfo;
using ::llvm::DILineInfo;
using ::llvm::DILineInfoSpecifier;
using ::llvm::DILocal;
using ::llvm::StringRef;
using ::llvm::object::SectionedAddress;
using ::llvm::symbolize::SymbolizableModule;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::Return;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

class MockSymbolizer : public SymbolizableModule {
public:
  MOCK_CONST_METHOD3(symbolizeInlinedCode,
                     DIInliningInfo(SectionedAddress, DILineInfoSpecifier,
                                    bool));
  // Most of the methods in the interface are unused. We only mock the
  // method that we expect to be called from the memprof reader.
  virtual DILineInfo symbolizeCode(SectionedAddress, DILineInfoSpecifier,
                                   bool) const {
    llvm_unreachable("unused");
  }
  virtual DIGlobal symbolizeData(SectionedAddress) const {
    llvm_unreachable("unused");
  }
  virtual std::vector<DILocal> symbolizeFrame(SectionedAddress) const {
    llvm_unreachable("unused");
  }
  virtual std::vector<SectionedAddress> findSymbol(StringRef Symbol,
                                                   uint64_t Offset) const {
    llvm_unreachable("unused");
  }
  virtual bool isWin32Module() const { llvm_unreachable("unused"); }
  virtual uint64_t getModulePreferredBase() const {
    llvm_unreachable("unused");
  }
};

struct MockInfo {
  std::string FunctionName;
  uint32_t Line;
  uint32_t StartLine;
  uint32_t Column;
  std::string FileName = "valid/path.cc";
};
DIInliningInfo makeInliningInfo(std::initializer_list<MockInfo> MockFrames) {
  DIInliningInfo Result;
  for (const auto &Item : MockFrames) {
    DILineInfo Frame;
    Frame.FunctionName = Item.FunctionName;
    Frame.Line = Item.Line;
    Frame.StartLine = Item.StartLine;
    Frame.Column = Item.Column;
    Frame.FileName = Item.FileName;
    Result.addFrame(Frame);
  }
  return Result;
}

llvm::SmallVector<SegmentEntry, 4> makeSegments() {
  llvm::SmallVector<SegmentEntry, 4> Result;
  // Mimic an entry for a non position independent executable.
  Result.emplace_back(0x0, 0x40000, 0x0);
  return Result;
}

const DILineInfoSpecifier specifier() {
  return DILineInfoSpecifier(
      DILineInfoSpecifier::FileLineInfoKind::RawValue,
      DILineInfoSpecifier::FunctionNameKind::LinkageName);
}

MATCHER_P4(FrameContains, FunctionName, LineOffset, Column, Inline, "") {
  const Frame &F = arg;

  const uint64_t ExpectedHash = IndexedMemProfRecord::getGUID(FunctionName);
  if (F.Function != ExpectedHash) {
    *result_listener << "Hash mismatch";
    return false;
  }
  if (F.SymbolName && *F.SymbolName != FunctionName) {
    *result_listener << "SymbolName mismatch\nWant: " << FunctionName
                     << "\nGot: " << *F.SymbolName;
    return false;
  }
  if (F.LineOffset == LineOffset && F.Column == Column &&
      F.IsInlineFrame == Inline) {
    return true;
  }
  *result_listener << "LineOffset, Column or Inline mismatch";
  return false;
}

TEST(MemProf, FillsValue) {
  auto Symbolizer = std::make_unique<MockSymbolizer>();

  EXPECT_CALL(*Symbolizer, symbolizeInlinedCode(SectionedAddress{0x1000},
                                                specifier(), false))
      .Times(1) // Only once since we remember invalid PCs.
      .WillRepeatedly(Return(makeInliningInfo({
          {"new", 70, 57, 3, "memprof/memprof_new_delete.cpp"},
      })));

  EXPECT_CALL(*Symbolizer, symbolizeInlinedCode(SectionedAddress{0x2000},
                                                specifier(), false))
      .Times(1) // Only once since we cache the result for future lookups.
      .WillRepeatedly(Return(makeInliningInfo({
          {"foo", 10, 5, 30},
          {"bar", 201, 150, 20},
      })));

  EXPECT_CALL(*Symbolizer, symbolizeInlinedCode(SectionedAddress{0x3000},
                                                specifier(), false))
      .Times(1)
      .WillRepeatedly(Return(makeInliningInfo({
          {"xyz.llvm.123", 10, 5, 30},
          {"abc", 10, 5, 30},
      })));

  CallStackMap CSM;
  CSM[0x1] = {0x1000, 0x2000, 0x3000};

  llvm::MapVector<uint64_t, MemInfoBlock> Prof;
  Prof[0x1].AllocCount = 1;

  auto Seg = makeSegments();

  RawMemProfReader Reader(std::move(Symbolizer), Seg, Prof, CSM,
                          /*KeepName=*/true);

  llvm::DenseMap<llvm::GlobalValue::GUID, MemProfRecord> Records;
  for (const auto &Pair : Reader)
    Records.insert({Pair.first, Pair.second});

  // Mock program pseudocode and expected memprof record contents.
  //
  //                              AllocSite       CallSite
  // inline foo() { new(); }         Y               N
  // bar() { foo(); }                Y               Y
  // inline xyz() { bar(); }         N               Y
  // abc() { xyz(); }                N               Y

  // We expect 4 records. We attach alloc site data to foo and bar, i.e.
  // all frames bottom up until we find a non-inline frame. We attach call site
  // data to bar, xyz and abc.
  ASSERT_THAT(Records, SizeIs(4));

  // Check the memprof record for foo.
  const llvm::GlobalValue::GUID FooId = IndexedMemProfRecord::getGUID("foo");
  ASSERT_TRUE(Records.contains(FooId));
  const MemProfRecord &Foo = Records[FooId];
  ASSERT_THAT(Foo.AllocSites, SizeIs(1));
  EXPECT_EQ(Foo.AllocSites[0].Info.getAllocCount(), 1U);
  EXPECT_THAT(Foo.AllocSites[0].CallStack[0],
              FrameContains("foo", 5U, 30U, true));
  EXPECT_THAT(Foo.AllocSites[0].CallStack[1],
              FrameContains("bar", 51U, 20U, false));
  EXPECT_THAT(Foo.AllocSites[0].CallStack[2],
              FrameContains("xyz", 5U, 30U, true));
  EXPECT_THAT(Foo.AllocSites[0].CallStack[3],
              FrameContains("abc", 5U, 30U, false));
  EXPECT_TRUE(Foo.CallSites.empty());

  // Check the memprof record for bar.
  const llvm::GlobalValue::GUID BarId = IndexedMemProfRecord::getGUID("bar");
  ASSERT_TRUE(Records.contains(BarId));
  const MemProfRecord &Bar = Records[BarId];
  ASSERT_THAT(Bar.AllocSites, SizeIs(1));
  EXPECT_EQ(Bar.AllocSites[0].Info.getAllocCount(), 1U);
  EXPECT_THAT(Bar.AllocSites[0].CallStack[0],
              FrameContains("foo", 5U, 30U, true));
  EXPECT_THAT(Bar.AllocSites[0].CallStack[1],
              FrameContains("bar", 51U, 20U, false));
  EXPECT_THAT(Bar.AllocSites[0].CallStack[2],
              FrameContains("xyz", 5U, 30U, true));
  EXPECT_THAT(Bar.AllocSites[0].CallStack[3],
              FrameContains("abc", 5U, 30U, false));

  EXPECT_THAT(Bar.CallSites,
              ElementsAre(ElementsAre(FrameContains("foo", 5U, 30U, true),
                                      FrameContains("bar", 51U, 20U, false))));

  // Check the memprof record for xyz.
  const llvm::GlobalValue::GUID XyzId = IndexedMemProfRecord::getGUID("xyz");
  ASSERT_TRUE(Records.contains(XyzId));
  const MemProfRecord &Xyz = Records[XyzId];
  // Expect the entire frame even though in practice we only need the first
  // entry here.
  EXPECT_THAT(Xyz.CallSites,
              ElementsAre(ElementsAre(FrameContains("xyz", 5U, 30U, true),
                                      FrameContains("abc", 5U, 30U, false))));

  // Check the memprof record for abc.
  const llvm::GlobalValue::GUID AbcId = IndexedMemProfRecord::getGUID("abc");
  ASSERT_TRUE(Records.contains(AbcId));
  const MemProfRecord &Abc = Records[AbcId];
  EXPECT_TRUE(Abc.AllocSites.empty());
  EXPECT_THAT(Abc.CallSites,
              ElementsAre(ElementsAre(FrameContains("xyz", 5U, 30U, true),
                                      FrameContains("abc", 5U, 30U, false))));
}

TEST(MemProf, PortableWrapper) {
  MemInfoBlock Info(/*size=*/16, /*access_count=*/7, /*alloc_timestamp=*/1000,
                    /*dealloc_timestamp=*/2000, /*alloc_cpu=*/3,
                    /*dealloc_cpu=*/4, /*Histogram=*/0, /*HistogramSize=*/0);

  const auto Schema = getFullSchema();
  PortableMemInfoBlock WriteBlock(Info, Schema);

  std::string Buffer;
  llvm::raw_string_ostream OS(Buffer);
  WriteBlock.serialize(Schema, OS);

  PortableMemInfoBlock ReadBlock(
      Schema, reinterpret_cast<const unsigned char *>(Buffer.data()));

  EXPECT_EQ(ReadBlock, WriteBlock);
  // Here we compare directly with the actual counts instead of MemInfoBlock
  // members. Since the MemInfoBlock struct is packed and the EXPECT_EQ macros
  // take a reference to the params, this results in unaligned accesses.
  EXPECT_EQ(1UL, ReadBlock.getAllocCount());
  EXPECT_EQ(7ULL, ReadBlock.getTotalAccessCount());
  EXPECT_EQ(3UL, ReadBlock.getAllocCpuId());
}

TEST(MemProf, RecordSerializationRoundTripVerion2) {
  const auto Schema = getFullSchema();

  MemInfoBlock Info(/*size=*/16, /*access_count=*/7, /*alloc_timestamp=*/1000,
                    /*dealloc_timestamp=*/2000, /*alloc_cpu=*/3,
                    /*dealloc_cpu=*/4, /*Histogram=*/0, /*HistogramSize=*/0);

  llvm::SmallVector<CallStackId> CallStackIds = {0x123, 0x456};

  llvm::SmallVector<CallStackId> CallSiteIds = {0x333, 0x444};

  IndexedMemProfRecord Record;
  for (const auto &CSId : CallStackIds) {
    // Use the same info block for both allocation sites.
    Record.AllocSites.emplace_back(CSId, Info);
  }
  Record.CallSiteIds.assign(CallSiteIds);

  std::string Buffer;
  llvm::raw_string_ostream OS(Buffer);
  Record.serialize(Schema, OS, Version2);

  const IndexedMemProfRecord GotRecord = IndexedMemProfRecord::deserialize(
      Schema, reinterpret_cast<const unsigned char *>(Buffer.data()), Version2);

  EXPECT_EQ(Record, GotRecord);
}

TEST(MemProf, RecordSerializationRoundTripVersion2HotColdSchema) {
  const auto Schema = getHotColdSchema();

  MemInfoBlock Info;
  Info.AllocCount = 11;
  Info.TotalSize = 22;
  Info.TotalLifetime = 33;
  Info.TotalLifetimeAccessDensity = 44;

  llvm::SmallVector<CallStackId> CallStackIds = {0x123, 0x456};

  llvm::SmallVector<CallStackId> CallSiteIds = {0x333, 0x444};

  IndexedMemProfRecord Record;
  for (const auto &CSId : CallStackIds) {
    // Use the same info block for both allocation sites.
    Record.AllocSites.emplace_back(CSId, Info, Schema);
  }
  Record.CallSiteIds.assign(CallSiteIds);

  std::bitset<llvm::to_underlying(Meta::Size)> SchemaBitSet;
  for (auto Id : Schema)
    SchemaBitSet.set(llvm::to_underlying(Id));

  // Verify that SchemaBitSet has the fields we expect and nothing else, which
  // we check with count().
  EXPECT_EQ(SchemaBitSet.count(), 4U);
  EXPECT_TRUE(SchemaBitSet[llvm::to_underlying(Meta::AllocCount)]);
  EXPECT_TRUE(SchemaBitSet[llvm::to_underlying(Meta::TotalSize)]);
  EXPECT_TRUE(SchemaBitSet[llvm::to_underlying(Meta::TotalLifetime)]);
  EXPECT_TRUE(
      SchemaBitSet[llvm::to_underlying(Meta::TotalLifetimeAccessDensity)]);

  // Verify that Schema has propagated all the way to the Info field in each
  // IndexedAllocationInfo.
  ASSERT_THAT(Record.AllocSites, SizeIs(2));
  EXPECT_EQ(Record.AllocSites[0].Info.getSchema(), SchemaBitSet);
  EXPECT_EQ(Record.AllocSites[1].Info.getSchema(), SchemaBitSet);

  std::string Buffer;
  llvm::raw_string_ostream OS(Buffer);
  Record.serialize(Schema, OS, Version2);

  const IndexedMemProfRecord GotRecord = IndexedMemProfRecord::deserialize(
      Schema, reinterpret_cast<const unsigned char *>(Buffer.data()), Version2);

  // Verify that Schema comes back correctly after deserialization. Technically,
  // the comparison between Record and GotRecord below includes the comparison
  // of their Schemas, but we'll verify the Schemas on our own.
  ASSERT_THAT(GotRecord.AllocSites, SizeIs(2));
  EXPECT_EQ(GotRecord.AllocSites[0].Info.getSchema(), SchemaBitSet);
  EXPECT_EQ(GotRecord.AllocSites[1].Info.getSchema(), SchemaBitSet);

  EXPECT_EQ(Record, GotRecord);
}

TEST(MemProf, SymbolizationFilter) {
  auto Symbolizer = std::make_unique<MockSymbolizer>();

  EXPECT_CALL(*Symbolizer, symbolizeInlinedCode(SectionedAddress{0x1000},
                                                specifier(), false))
      .Times(1) // once since we don't lookup invalid PCs repeatedly.
      .WillRepeatedly(Return(makeInliningInfo({
          {"malloc", 70, 57, 3, "memprof/memprof_malloc_linux.cpp"},
      })));

  EXPECT_CALL(*Symbolizer, symbolizeInlinedCode(SectionedAddress{0x2000},
                                                specifier(), false))
      .Times(1) // once since we don't lookup invalid PCs repeatedly.
      .WillRepeatedly(Return(makeInliningInfo({
          {"new", 70, 57, 3, "memprof/memprof_new_delete.cpp"},
      })));

  EXPECT_CALL(*Symbolizer, symbolizeInlinedCode(SectionedAddress{0x3000},
                                                specifier(), false))
      .Times(1) // once since we don't lookup invalid PCs repeatedly.
      .WillRepeatedly(Return(makeInliningInfo({
          {DILineInfo::BadString, 0, 0, 0},
      })));

  EXPECT_CALL(*Symbolizer, symbolizeInlinedCode(SectionedAddress{0x4000},
                                                specifier(), false))
      .Times(1)
      .WillRepeatedly(Return(makeInliningInfo({
          {"foo", 10, 5, 30, "memprof/memprof_test_file.cpp"},
      })));

  EXPECT_CALL(*Symbolizer, symbolizeInlinedCode(SectionedAddress{0x5000},
                                                specifier(), false))
      .Times(1)
      .WillRepeatedly(Return(makeInliningInfo({
          // Depending on how the runtime was compiled, only the filename
          // may be present in the debug information.
          {"malloc", 70, 57, 3, "memprof_malloc_linux.cpp"},
      })));

  CallStackMap CSM;
  CSM[0x1] = {0x1000, 0x2000, 0x3000, 0x4000};
  // This entry should be dropped since all PCs are either not
  // symbolizable or belong to the runtime.
  CSM[0x2] = {0x1000, 0x2000, 0x5000};

  llvm::MapVector<uint64_t, MemInfoBlock> Prof;
  Prof[0x1].AllocCount = 1;
  Prof[0x2].AllocCount = 1;

  auto Seg = makeSegments();

  RawMemProfReader Reader(std::move(Symbolizer), Seg, Prof, CSM);

  llvm::SmallVector<MemProfRecord, 1> Records;
  for (const auto &KeyRecordPair : Reader)
    Records.push_back(KeyRecordPair.second);

  ASSERT_THAT(Records, SizeIs(1));
  ASSERT_THAT(Records[0].AllocSites, SizeIs(1));
  EXPECT_THAT(Records[0].AllocSites[0].CallStack,
              ElementsAre(FrameContains("foo", 5U, 30U, false)));
}

TEST(MemProf, BaseMemProfReader) {
  IndexedMemProfData MemProfData;
  Frame F1(/*Hash=*/IndexedMemProfRecord::getGUID("foo"), /*LineOffset=*/20,
           /*Column=*/5, /*IsInlineFrame=*/true);
  Frame F2(/*Hash=*/IndexedMemProfRecord::getGUID("bar"), /*LineOffset=*/10,
           /*Column=*/2, /*IsInlineFrame=*/false);
  auto F1Id = MemProfData.addFrame(F1);
  auto F2Id = MemProfData.addFrame(F2);

  llvm::SmallVector<FrameId> CallStack{F1Id, F2Id};
  CallStackId CSId = MemProfData.addCallStack(std::move(CallStack));

  IndexedMemProfRecord FakeRecord;
  MemInfoBlock Block;
  Block.AllocCount = 1U, Block.TotalAccessDensity = 4,
  Block.TotalLifetime = 200001;
  FakeRecord.AllocSites.emplace_back(/*CSId=*/CSId, /*MB=*/Block);
  MemProfData.Records.try_emplace(0x1234, std::move(FakeRecord));

  MemProfReader Reader(std::move(MemProfData));

  llvm::SmallVector<MemProfRecord, 1> Records;
  for (const auto &KeyRecordPair : Reader)
    Records.push_back(KeyRecordPair.second);

  ASSERT_THAT(Records, SizeIs(1));
  ASSERT_THAT(Records[0].AllocSites, SizeIs(1));
  EXPECT_THAT(Records[0].AllocSites[0].CallStack,
              ElementsAre(FrameContains("foo", 20U, 5U, true),
                          FrameContains("bar", 10U, 2U, false)));
}

TEST(MemProf, BaseMemProfReaderWithCSIdMap) {
  IndexedMemProfData MemProfData;
  Frame F1(/*Hash=*/IndexedMemProfRecord::getGUID("foo"), /*LineOffset=*/20,
           /*Column=*/5, /*IsInlineFrame=*/true);
  Frame F2(/*Hash=*/IndexedMemProfRecord::getGUID("bar"), /*LineOffset=*/10,
           /*Column=*/2, /*IsInlineFrame=*/false);
  auto F1Id = MemProfData.addFrame(F1);
  auto F2Id = MemProfData.addFrame(F2);

  llvm::SmallVector<FrameId> CallStack = {F1Id, F2Id};
  auto CSId = MemProfData.addCallStack(std::move(CallStack));

  IndexedMemProfRecord FakeRecord;
  MemInfoBlock Block;
  Block.AllocCount = 1U, Block.TotalAccessDensity = 4,
  Block.TotalLifetime = 200001;
  FakeRecord.AllocSites.emplace_back(/*CSId=*/CSId, /*MB=*/Block);
  MemProfData.Records.try_emplace(0x1234, std::move(FakeRecord));

  MemProfReader Reader(std::move(MemProfData));

  llvm::SmallVector<MemProfRecord, 1> Records;
  for (const auto &KeyRecordPair : Reader)
    Records.push_back(KeyRecordPair.second);

  ASSERT_THAT(Records, SizeIs(1));
  ASSERT_THAT(Records[0].AllocSites, SizeIs(1));
  EXPECT_THAT(Records[0].AllocSites[0].CallStack,
              ElementsAre(FrameContains("foo", 20U, 5U, true),
                          FrameContains("bar", 10U, 2U, false)));
}

TEST(MemProf, IndexedMemProfRecordToMemProfRecord) {
  // Verify that MemProfRecord can be constructed from IndexedMemProfRecord with
  // CallStackIds only.

  IndexedMemProfData MemProfData;
  Frame F1(1, 0, 0, false);
  Frame F2(2, 0, 0, false);
  Frame F3(3, 0, 0, false);
  Frame F4(4, 0, 0, false);
  auto F1Id = MemProfData.addFrame(F1);
  auto F2Id = MemProfData.addFrame(F2);
  auto F3Id = MemProfData.addFrame(F3);
  auto F4Id = MemProfData.addFrame(F4);

  llvm::SmallVector<FrameId> CS1 = {F1Id, F2Id};
  llvm::SmallVector<FrameId> CS2 = {F1Id, F3Id};
  llvm::SmallVector<FrameId> CS3 = {F2Id, F3Id};
  llvm::SmallVector<FrameId> CS4 = {F2Id, F4Id};
  auto CS1Id = MemProfData.addCallStack(std::move(CS1));
  auto CS2Id = MemProfData.addCallStack(std::move(CS2));
  auto CS3Id = MemProfData.addCallStack(std::move(CS3));
  auto CS4Id = MemProfData.addCallStack(std::move(CS4));

  IndexedMemProfRecord IndexedRecord;
  IndexedAllocationInfo AI;
  AI.CSId = CS1Id;
  IndexedRecord.AllocSites.push_back(AI);
  AI.CSId = CS2Id;
  IndexedRecord.AllocSites.push_back(AI);
  IndexedRecord.CallSiteIds.push_back(CS3Id);
  IndexedRecord.CallSiteIds.push_back(CS4Id);

  IndexedCallstackIdConveter CSIdConv(MemProfData);

  MemProfRecord Record = IndexedRecord.toMemProfRecord(CSIdConv);

  // Make sure that all lookups are successful.
  ASSERT_EQ(CSIdConv.FrameIdConv.LastUnmappedId, std::nullopt);
  ASSERT_EQ(CSIdConv.CSIdConv.LastUnmappedId, std::nullopt);

  // Verify the contents of Record.
  ASSERT_THAT(Record.AllocSites, SizeIs(2));
  EXPECT_THAT(Record.AllocSites[0].CallStack, ElementsAre(F1, F2));
  EXPECT_THAT(Record.AllocSites[1].CallStack, ElementsAre(F1, F3));
  EXPECT_THAT(Record.CallSites,
              ElementsAre(ElementsAre(F2, F3), ElementsAre(F2, F4)));
}

// Populate those fields returned by getHotColdSchema.
MemInfoBlock makePartialMIB() {
  MemInfoBlock MIB;
  MIB.AllocCount = 1;
  MIB.TotalSize = 5;
  MIB.TotalLifetime = 10;
  MIB.TotalLifetimeAccessDensity = 23;
  return MIB;
}

TEST(MemProf, MissingCallStackId) {
  // Use a non-existent CallStackId to trigger a mapping error in
  // toMemProfRecord.
  IndexedAllocationInfo AI(0xdeadbeefU, makePartialMIB(), getHotColdSchema());

  IndexedMemProfRecord IndexedMR;
  IndexedMR.AllocSites.push_back(AI);

  // Create empty maps.
  IndexedMemProfData MemProfData;
  IndexedCallstackIdConveter CSIdConv(MemProfData);

  // We are only interested in errors, not the return value.
  (void)IndexedMR.toMemProfRecord(CSIdConv);

  ASSERT_TRUE(CSIdConv.CSIdConv.LastUnmappedId.has_value());
  EXPECT_EQ(*CSIdConv.CSIdConv.LastUnmappedId, 0xdeadbeefU);
  EXPECT_EQ(CSIdConv.FrameIdConv.LastUnmappedId, std::nullopt);
}

TEST(MemProf, MissingFrameId) {
  // An empty Frame map to trigger a mapping error.
  IndexedMemProfData MemProfData;
  auto CSId = MemProfData.addCallStack(SmallVector<FrameId>{2, 3});

  IndexedMemProfRecord IndexedMR;
  IndexedMR.AllocSites.emplace_back(CSId, makePartialMIB(), getHotColdSchema());

  IndexedCallstackIdConveter CSIdConv(MemProfData);

  // We are only interested in errors, not the return value.
  (void)IndexedMR.toMemProfRecord(CSIdConv);

  EXPECT_EQ(CSIdConv.CSIdConv.LastUnmappedId, std::nullopt);
  ASSERT_TRUE(CSIdConv.FrameIdConv.LastUnmappedId.has_value());
  EXPECT_EQ(*CSIdConv.FrameIdConv.LastUnmappedId, 3U);
}

// Verify CallStackRadixTreeBuilder can handle empty inputs.
TEST(MemProf, RadixTreeBuilderEmpty) {
  llvm::DenseMap<FrameId, LinearFrameId> MemProfFrameIndexes;
  IndexedMemProfData MemProfData;
  llvm::DenseMap<FrameId, FrameStat> FrameHistogram =
      computeFrameHistogram<FrameId>(MemProfData.CallStacks);
  CallStackRadixTreeBuilder<FrameId> Builder;
  Builder.build(std::move(MemProfData.CallStacks), &MemProfFrameIndexes,
                FrameHistogram);
  ASSERT_THAT(Builder.getRadixArray(), IsEmpty());
  const auto Mappings = Builder.takeCallStackPos();
  ASSERT_THAT(Mappings, IsEmpty());
}

// Verify CallStackRadixTreeBuilder can handle one trivial call stack.
TEST(MemProf, RadixTreeBuilderOne) {
  llvm::DenseMap<FrameId, LinearFrameId> MemProfFrameIndexes = {
      {11, 1}, {12, 2}, {13, 3}};
  llvm::SmallVector<FrameId> CS1 = {13, 12, 11};
  IndexedMemProfData MemProfData;
  auto CS1Id = MemProfData.addCallStack(std::move(CS1));
  llvm::DenseMap<FrameId, FrameStat> FrameHistogram =
      computeFrameHistogram<FrameId>(MemProfData.CallStacks);
  CallStackRadixTreeBuilder<FrameId> Builder;
  Builder.build(std::move(MemProfData.CallStacks), &MemProfFrameIndexes,
                FrameHistogram);
  EXPECT_THAT(Builder.getRadixArray(),
              ElementsAre(3U, // Size of CS1,
                          3U, // MemProfFrameIndexes[13]
                          2U, // MemProfFrameIndexes[12]
                          1U  // MemProfFrameIndexes[11]
                          ));
  const auto Mappings = Builder.takeCallStackPos();
  EXPECT_THAT(Mappings, UnorderedElementsAre(Pair(CS1Id, 0U)));
}

// Verify CallStackRadixTreeBuilder can form a link between two call stacks.
TEST(MemProf, RadixTreeBuilderTwo) {
  llvm::DenseMap<FrameId, LinearFrameId> MemProfFrameIndexes = {
      {11, 1}, {12, 2}, {13, 3}};
  llvm::SmallVector<FrameId> CS1 = {12, 11};
  llvm::SmallVector<FrameId> CS2 = {13, 12, 11};
  IndexedMemProfData MemProfData;
  auto CS1Id = MemProfData.addCallStack(std::move(CS1));
  auto CS2Id = MemProfData.addCallStack(std::move(CS2));
  llvm::DenseMap<FrameId, FrameStat> FrameHistogram =
      computeFrameHistogram<FrameId>(MemProfData.CallStacks);
  CallStackRadixTreeBuilder<FrameId> Builder;
  Builder.build(std::move(MemProfData.CallStacks), &MemProfFrameIndexes,
                FrameHistogram);
  EXPECT_THAT(Builder.getRadixArray(),
              ElementsAre(2U,                        // Size of CS1
                          static_cast<uint32_t>(-3), // Jump 3 steps
                          3U,                        // Size of CS2
                          3U,                        // MemProfFrameIndexes[13]
                          2U,                        // MemProfFrameIndexes[12]
                          1U                         // MemProfFrameIndexes[11]
                          ));
  const auto Mappings = Builder.takeCallStackPos();
  EXPECT_THAT(Mappings, UnorderedElementsAre(Pair(CS1Id, 0U), Pair(CS2Id, 2U)));
}

// Verify CallStackRadixTreeBuilder can form a jump to a prefix that itself has
// another jump to another prefix.
TEST(MemProf, RadixTreeBuilderSuccessiveJumps) {
  llvm::DenseMap<FrameId, LinearFrameId> MemProfFrameIndexes = {
      {11, 1}, {12, 2}, {13, 3}, {14, 4}, {15, 5}, {16, 6}, {17, 7}, {18, 8},
  };
  llvm::SmallVector<FrameId> CS1 = {14, 13, 12, 11};
  llvm::SmallVector<FrameId> CS2 = {15, 13, 12, 11};
  llvm::SmallVector<FrameId> CS3 = {17, 16, 12, 11};
  llvm::SmallVector<FrameId> CS4 = {18, 16, 12, 11};
  IndexedMemProfData MemProfData;
  auto CS1Id = MemProfData.addCallStack(std::move(CS1));
  auto CS2Id = MemProfData.addCallStack(std::move(CS2));
  auto CS3Id = MemProfData.addCallStack(std::move(CS3));
  auto CS4Id = MemProfData.addCallStack(std::move(CS4));
  llvm::DenseMap<FrameId, FrameStat> FrameHistogram =
      computeFrameHistogram<FrameId>(MemProfData.CallStacks);
  CallStackRadixTreeBuilder<FrameId> Builder;
  Builder.build(std::move(MemProfData.CallStacks), &MemProfFrameIndexes,
                FrameHistogram);
  EXPECT_THAT(Builder.getRadixArray(),
              ElementsAre(4U,                        // Size of CS1
                          4U,                        // MemProfFrameIndexes[14]
                          static_cast<uint32_t>(-3), // Jump 3 steps
                          4U,                        // Size of CS2
                          5U,                        // MemProfFrameIndexes[15]
                          3U,                        // MemProfFrameIndexes[13]
                          static_cast<uint32_t>(-7), // Jump 7 steps
                          4U,                        // Size of CS3
                          7U,                        // MemProfFrameIndexes[17]
                          static_cast<uint32_t>(-3), // Jump 3 steps
                          4U,                        // Size of CS4
                          8U,                        // MemProfFrameIndexes[18]
                          6U,                        // MemProfFrameIndexes[16]
                          2U,                        // MemProfFrameIndexes[12]
                          1U                         // MemProfFrameIndexes[11]
                          ));
  const auto Mappings = Builder.takeCallStackPos();
  EXPECT_THAT(Mappings,
              UnorderedElementsAre(Pair(CS1Id, 0U), Pair(CS2Id, 3U),
                                   Pair(CS3Id, 7U), Pair(CS4Id, 10U)));
}

// Verify that we can parse YAML and retrieve IndexedMemProfData as expected.
TEST(MemProf, YAMLParser) {
  StringRef YAMLData = R"YAML(
---
HeapProfileRecords:
- GUID: 0xdeadbeef12345678
  AllocSites:
  - Callstack:
    - {Function: 0x100, LineOffset: 11, Column: 10, IsInlineFrame: true}
    - {Function: 0x200, LineOffset: 22, Column: 20, IsInlineFrame: false}
    MemInfoBlock:
      AllocCount: 777
      TotalSize: 888
  - Callstack:
    - {Function: 0x300, LineOffset: 33, Column: 30, IsInlineFrame: false}
    - {Function: 0x400, LineOffset: 44, Column: 40, IsInlineFrame: true}
    MemInfoBlock:
      AllocCount: 666
      TotalSize: 555
  CallSites:
  - - {Function: 0x500, LineOffset: 55, Column: 50, IsInlineFrame: true}
    - {Function: 0x600, LineOffset: 66, Column: 60, IsInlineFrame: false}
  - - {Function: 0x700, LineOffset: 77, Column: 70, IsInlineFrame: true}
    - {Function: 0x800, LineOffset: 88, Column: 80, IsInlineFrame: false}
)YAML";

  YAMLMemProfReader YAMLReader;
  YAMLReader.parse(YAMLData);
  IndexedMemProfData MemProfData = YAMLReader.takeMemProfData();

  // Verify the entire contents of MemProfData.Records.
  ASSERT_THAT(MemProfData.Records, SizeIs(1));
  const auto &[GUID, IndexedRecord] = MemProfData.Records.front();
  EXPECT_EQ(GUID, 0xdeadbeef12345678ULL);

  IndexedCallstackIdConveter CSIdConv(MemProfData);
  MemProfRecord Record = IndexedRecord.toMemProfRecord(CSIdConv);

  ASSERT_THAT(Record.AllocSites, SizeIs(2));
  EXPECT_THAT(
      Record.AllocSites[0].CallStack,
      ElementsAre(Frame(0x100, 11, 10, true), Frame(0x200, 22, 20, false)));
  EXPECT_EQ(Record.AllocSites[0].Info.getAllocCount(), 777U);
  EXPECT_EQ(Record.AllocSites[0].Info.getTotalSize(), 888U);
  EXPECT_THAT(
      Record.AllocSites[1].CallStack,
      ElementsAre(Frame(0x300, 33, 30, false), Frame(0x400, 44, 40, true)));
  EXPECT_EQ(Record.AllocSites[1].Info.getAllocCount(), 666U);
  EXPECT_EQ(Record.AllocSites[1].Info.getTotalSize(), 555U);
  EXPECT_THAT(Record.CallSites,
              ElementsAre(ElementsAre(Frame(0x500, 55, 50, true),
                                      Frame(0x600, 66, 60, false)),
                          ElementsAre(Frame(0x700, 77, 70, true),
                                      Frame(0x800, 88, 80, false))));
}

// Verify that the YAML parser accepts a GUID expressed as a function name.
TEST(MemProf, YAMLParserGUID) {
  StringRef YAMLData = R"YAML(
---
HeapProfileRecords:
- GUID: _Z3fooi
  AllocSites:
  - Callstack:
    - {Function: 0x100, LineOffset: 11, Column: 10, IsInlineFrame: true}
    MemInfoBlock: {}
  CallSites: []
)YAML";

  YAMLMemProfReader YAMLReader;
  YAMLReader.parse(YAMLData);
  IndexedMemProfData MemProfData = YAMLReader.takeMemProfData();

  // Verify the entire contents of MemProfData.Records.
  ASSERT_THAT(MemProfData.Records, SizeIs(1));
  const auto &[GUID, IndexedRecord] = MemProfData.Records.front();
  EXPECT_EQ(GUID, IndexedMemProfRecord::getGUID("_Z3fooi"));

  IndexedCallstackIdConveter CSIdConv(MemProfData);
  MemProfRecord Record = IndexedRecord.toMemProfRecord(CSIdConv);

  ASSERT_THAT(Record.AllocSites, SizeIs(1));
  EXPECT_THAT(Record.AllocSites[0].CallStack,
              ElementsAre(Frame(0x100, 11, 10, true)));
  EXPECT_THAT(Record.CallSites, IsEmpty());
}

template <typename T> std::string serializeInYAML(T &Val) {
  std::string Out;
  llvm::raw_string_ostream OS(Out);
  llvm::yaml::Output Yout(OS);
  Yout << Val;
  return Out;
}

TEST(MemProf, YAMLWriterFrame) {
  Frame F(0x0123456789abcdefULL, 22, 33, true);

  std::string Out = serializeInYAML(F);
  EXPECT_EQ(Out, R"YAML(---
{ Function: 0x0123456789abcdef, LineOffset: 22, Column: 33, IsInlineFrame: true }
...
)YAML");
}

TEST(MemProf, YAMLWriterMIB) {
  MemInfoBlock MIB;
  MIB.AllocCount = 111;
  MIB.TotalSize = 222;
  MIB.TotalLifetime = 333;
  MIB.TotalLifetimeAccessDensity = 444;
  PortableMemInfoBlock PMIB(MIB, getHotColdSchema());

  std::string Out = serializeInYAML(PMIB);
  EXPECT_EQ(Out, R"YAML(---
AllocCount:      111
TotalSize:       222
TotalLifetime:   333
TotalLifetimeAccessDensity: 444
...
)YAML");
}
} // namespace
} // namespace memprof
} // namespace llvm
