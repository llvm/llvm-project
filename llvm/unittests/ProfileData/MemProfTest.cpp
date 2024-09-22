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
#include "llvm/Support/raw_ostream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <initializer_list>

namespace {

using ::llvm::DIGlobal;
using ::llvm::DIInliningInfo;
using ::llvm::DILineInfo;
using ::llvm::DILineInfoSpecifier;
using ::llvm::DILocal;
using ::llvm::StringRef;
using ::llvm::memprof::CallStackId;
using ::llvm::memprof::CallStackMap;
using ::llvm::memprof::Frame;
using ::llvm::memprof::FrameId;
using ::llvm::memprof::IndexedAllocationInfo;
using ::llvm::memprof::IndexedMemProfRecord;
using ::llvm::memprof::MemInfoBlock;
using ::llvm::memprof::MemProfReader;
using ::llvm::memprof::MemProfRecord;
using ::llvm::memprof::MemProfSchema;
using ::llvm::memprof::Meta;
using ::llvm::memprof::PortableMemInfoBlock;
using ::llvm::memprof::RawMemProfReader;
using ::llvm::memprof::SegmentEntry;
using ::llvm::object::SectionedAddress;
using ::llvm::symbolize::SymbolizableModule;
using ::testing::Return;
using ::testing::SizeIs;

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
  std::unique_ptr<MockSymbolizer> Symbolizer(new MockSymbolizer());

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
  for (const auto &Pair : Reader) {
    Records.insert({Pair.first, Pair.second});
  }

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

  ASSERT_THAT(Bar.CallSites, SizeIs(1));
  ASSERT_THAT(Bar.CallSites[0], SizeIs(2));
  EXPECT_THAT(Bar.CallSites[0][0], FrameContains("foo", 5U, 30U, true));
  EXPECT_THAT(Bar.CallSites[0][1], FrameContains("bar", 51U, 20U, false));

  // Check the memprof record for xyz.
  const llvm::GlobalValue::GUID XyzId = IndexedMemProfRecord::getGUID("xyz");
  ASSERT_TRUE(Records.contains(XyzId));
  const MemProfRecord &Xyz = Records[XyzId];
  ASSERT_THAT(Xyz.CallSites, SizeIs(1));
  ASSERT_THAT(Xyz.CallSites[0], SizeIs(2));
  // Expect the entire frame even though in practice we only need the first
  // entry here.
  EXPECT_THAT(Xyz.CallSites[0][0], FrameContains("xyz", 5U, 30U, true));
  EXPECT_THAT(Xyz.CallSites[0][1], FrameContains("abc", 5U, 30U, false));

  // Check the memprof record for abc.
  const llvm::GlobalValue::GUID AbcId = IndexedMemProfRecord::getGUID("abc");
  ASSERT_TRUE(Records.contains(AbcId));
  const MemProfRecord &Abc = Records[AbcId];
  EXPECT_TRUE(Abc.AllocSites.empty());
  ASSERT_THAT(Abc.CallSites, SizeIs(1));
  ASSERT_THAT(Abc.CallSites[0], SizeIs(2));
  EXPECT_THAT(Abc.CallSites[0][0], FrameContains("xyz", 5U, 30U, true));
  EXPECT_THAT(Abc.CallSites[0][1], FrameContains("abc", 5U, 30U, false));
}

TEST(MemProf, PortableWrapper) {
  MemInfoBlock Info(/*size=*/16, /*access_count=*/7, /*alloc_timestamp=*/1000,
                    /*dealloc_timestamp=*/2000, /*alloc_cpu=*/3,
                    /*dealloc_cpu=*/4, /*Histogram=*/0, /*HistogramSize=*/0);

  const auto Schema = llvm::memprof::getFullSchema();
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

// Version0 and Version1 serialize IndexedMemProfRecord in the same format, so
// we share one test.
TEST(MemProf, RecordSerializationRoundTripVersion0And1) {
  const auto Schema = llvm::memprof::getFullSchema();

  MemInfoBlock Info(/*size=*/16, /*access_count=*/7, /*alloc_timestamp=*/1000,
                    /*dealloc_timestamp=*/2000, /*alloc_cpu=*/3,
                    /*dealloc_cpu=*/4, /*Histogram=*/0, /*HistogramSize=*/0);

  llvm::SmallVector<llvm::SmallVector<FrameId>> AllocCallStacks = {
      {0x123, 0x345}, {0x123, 0x567}};

  llvm::SmallVector<llvm::SmallVector<FrameId>> CallSites = {{0x333, 0x777}};

  IndexedMemProfRecord Record;
  for (const auto &ACS : AllocCallStacks) {
    // Use the same info block for both allocation sites.
    Record.AllocSites.emplace_back(ACS, llvm::memprof::hashCallStack(ACS),
                                   Info);
  }
  Record.CallSites.assign(CallSites);
  for (const auto &CS : CallSites)
    Record.CallSiteIds.push_back(llvm::memprof::hashCallStack(CS));

  std::string Buffer;
  llvm::raw_string_ostream OS(Buffer);
  Record.serialize(Schema, OS, llvm::memprof::Version0);

  const IndexedMemProfRecord GotRecord = IndexedMemProfRecord::deserialize(
      Schema, reinterpret_cast<const unsigned char *>(Buffer.data()),
      llvm::memprof::Version0);

  EXPECT_EQ(Record, GotRecord);
}

TEST(MemProf, RecordSerializationRoundTripVerion2) {
  const auto Schema = llvm::memprof::getFullSchema();

  MemInfoBlock Info(/*size=*/16, /*access_count=*/7, /*alloc_timestamp=*/1000,
                    /*dealloc_timestamp=*/2000, /*alloc_cpu=*/3,
                    /*dealloc_cpu=*/4, /*Histogram=*/0, /*HistogramSize=*/0);

  llvm::SmallVector<llvm::memprof::CallStackId> CallStackIds = {0x123, 0x456};

  llvm::SmallVector<llvm::memprof::CallStackId> CallSiteIds = {0x333, 0x444};

  IndexedMemProfRecord Record;
  for (const auto &CSId : CallStackIds) {
    // Use the same info block for both allocation sites.
    Record.AllocSites.emplace_back(llvm::SmallVector<FrameId>(), CSId, Info);
  }
  Record.CallSiteIds.assign(CallSiteIds);

  std::string Buffer;
  llvm::raw_string_ostream OS(Buffer);
  Record.serialize(Schema, OS, llvm::memprof::Version2);

  const IndexedMemProfRecord GotRecord = IndexedMemProfRecord::deserialize(
      Schema, reinterpret_cast<const unsigned char *>(Buffer.data()),
      llvm::memprof::Version2);

  EXPECT_EQ(Record, GotRecord);
}

TEST(MemProf, RecordSerializationRoundTripVersion2HotColdSchema) {
  const auto Schema = llvm::memprof::getHotColdSchema();

  MemInfoBlock Info;
  Info.AllocCount = 11;
  Info.TotalSize = 22;
  Info.TotalLifetime = 33;
  Info.TotalLifetimeAccessDensity = 44;

  llvm::SmallVector<llvm::memprof::CallStackId> CallStackIds = {0x123, 0x456};

  llvm::SmallVector<llvm::memprof::CallStackId> CallSiteIds = {0x333, 0x444};

  IndexedMemProfRecord Record;
  for (const auto &CSId : CallStackIds) {
    // Use the same info block for both allocation sites.
    Record.AllocSites.emplace_back(llvm::SmallVector<FrameId>(), CSId, Info,
                                   Schema);
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
  ASSERT_THAT(Record.AllocSites, ::SizeIs(2));
  EXPECT_EQ(Record.AllocSites[0].Info.getSchema(), SchemaBitSet);
  EXPECT_EQ(Record.AllocSites[1].Info.getSchema(), SchemaBitSet);

  std::string Buffer;
  llvm::raw_string_ostream OS(Buffer);
  Record.serialize(Schema, OS, llvm::memprof::Version2);

  const IndexedMemProfRecord GotRecord = IndexedMemProfRecord::deserialize(
      Schema, reinterpret_cast<const unsigned char *>(Buffer.data()),
      llvm::memprof::Version2);

  // Verify that Schema comes back correctly after deserialization. Technically,
  // the comparison between Record and GotRecord below includes the comparison
  // of their Schemas, but we'll verify the Schemas on our own.
  ASSERT_THAT(GotRecord.AllocSites, ::SizeIs(2));
  EXPECT_EQ(GotRecord.AllocSites[0].Info.getSchema(), SchemaBitSet);
  EXPECT_EQ(GotRecord.AllocSites[1].Info.getSchema(), SchemaBitSet);

  EXPECT_EQ(Record, GotRecord);
}

TEST(MemProf, SymbolizationFilter) {
  std::unique_ptr<MockSymbolizer> Symbolizer(new MockSymbolizer());

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
  for (const auto &KeyRecordPair : Reader) {
    Records.push_back(KeyRecordPair.second);
  }

  ASSERT_THAT(Records, SizeIs(1));
  ASSERT_THAT(Records[0].AllocSites, SizeIs(1));
  ASSERT_THAT(Records[0].AllocSites[0].CallStack, SizeIs(1));
  EXPECT_THAT(Records[0].AllocSites[0].CallStack[0],
              FrameContains("foo", 5U, 30U, false));
}

TEST(MemProf, BaseMemProfReader) {
  llvm::DenseMap<FrameId, Frame> FrameIdMap;
  Frame F1(/*Hash=*/IndexedMemProfRecord::getGUID("foo"), /*LineOffset=*/20,
           /*Column=*/5, /*IsInlineFrame=*/true);
  Frame F2(/*Hash=*/IndexedMemProfRecord::getGUID("bar"), /*LineOffset=*/10,
           /*Column=*/2, /*IsInlineFrame=*/false);
  FrameIdMap.insert({F1.hash(), F1});
  FrameIdMap.insert({F2.hash(), F2});

  llvm::MapVector<llvm::GlobalValue::GUID, IndexedMemProfRecord> ProfData;
  IndexedMemProfRecord FakeRecord;
  MemInfoBlock Block;
  Block.AllocCount = 1U, Block.TotalAccessDensity = 4,
  Block.TotalLifetime = 200001;
  std::array<FrameId, 2> CallStack{F1.hash(), F2.hash()};
  FakeRecord.AllocSites.emplace_back(
      /*CS=*/CallStack, /*CSId=*/llvm::memprof::hashCallStack(CallStack),
      /*MB=*/Block);
  ProfData.insert({F1.hash(), FakeRecord});

  MemProfReader Reader(FrameIdMap, ProfData);

  llvm::SmallVector<MemProfRecord, 1> Records;
  for (const auto &KeyRecordPair : Reader) {
    Records.push_back(KeyRecordPair.second);
  }

  ASSERT_THAT(Records, SizeIs(1));
  ASSERT_THAT(Records[0].AllocSites, SizeIs(1));
  ASSERT_THAT(Records[0].AllocSites[0].CallStack, SizeIs(2));
  EXPECT_THAT(Records[0].AllocSites[0].CallStack[0],
              FrameContains("foo", 20U, 5U, true));
  EXPECT_THAT(Records[0].AllocSites[0].CallStack[1],
              FrameContains("bar", 10U, 2U, false));
}

TEST(MemProf, BaseMemProfReaderWithCSIdMap) {
  llvm::DenseMap<FrameId, Frame> FrameIdMap;
  Frame F1(/*Hash=*/IndexedMemProfRecord::getGUID("foo"), /*LineOffset=*/20,
           /*Column=*/5, /*IsInlineFrame=*/true);
  Frame F2(/*Hash=*/IndexedMemProfRecord::getGUID("bar"), /*LineOffset=*/10,
           /*Column=*/2, /*IsInlineFrame=*/false);
  FrameIdMap.insert({F1.hash(), F1});
  FrameIdMap.insert({F2.hash(), F2});

  llvm::DenseMap<CallStackId, llvm::SmallVector<FrameId>> CSIdMap;
  llvm::SmallVector<FrameId> CallStack = {F1.hash(), F2.hash()};
  CallStackId CSId = llvm::memprof::hashCallStack(CallStack);
  CSIdMap.insert({CSId, CallStack});

  llvm::MapVector<llvm::GlobalValue::GUID, IndexedMemProfRecord> ProfData;
  IndexedMemProfRecord FakeRecord;
  MemInfoBlock Block;
  Block.AllocCount = 1U, Block.TotalAccessDensity = 4,
  Block.TotalLifetime = 200001;
  FakeRecord.AllocSites.emplace_back(
      /*CS=*/llvm::SmallVector<FrameId>(),
      /*CSId=*/llvm::memprof::hashCallStack(CallStack),
      /*MB=*/Block);
  ProfData.insert({F1.hash(), FakeRecord});

  MemProfReader Reader(FrameIdMap, CSIdMap, ProfData);

  llvm::SmallVector<MemProfRecord, 1> Records;
  for (const auto &KeyRecordPair : Reader) {
    Records.push_back(KeyRecordPair.second);
  }

  ASSERT_THAT(Records, SizeIs(1));
  ASSERT_THAT(Records[0].AllocSites, SizeIs(1));
  ASSERT_THAT(Records[0].AllocSites[0].CallStack, SizeIs(2));
  EXPECT_THAT(Records[0].AllocSites[0].CallStack[0],
              FrameContains("foo", 20U, 5U, true));
  EXPECT_THAT(Records[0].AllocSites[0].CallStack[1],
              FrameContains("bar", 10U, 2U, false));
}

TEST(MemProf, IndexedMemProfRecordToMemProfRecord) {
  // Verify that MemProfRecord can be constructed from IndexedMemProfRecord with
  // CallStackIds only.

  llvm::DenseMap<FrameId, Frame> FrameIdMap;
  Frame F1(1, 0, 0, false);
  Frame F2(2, 0, 0, false);
  Frame F3(3, 0, 0, false);
  Frame F4(4, 0, 0, false);
  FrameIdMap.insert({F1.hash(), F1});
  FrameIdMap.insert({F2.hash(), F2});
  FrameIdMap.insert({F3.hash(), F3});
  FrameIdMap.insert({F4.hash(), F4});

  llvm::DenseMap<CallStackId, llvm::SmallVector<FrameId>> CallStackIdMap;
  llvm::SmallVector<FrameId> CS1 = {F1.hash(), F2.hash()};
  llvm::SmallVector<FrameId> CS2 = {F1.hash(), F3.hash()};
  llvm::SmallVector<FrameId> CS3 = {F2.hash(), F3.hash()};
  llvm::SmallVector<FrameId> CS4 = {F2.hash(), F4.hash()};
  CallStackIdMap.insert({llvm::memprof::hashCallStack(CS1), CS1});
  CallStackIdMap.insert({llvm::memprof::hashCallStack(CS2), CS2});
  CallStackIdMap.insert({llvm::memprof::hashCallStack(CS3), CS3});
  CallStackIdMap.insert({llvm::memprof::hashCallStack(CS4), CS4});

  IndexedMemProfRecord IndexedRecord;
  IndexedAllocationInfo AI;
  AI.CSId = llvm::memprof::hashCallStack(CS1);
  IndexedRecord.AllocSites.push_back(AI);
  AI.CSId = llvm::memprof::hashCallStack(CS2);
  IndexedRecord.AllocSites.push_back(AI);
  IndexedRecord.CallSiteIds.push_back(llvm::memprof::hashCallStack(CS3));
  IndexedRecord.CallSiteIds.push_back(llvm::memprof::hashCallStack(CS4));

  llvm::memprof::FrameIdConverter<decltype(FrameIdMap)> FrameIdConv(FrameIdMap);
  llvm::memprof::CallStackIdConverter<decltype(CallStackIdMap)> CSIdConv(
      CallStackIdMap, FrameIdConv);

  MemProfRecord Record = IndexedRecord.toMemProfRecord(CSIdConv);

  // Make sure that all lookups are successful.
  ASSERT_EQ(FrameIdConv.LastUnmappedId, std::nullopt);
  ASSERT_EQ(CSIdConv.LastUnmappedId, std::nullopt);

  // Verify the contents of Record.
  ASSERT_THAT(Record.AllocSites, SizeIs(2));
  ASSERT_THAT(Record.AllocSites[0].CallStack, SizeIs(2));
  EXPECT_EQ(Record.AllocSites[0].CallStack[0].hash(), F1.hash());
  EXPECT_EQ(Record.AllocSites[0].CallStack[1].hash(), F2.hash());
  ASSERT_THAT(Record.AllocSites[1].CallStack, SizeIs(2));
  EXPECT_EQ(Record.AllocSites[1].CallStack[0].hash(), F1.hash());
  EXPECT_EQ(Record.AllocSites[1].CallStack[1].hash(), F3.hash());
  ASSERT_THAT(Record.CallSites, SizeIs(2));
  ASSERT_THAT(Record.CallSites[0], SizeIs(2));
  EXPECT_EQ(Record.CallSites[0][0].hash(), F2.hash());
  EXPECT_EQ(Record.CallSites[0][1].hash(), F3.hash());
  ASSERT_THAT(Record.CallSites[1], SizeIs(2));
  EXPECT_EQ(Record.CallSites[1][0].hash(), F2.hash());
  EXPECT_EQ(Record.CallSites[1][1].hash(), F4.hash());
}

using FrameIdMapTy =
    llvm::DenseMap<::llvm::memprof::FrameId, ::llvm::memprof::Frame>;
using CallStackIdMapTy =
    llvm::DenseMap<::llvm::memprof::CallStackId,
                   ::llvm::SmallVector<::llvm::memprof::FrameId>>;

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
  llvm::memprof::IndexedAllocationInfo AI({}, 0xdeadbeefU, makePartialMIB(),
                                          llvm::memprof::getHotColdSchema());

  IndexedMemProfRecord IndexedMR;
  IndexedMR.AllocSites.push_back(AI);

  // Create empty maps.
  const FrameIdMapTy IdToFrameMap;
  const CallStackIdMapTy CSIdToCallStackMap;
  llvm::memprof::FrameIdConverter<decltype(IdToFrameMap)> FrameIdConv(
      IdToFrameMap);
  llvm::memprof::CallStackIdConverter<decltype(CSIdToCallStackMap)> CSIdConv(
      CSIdToCallStackMap, FrameIdConv);

  // We are only interested in errors, not the return value.
  (void)IndexedMR.toMemProfRecord(CSIdConv);

  ASSERT_TRUE(CSIdConv.LastUnmappedId.has_value());
  EXPECT_EQ(*CSIdConv.LastUnmappedId, 0xdeadbeefU);
  EXPECT_EQ(FrameIdConv.LastUnmappedId, std::nullopt);
}

TEST(MemProf, MissingFrameId) {
  llvm::memprof::IndexedAllocationInfo AI({}, 0x222, makePartialMIB(),
                                          llvm::memprof::getHotColdSchema());

  IndexedMemProfRecord IndexedMR;
  IndexedMR.AllocSites.push_back(AI);

  // An empty map to trigger a mapping error.
  const FrameIdMapTy IdToFrameMap;
  CallStackIdMapTy CSIdToCallStackMap;
  CSIdToCallStackMap.insert({0x222, {2, 3}});

  llvm::memprof::FrameIdConverter<decltype(IdToFrameMap)> FrameIdConv(
      IdToFrameMap);
  llvm::memprof::CallStackIdConverter<decltype(CSIdToCallStackMap)> CSIdConv(
      CSIdToCallStackMap, FrameIdConv);

  // We are only interested in errors, not the return value.
  (void)IndexedMR.toMemProfRecord(CSIdConv);

  EXPECT_EQ(CSIdConv.LastUnmappedId, std::nullopt);
  ASSERT_TRUE(FrameIdConv.LastUnmappedId.has_value());
  EXPECT_EQ(*FrameIdConv.LastUnmappedId, 3U);
}

// Verify CallStackRadixTreeBuilder can handle empty inputs.
TEST(MemProf, RadixTreeBuilderEmpty) {
  llvm::DenseMap<FrameId, llvm::memprof::LinearFrameId> MemProfFrameIndexes;
  llvm::MapVector<CallStackId, llvm::SmallVector<FrameId>> MemProfCallStackData;
  llvm::DenseMap<llvm::memprof::FrameId, llvm::memprof::FrameStat>
      FrameHistogram =
          llvm::memprof::computeFrameHistogram(MemProfCallStackData);
  llvm::memprof::CallStackRadixTreeBuilder Builder;
  Builder.build(std::move(MemProfCallStackData), MemProfFrameIndexes,
                FrameHistogram);
  ASSERT_THAT(Builder.getRadixArray(), testing::IsEmpty());
  const auto Mappings = Builder.takeCallStackPos();
  ASSERT_THAT(Mappings, testing::IsEmpty());
}

// Verify CallStackRadixTreeBuilder can handle one trivial call stack.
TEST(MemProf, RadixTreeBuilderOne) {
  llvm::DenseMap<FrameId, llvm::memprof::LinearFrameId> MemProfFrameIndexes = {
      {11, 1}, {12, 2}, {13, 3}};
  llvm::SmallVector<llvm::memprof::FrameId> CS1 = {13, 12, 11};
  llvm::MapVector<CallStackId, llvm::SmallVector<FrameId>> MemProfCallStackData;
  MemProfCallStackData.insert({llvm::memprof::hashCallStack(CS1), CS1});
  llvm::DenseMap<llvm::memprof::FrameId, llvm::memprof::FrameStat>
      FrameHistogram =
          llvm::memprof::computeFrameHistogram(MemProfCallStackData);
  llvm::memprof::CallStackRadixTreeBuilder Builder;
  Builder.build(std::move(MemProfCallStackData), MemProfFrameIndexes,
                FrameHistogram);
  EXPECT_THAT(Builder.getRadixArray(), testing::ElementsAreArray({
                                           3U, // Size of CS1,
                                           3U, // MemProfFrameIndexes[13]
                                           2U, // MemProfFrameIndexes[12]
                                           1U  // MemProfFrameIndexes[11]
                                       }));
  const auto Mappings = Builder.takeCallStackPos();
  ASSERT_THAT(Mappings, SizeIs(1));
  EXPECT_THAT(Mappings, testing::Contains(testing::Pair(
                            llvm::memprof::hashCallStack(CS1), 0U)));
}

// Verify CallStackRadixTreeBuilder can form a link between two call stacks.
TEST(MemProf, RadixTreeBuilderTwo) {
  llvm::DenseMap<FrameId, llvm::memprof::LinearFrameId> MemProfFrameIndexes = {
      {11, 1}, {12, 2}, {13, 3}};
  llvm::SmallVector<llvm::memprof::FrameId> CS1 = {12, 11};
  llvm::SmallVector<llvm::memprof::FrameId> CS2 = {13, 12, 11};
  llvm::MapVector<CallStackId, llvm::SmallVector<FrameId>> MemProfCallStackData;
  MemProfCallStackData.insert({llvm::memprof::hashCallStack(CS1), CS1});
  MemProfCallStackData.insert({llvm::memprof::hashCallStack(CS2), CS2});
  llvm::DenseMap<llvm::memprof::FrameId, llvm::memprof::FrameStat>
      FrameHistogram =
          llvm::memprof::computeFrameHistogram(MemProfCallStackData);
  llvm::memprof::CallStackRadixTreeBuilder Builder;
  Builder.build(std::move(MemProfCallStackData), MemProfFrameIndexes,
                FrameHistogram);
  EXPECT_THAT(Builder.getRadixArray(),
              testing::ElementsAreArray({
                  2U,                        // Size of CS1
                  static_cast<uint32_t>(-3), // Jump 3 steps
                  3U,                        // Size of CS2
                  3U,                        // MemProfFrameIndexes[13]
                  2U,                        // MemProfFrameIndexes[12]
                  1U                         // MemProfFrameIndexes[11]
              }));
  const auto Mappings = Builder.takeCallStackPos();
  ASSERT_THAT(Mappings, SizeIs(2));
  EXPECT_THAT(Mappings, testing::Contains(testing::Pair(
                            llvm::memprof::hashCallStack(CS1), 0U)));
  EXPECT_THAT(Mappings, testing::Contains(testing::Pair(
                            llvm::memprof::hashCallStack(CS2), 2U)));
}

// Verify CallStackRadixTreeBuilder can form a jump to a prefix that itself has
// another jump to another prefix.
TEST(MemProf, RadixTreeBuilderSuccessiveJumps) {
  llvm::DenseMap<FrameId, llvm::memprof::LinearFrameId> MemProfFrameIndexes = {
      {11, 1}, {12, 2}, {13, 3}, {14, 4}, {15, 5}, {16, 6}, {17, 7}, {18, 8},
  };
  llvm::SmallVector<llvm::memprof::FrameId> CS1 = {14, 13, 12, 11};
  llvm::SmallVector<llvm::memprof::FrameId> CS2 = {15, 13, 12, 11};
  llvm::SmallVector<llvm::memprof::FrameId> CS3 = {17, 16, 12, 11};
  llvm::SmallVector<llvm::memprof::FrameId> CS4 = {18, 16, 12, 11};
  llvm::MapVector<CallStackId, llvm::SmallVector<FrameId>> MemProfCallStackData;
  MemProfCallStackData.insert({llvm::memprof::hashCallStack(CS1), CS1});
  MemProfCallStackData.insert({llvm::memprof::hashCallStack(CS2), CS2});
  MemProfCallStackData.insert({llvm::memprof::hashCallStack(CS3), CS3});
  MemProfCallStackData.insert({llvm::memprof::hashCallStack(CS4), CS4});
  llvm::DenseMap<llvm::memprof::FrameId, llvm::memprof::FrameStat>
      FrameHistogram =
          llvm::memprof::computeFrameHistogram(MemProfCallStackData);
  llvm::memprof::CallStackRadixTreeBuilder Builder;
  Builder.build(std::move(MemProfCallStackData), MemProfFrameIndexes,
                FrameHistogram);
  EXPECT_THAT(Builder.getRadixArray(),
              testing::ElementsAreArray({
                  4U,                        // Size of CS1
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
              }));
  const auto Mappings = Builder.takeCallStackPos();
  ASSERT_THAT(Mappings, SizeIs(4));
  EXPECT_THAT(Mappings, testing::Contains(testing::Pair(
                            llvm::memprof::hashCallStack(CS1), 0U)));
  EXPECT_THAT(Mappings, testing::Contains(testing::Pair(
                            llvm::memprof::hashCallStack(CS2), 3U)));
  EXPECT_THAT(Mappings, testing::Contains(testing::Pair(
                            llvm::memprof::hashCallStack(CS3), 7U)));
  EXPECT_THAT(Mappings, testing::Contains(testing::Pair(
                            llvm::memprof::hashCallStack(CS4), 10U)));
}
} // namespace
