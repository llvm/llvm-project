#include "llvm/ProfileData/MemProf.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/Symbolize/SymbolizableModule.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Value.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/ProfileData/MemProfData.inc"
#include "llvm/ProfileData/RawMemProfReader.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MD5.h"
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
using ::llvm::memprof::CallStackMap;
using ::llvm::memprof::Frame;
using ::llvm::memprof::FrameId;
using ::llvm::memprof::IndexedMemProfRecord;
using ::llvm::memprof::MemInfoBlock;
using ::llvm::memprof::MemProfRecord;
using ::llvm::memprof::MemProfSchema;
using ::llvm::memprof::Meta;
using ::llvm::memprof::PortableMemInfoBlock;
using ::llvm::memprof::RawMemProfReader;
using ::llvm::memprof::SegmentEntry;
using ::llvm::object::SectionedAddress;
using ::llvm::symbolize::SymbolizableModule;
using ::testing::Return;

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

  const uint64_t ExpectedHash = llvm::Function::getGUID(FunctionName);
  if (F.Function != ExpectedHash) {
    *result_listener << "Hash mismatch";
    return false;
  }
  if (F.SymbolName.hasValue() && F.SymbolName.getValue() != FunctionName) {
    *result_listener << "SymbolName mismatch\nWant: " << FunctionName
                     << "\nGot: " << F.SymbolName.getValue();
    return false;
  }
  if (F.LineOffset == LineOffset && F.Column == Column &&
      F.IsInlineFrame == Inline) {
    return true;
  }
  *result_listener << "LineOffset, Column or Inline mismatch";
  return false;
}

MemProfSchema getFullSchema() {
  MemProfSchema Schema;
#define MIBEntryDef(NameTag, Name, Type) Schema.push_back(Meta::Name);
#include "llvm/ProfileData/MIBEntryDef.inc"
#undef MIBEntryDef
  return Schema;
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
          {"xyz", 10, 5, 30},
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

  // Mock program psuedocode and expected memprof record contents.
  //
  //                              AllocSite       CallSite
  // inline foo() { new(); }         Y               N
  // bar() { foo(); }                Y               Y
  // inline xyz() { bar(); }         N               Y
  // abc() { xyz(); }                N               Y

  // We expect 4 records. We attach alloc site data to foo and bar, i.e.
  // all frames bottom up until we find a non-inline frame. We attach call site
  // data to bar, xyz and abc.
  ASSERT_EQ(Records.size(), 4U);

  // Check the memprof record for foo.
  const llvm::GlobalValue::GUID FooId = IndexedMemProfRecord::getGUID("foo");
  ASSERT_EQ(Records.count(FooId), 1U);
  const MemProfRecord &Foo = Records[FooId];
  ASSERT_EQ(Foo.AllocSites.size(), 1U);
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
  ASSERT_EQ(Records.count(BarId), 1U);
  const MemProfRecord &Bar = Records[BarId];
  ASSERT_EQ(Bar.AllocSites.size(), 1U);
  EXPECT_EQ(Bar.AllocSites[0].Info.getAllocCount(), 1U);
  EXPECT_THAT(Bar.AllocSites[0].CallStack[0],
              FrameContains("foo", 5U, 30U, true));
  EXPECT_THAT(Bar.AllocSites[0].CallStack[1],
              FrameContains("bar", 51U, 20U, false));
  EXPECT_THAT(Bar.AllocSites[0].CallStack[2],
              FrameContains("xyz", 5U, 30U, true));
  EXPECT_THAT(Bar.AllocSites[0].CallStack[3],
              FrameContains("abc", 5U, 30U, false));

  ASSERT_EQ(Bar.CallSites.size(), 1U);
  ASSERT_EQ(Bar.CallSites[0].size(), 2U);
  EXPECT_THAT(Bar.CallSites[0][0], FrameContains("foo", 5U, 30U, true));
  EXPECT_THAT(Bar.CallSites[0][1], FrameContains("bar", 51U, 20U, false));

  // Check the memprof record for xyz.
  const llvm::GlobalValue::GUID XyzId = IndexedMemProfRecord::getGUID("xyz");
  ASSERT_EQ(Records.count(XyzId), 1U);
  const MemProfRecord &Xyz = Records[XyzId];
  ASSERT_EQ(Xyz.CallSites.size(), 1U);
  ASSERT_EQ(Xyz.CallSites[0].size(), 2U);
  // Expect the entire frame even though in practice we only need the first
  // entry here.
  EXPECT_THAT(Xyz.CallSites[0][0], FrameContains("xyz", 5U, 30U, true));
  EXPECT_THAT(Xyz.CallSites[0][1], FrameContains("abc", 5U, 30U, false));

  // Check the memprof record for abc.
  const llvm::GlobalValue::GUID AbcId = IndexedMemProfRecord::getGUID("abc");
  ASSERT_EQ(Records.count(AbcId), 1U);
  const MemProfRecord &Abc = Records[AbcId];
  EXPECT_TRUE(Abc.AllocSites.empty());
  ASSERT_EQ(Abc.CallSites.size(), 1U);
  ASSERT_EQ(Abc.CallSites[0].size(), 2U);
  EXPECT_THAT(Abc.CallSites[0][0], FrameContains("xyz", 5U, 30U, true));
  EXPECT_THAT(Abc.CallSites[0][1], FrameContains("abc", 5U, 30U, false));
}

TEST(MemProf, PortableWrapper) {
  MemInfoBlock Info(/*size=*/16, /*access_count=*/7, /*alloc_timestamp=*/1000,
                    /*dealloc_timestamp=*/2000, /*alloc_cpu=*/3,
                    /*dealloc_cpu=*/4);

  const auto Schema = getFullSchema();
  PortableMemInfoBlock WriteBlock(Info);

  std::string Buffer;
  llvm::raw_string_ostream OS(Buffer);
  WriteBlock.serialize(Schema, OS);
  OS.flush();

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

TEST(MemProf, RecordSerializationRoundTrip) {
  const MemProfSchema Schema = getFullSchema();

  MemInfoBlock Info(/*size=*/16, /*access_count=*/7, /*alloc_timestamp=*/1000,
                    /*dealloc_timestamp=*/2000, /*alloc_cpu=*/3,
                    /*dealloc_cpu=*/4);

  llvm::SmallVector<llvm::SmallVector<FrameId>> AllocCallStacks = {
      {0x123, 0x345}, {0x123, 0x567}};

  llvm::SmallVector<llvm::SmallVector<FrameId>> CallSites = {{0x333, 0x777}};

  IndexedMemProfRecord Record;
  for (const auto &ACS : AllocCallStacks) {
    // Use the same info block for both allocation sites.
    Record.AllocSites.emplace_back(ACS, Info);
  }
  Record.CallSites.assign(CallSites);

  std::string Buffer;
  llvm::raw_string_ostream OS(Buffer);
  Record.serialize(Schema, OS);
  OS.flush();

  const IndexedMemProfRecord GotRecord = IndexedMemProfRecord::deserialize(
      Schema, reinterpret_cast<const unsigned char *>(Buffer.data()));

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
          {"foo", 10, 5, 30},
      })));

  CallStackMap CSM;
  CSM[0x1] = {0x1000, 0x2000, 0x3000, 0x4000};
  // This entry should be dropped since all PCs are either not
  // symbolizable or belong to the runtime.
  CSM[0x2] = {0x1000, 0x2000};

  llvm::MapVector<uint64_t, MemInfoBlock> Prof;
  Prof[0x1].AllocCount = 1;
  Prof[0x2].AllocCount = 1;

  auto Seg = makeSegments();

  RawMemProfReader Reader(std::move(Symbolizer), Seg, Prof, CSM);

  llvm::SmallVector<MemProfRecord, 1> Records;
  for (const auto &KeyRecordPair : Reader) {
    Records.push_back(KeyRecordPair.second);
  }

  ASSERT_EQ(Records.size(), 1U);
  ASSERT_EQ(Records[0].AllocSites.size(), 1U);
  ASSERT_EQ(Records[0].AllocSites[0].CallStack.size(), 1U);
  EXPECT_THAT(Records[0].AllocSites[0].CallStack[0],
              FrameContains("foo", 5U, 30U, false));
}
} // namespace
