#include "llvm/ProfileData/MemProf.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/Symbolize/SymbolizableModule.h"
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
};
DIInliningInfo makeInliningInfo(std::initializer_list<MockInfo> MockFrames) {
  DIInliningInfo Result;
  for (const auto &Item : MockFrames) {
    DILineInfo Frame;
    Frame.FunctionName = Item.FunctionName;
    Frame.Line = Item.Line;
    Frame.StartLine = Item.StartLine;
    Frame.Column = Item.Column;
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
  const uint64_t ExpectedHash = llvm::Function::getGUID(FunctionName);
  if (arg.Function != ExpectedHash) {
    *result_listener << "Hash mismatch";
    return false;
  }
  if (arg.LineOffset == LineOffset && arg.Column == Column &&
      arg.IsInlineFrame == Inline) {
    return true;
  }
  *result_listener << "LineOffset, Column or Inline mismatch";
  return false;
}

MATCHER_P(EqualsRecord, Want, "") {
  if (arg == Want)
    return true;

  std::string Explanation;
  llvm::raw_string_ostream OS(Explanation);
  OS << "\n Want: \n";
  Want.print(OS);
  OS << "\n Got: \n";
  arg.print(OS);
  OS.flush();

  *result_listener << Explanation;
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

  EXPECT_CALL(*Symbolizer, symbolizeInlinedCode(SectionedAddress{0x2000},
                                                specifier(), false))
      .Times(1) // Only once since we cache the result for future lookups.
      .WillRepeatedly(Return(makeInliningInfo({
          {"foo", 10, 5, 30},
          {"bar", 201, 150, 20},
      })));

  EXPECT_CALL(*Symbolizer, symbolizeInlinedCode(SectionedAddress{0x6000},
                                                specifier(), false))
      .Times(1)
      .WillRepeatedly(Return(makeInliningInfo({
          {"baz", 10, 5, 30},
          {"qux.llvm.12345", 75, 70, 10},
      })));

  CallStackMap CSM;
  CSM[0x1] = {0x2000};
  CSM[0x2] = {0x6000, 0x2000};

  llvm::MapVector<uint64_t, MemInfoBlock> Prof;
  Prof[0x1].AllocCount = 1;
  Prof[0x2].AllocCount = 2;

  auto Seg = makeSegments();

  RawMemProfReader Reader(std::move(Symbolizer), Seg, Prof, CSM);

  std::vector<MemProfRecord> Records;
  for (const MemProfRecord &R : Reader) {
    Records.push_back(R);
  }
  EXPECT_EQ(Records.size(), 2U);

  EXPECT_EQ(Records[0].Info.getAllocCount(), 1U);
  EXPECT_EQ(Records[1].Info.getAllocCount(), 2U);
  EXPECT_THAT(Records[0].CallStack[0], FrameContains("foo", 5U, 30U, false));
  EXPECT_THAT(Records[0].CallStack[1], FrameContains("bar", 51U, 20U, true));

  EXPECT_THAT(Records[1].CallStack[0], FrameContains("baz", 5U, 30U, false));
  EXPECT_THAT(Records[1].CallStack[1], FrameContains("qux", 5U, 10U, true));
  EXPECT_THAT(Records[1].CallStack[2], FrameContains("foo", 5U, 30U, false));
  EXPECT_THAT(Records[1].CallStack[3], FrameContains("bar", 51U, 20U, true));
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

  llvm::SmallVector<MemProfRecord, 3> Records;
  MemProfRecord MR;

  MemInfoBlock Info(/*size=*/16, /*access_count=*/7, /*alloc_timestamp=*/1000,
                    /*dealloc_timestamp=*/2000, /*alloc_cpu=*/3,
                    /*dealloc_cpu=*/4);

  MR.Info = PortableMemInfoBlock(Info);
  MR.CallStack.push_back({0x123, 1, 2, false});
  MR.CallStack.push_back({0x345, 3, 4, false});
  Records.push_back(MR);

  MR.clear();
  MR.Info = PortableMemInfoBlock(Info);
  MR.CallStack.push_back({0x567, 5, 6, false});
  MR.CallStack.push_back({0x789, 7, 8, false});
  Records.push_back(MR);

  std::string Buffer;
  llvm::raw_string_ostream OS(Buffer);
  serializeRecords(Records, Schema, OS);
  OS.flush();

  const llvm::SmallVector<MemProfRecord, 4> GotRecords = deserializeRecords(
      Schema, reinterpret_cast<const unsigned char *>(Buffer.data()));

  ASSERT_TRUE(!GotRecords.empty());
  EXPECT_EQ(GotRecords.size(), Records.size());
  EXPECT_THAT(GotRecords[0], EqualsRecord(Records[0]));
  EXPECT_THAT(GotRecords[1], EqualsRecord(Records[1]));
}
} // namespace
