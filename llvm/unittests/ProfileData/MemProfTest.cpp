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

MATCHER_P4(FrameContains, Function, LineOffset, Column, Inline, "") {
  const std::string ExpectedHash = std::to_string(llvm::MD5Hash(Function));
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

TEST(MemProf, FillsValue) {
  std::unique_ptr<MockSymbolizer> Symbolizer(new MockSymbolizer());

  EXPECT_CALL(*Symbolizer, symbolizeInlinedCode(SectionedAddress{0x2000},
                                                specifier(), false))
      .Times(2)
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
  Prof[0x1].alloc_count = 1;
  Prof[0x2].alloc_count = 2;

  auto Seg = makeSegments();

  RawMemProfReader Reader(std::move(Symbolizer), Seg, Prof, CSM);

  std::vector<MemProfRecord> Records;
  for (const MemProfRecord &R : Reader) {
    Records.push_back(R);
  }
  EXPECT_EQ(Records.size(), 2U);

  EXPECT_EQ(Records[0].Info.alloc_count, 1U);
  EXPECT_EQ(Records[1].Info.alloc_count, 2U);
  EXPECT_THAT(Records[0].CallStack[0], FrameContains("foo", 5U, 30U, false));
  EXPECT_THAT(Records[0].CallStack[1], FrameContains("bar", 51U, 20U, true));

  EXPECT_THAT(Records[1].CallStack[0], FrameContains("baz", 5U, 30U, false));
  EXPECT_THAT(Records[1].CallStack[1], FrameContains("qux", 5U, 10U, true));
  EXPECT_THAT(Records[1].CallStack[2], FrameContains("foo", 5U, 30U, false));
  EXPECT_THAT(Records[1].CallStack[3], FrameContains("bar", 51U, 20U, true));
}

} // namespace
