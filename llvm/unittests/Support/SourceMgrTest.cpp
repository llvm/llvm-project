//===- unittests/Support/SourceMgrTest.cpp - SourceMgr tests --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class SourceMgrTest : public testing::Test {
public:
  SourceMgr SM;
  unsigned MainBufferID;
  std::string Output;

  void setMainBuffer(StringRef Text, StringRef BufferName) {
    std::unique_ptr<MemoryBuffer> MainBuffer =
        MemoryBuffer::getMemBuffer(Text, BufferName);
    MainBufferID = SM.AddNewSourceBuffer(std::move(MainBuffer), llvm::SMLoc());
  }

  SMLoc getLoc(unsigned Offset) {
    return SMLoc::getFromPointer(
        SM.getMemoryBuffer(MainBufferID)->getBufferStart() + Offset);
  }

  SMRange getRange(unsigned Offset, unsigned Length) {
    return SMRange(getLoc(Offset), getLoc(Offset + Length));
  }

  void printMessage(SMLoc Loc, SourceMgr::DiagKind Kind,
                    const Twine &Msg, ArrayRef<SMRange> Ranges,
                    ArrayRef<SMFixIt> FixIts) {
    raw_string_ostream OS(Output);
    SM.PrintMessage(OS, Loc, Kind, Msg, Ranges, FixIts);
  }
};

} // unnamed namespace

TEST_F(SourceMgrTest, BasicError) {
  setMainBuffer("aaa bbb\nccc ddd\n", "file.in");
  printMessage(getLoc(4), SourceMgr::DK_Error, "message", None, None);

  EXPECT_EQ("file.in:1:5: error: message\n"
            "aaa bbb\n"
            "    ^\n",
            Output);
}

TEST_F(SourceMgrTest, BasicWarning) {
  setMainBuffer("aaa bbb\nccc ddd\n", "file.in");
  printMessage(getLoc(4), SourceMgr::DK_Warning, "message", None, None);

  EXPECT_EQ("file.in:1:5: warning: message\n"
            "aaa bbb\n"
            "    ^\n",
            Output);
}

TEST_F(SourceMgrTest, BasicRemark) {
  setMainBuffer("aaa bbb\nccc ddd\n", "file.in");
  printMessage(getLoc(4), SourceMgr::DK_Remark, "message", None, None);

  EXPECT_EQ("file.in:1:5: remark: message\n"
            "aaa bbb\n"
            "    ^\n",
            Output);
}

TEST_F(SourceMgrTest, BasicNote) {
  setMainBuffer("aaa bbb\nccc ddd\n", "file.in");
  printMessage(getLoc(4), SourceMgr::DK_Note, "message", None, None);

  EXPECT_EQ("file.in:1:5: note: message\n"
            "aaa bbb\n"
            "    ^\n",
            Output);
}

TEST_F(SourceMgrTest, LocationAtEndOfLine) {
  setMainBuffer("aaa bbb\nccc ddd\n", "file.in");
  printMessage(getLoc(6), SourceMgr::DK_Error, "message", None, None);

  EXPECT_EQ("file.in:1:7: error: message\n"
            "aaa bbb\n"
            "      ^\n",
            Output);
}

TEST_F(SourceMgrTest, LocationAtNewline) {
  setMainBuffer("aaa bbb\nccc ddd\n", "file.in");
  printMessage(getLoc(7), SourceMgr::DK_Error, "message", None, None);

  EXPECT_EQ("file.in:1:8: error: message\n"
            "aaa bbb\n"
            "       ^\n",
            Output);
}

TEST_F(SourceMgrTest, BasicRange) {
  setMainBuffer("aaa bbb\nccc ddd\n", "file.in");
  printMessage(getLoc(4), SourceMgr::DK_Error, "message", getRange(4, 3), None);

  EXPECT_EQ("file.in:1:5: error: message\n"
            "aaa bbb\n"
            "    ^~~\n",
            Output);
}

TEST_F(SourceMgrTest, RangeWithTab) {
  setMainBuffer("aaa\tbbb\nccc ddd\n", "file.in");
  printMessage(getLoc(4), SourceMgr::DK_Error, "message", getRange(3, 3), None);

  EXPECT_EQ("file.in:1:5: error: message\n"
            "aaa     bbb\n"
            "   ~~~~~^~\n",
            Output);
}

TEST_F(SourceMgrTest, MultiLineRange) {
  setMainBuffer("aaa bbb\nccc ddd\n", "file.in");
  printMessage(getLoc(4), SourceMgr::DK_Error, "message", getRange(4, 7), None);

  EXPECT_EQ("file.in:1:5: error: message\n"
            "aaa bbb\n"
            "    ^~~\n",
            Output);
}

TEST_F(SourceMgrTest, MultipleRanges) {
  setMainBuffer("aaa bbb\nccc ddd\n", "file.in");
  SMRange Ranges[] = { getRange(0, 3), getRange(4, 3) };
  printMessage(getLoc(4), SourceMgr::DK_Error, "message", Ranges, None);

  EXPECT_EQ("file.in:1:5: error: message\n"
            "aaa bbb\n"
            "~~~ ^~~\n",
            Output);
}

TEST_F(SourceMgrTest, OverlappingRanges) {
  setMainBuffer("aaa bbb\nccc ddd\n", "file.in");
  SMRange Ranges[] = { getRange(0, 3), getRange(2, 4) };
  printMessage(getLoc(4), SourceMgr::DK_Error, "message", Ranges, None);

  EXPECT_EQ("file.in:1:5: error: message\n"
            "aaa bbb\n"
            "~~~~^~\n",
            Output);
}

TEST_F(SourceMgrTest, BasicFixit) {
  setMainBuffer("aaa bbb\nccc ddd\n", "file.in");
  printMessage(getLoc(4), SourceMgr::DK_Error, "message", None,
               makeArrayRef(SMFixIt(getRange(4, 3), "zzz")));

  EXPECT_EQ("file.in:1:5: error: message\n"
            "aaa bbb\n"
            "    ^~~\n"
            "    zzz\n",
            Output);
}

TEST_F(SourceMgrTest, FixitForTab) {
  setMainBuffer("aaa\tbbb\nccc ddd\n", "file.in");
  printMessage(getLoc(3), SourceMgr::DK_Error, "message", None,
               makeArrayRef(SMFixIt(getRange(3, 1), "zzz")));

  EXPECT_EQ("file.in:1:4: error: message\n"
            "aaa     bbb\n"
            "   ^^^^^\n"
            "   zzz\n",
            Output);
}

