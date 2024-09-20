//===- unittests/Support/SourceMgrTest.cpp - SourceMgr tests --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
  printMessage(getLoc(4), SourceMgr::DK_Error, "message", std::nullopt,
               std::nullopt);

  EXPECT_EQ("file.in:1:5: error: message\n"
            "aaa bbb\n"
            "    ^\n",
            Output);
}

TEST_F(SourceMgrTest, BasicWarning) {
  setMainBuffer("aaa bbb\nccc ddd\n", "file.in");
  printMessage(getLoc(4), SourceMgr::DK_Warning, "message", std::nullopt,
               std::nullopt);

  EXPECT_EQ("file.in:1:5: warning: message\n"
            "aaa bbb\n"
            "    ^\n",
            Output);
}

TEST_F(SourceMgrTest, BasicRemark) {
  setMainBuffer("aaa bbb\nccc ddd\n", "file.in");
  printMessage(getLoc(4), SourceMgr::DK_Remark, "message", std::nullopt,
               std::nullopt);

  EXPECT_EQ("file.in:1:5: remark: message\n"
            "aaa bbb\n"
            "    ^\n",
            Output);
}

TEST_F(SourceMgrTest, BasicNote) {
  setMainBuffer("aaa bbb\nccc ddd\n", "file.in");
  printMessage(getLoc(4), SourceMgr::DK_Note, "message", std::nullopt,
               std::nullopt);

  EXPECT_EQ("file.in:1:5: note: message\n"
            "aaa bbb\n"
            "    ^\n",
            Output);
}

TEST_F(SourceMgrTest, LocationAtEndOfLine) {
  setMainBuffer("aaa bbb\nccc ddd\n", "file.in");
  printMessage(getLoc(6), SourceMgr::DK_Error, "message", std::nullopt,
               std::nullopt);

  EXPECT_EQ("file.in:1:7: error: message\n"
            "aaa bbb\n"
            "      ^\n",
            Output);
}

TEST_F(SourceMgrTest, LocationAtNewline) {
  setMainBuffer("aaa bbb\nccc ddd\n", "file.in");
  printMessage(getLoc(7), SourceMgr::DK_Error, "message", std::nullopt,
               std::nullopt);

  EXPECT_EQ("file.in:1:8: error: message\n"
            "aaa bbb\n"
            "       ^\n",
            Output);
}

TEST_F(SourceMgrTest, LocationAtEmptyBuffer) {
  setMainBuffer("", "file.in");
  printMessage(getLoc(0), SourceMgr::DK_Error, "message", std::nullopt,
               std::nullopt);

  EXPECT_EQ("file.in:1:1: error: message\n"
            "\n"
            "^\n",
            Output);
}

TEST_F(SourceMgrTest, LocationJustOnSoleNewline) {
  setMainBuffer("\n", "file.in");
  printMessage(getLoc(0), SourceMgr::DK_Error, "message", std::nullopt,
               std::nullopt);

  EXPECT_EQ("file.in:1:1: error: message\n"
            "\n"
            "^\n",
            Output);
}

TEST_F(SourceMgrTest, LocationJustAfterSoleNewline) {
  setMainBuffer("\n", "file.in");
  printMessage(getLoc(1), SourceMgr::DK_Error, "message", std::nullopt,
               std::nullopt);

  EXPECT_EQ("file.in:2:1: error: message\n"
            "\n"
            "^\n",
            Output);
}

TEST_F(SourceMgrTest, LocationJustAfterNonNewline) {
  setMainBuffer("123", "file.in");
  printMessage(getLoc(3), SourceMgr::DK_Error, "message", std::nullopt,
               std::nullopt);

  EXPECT_EQ("file.in:1:4: error: message\n"
            "123\n"
            "   ^\n",
            Output);
}

TEST_F(SourceMgrTest, LocationOnFirstLineOfMultiline) {
  setMainBuffer("1234\n6789\n", "file.in");
  printMessage(getLoc(3), SourceMgr::DK_Error, "message", std::nullopt,
               std::nullopt);

  EXPECT_EQ("file.in:1:4: error: message\n"
            "1234\n"
            "   ^\n",
            Output);
}

TEST_F(SourceMgrTest, LocationOnEOLOfFirstLineOfMultiline) {
  setMainBuffer("1234\n6789\n", "file.in");
  printMessage(getLoc(4), SourceMgr::DK_Error, "message", std::nullopt,
               std::nullopt);

  EXPECT_EQ("file.in:1:5: error: message\n"
            "1234\n"
            "    ^\n",
            Output);
}

TEST_F(SourceMgrTest, LocationOnSecondLineOfMultiline) {
  setMainBuffer("1234\n6789\n", "file.in");
  printMessage(getLoc(5), SourceMgr::DK_Error, "message", std::nullopt,
               std::nullopt);

  EXPECT_EQ("file.in:2:1: error: message\n"
            "6789\n"
            "^\n",
            Output);
}

TEST_F(SourceMgrTest, LocationOnSecondLineOfMultilineNoSecondEOL) {
  setMainBuffer("1234\n6789", "file.in");
  printMessage(getLoc(5), SourceMgr::DK_Error, "message", std::nullopt,
               std::nullopt);

  EXPECT_EQ("file.in:2:1: error: message\n"
            "6789\n"
            "^\n",
            Output);
}

TEST_F(SourceMgrTest, LocationOnEOLOfSecondSecondLineOfMultiline) {
  setMainBuffer("1234\n6789\n", "file.in");
  printMessage(getLoc(9), SourceMgr::DK_Error, "message", std::nullopt,
               std::nullopt);

  EXPECT_EQ("file.in:2:5: error: message\n"
            "6789\n"
            "    ^\n",
            Output);
}

#define STRING_LITERAL_253_BYTES \
  "1234567890\n1234567890\n" \
  "1234567890\n1234567890\n" \
  "1234567890\n1234567890\n" \
  "1234567890\n1234567890\n" \
  "1234567890\n1234567890\n" \
  "1234567890\n1234567890\n" \
  "1234567890\n1234567890\n" \
  "1234567890\n1234567890\n" \
  "1234567890\n1234567890\n" \
  "1234567890\n1234567890\n" \
  "1234567890\n1234567890\n" \
  "1234567890\n"

//===----------------------------------------------------------------------===//
// 255-byte buffer tests
//===----------------------------------------------------------------------===//

TEST_F(SourceMgrTest, LocationBeforeEndOf255ByteBuffer) {
  setMainBuffer(STRING_LITERAL_253_BYTES   // first 253 bytes
                "12"                       // + 2 = 255 bytes
                , "file.in");
  printMessage(getLoc(253), SourceMgr::DK_Error, "message", std::nullopt,
               std::nullopt);
  EXPECT_EQ("file.in:24:1: error: message\n"
            "12\n"
            "^\n",
            Output);
}

TEST_F(SourceMgrTest, LocationAtEndOf255ByteBuffer) {
  setMainBuffer(STRING_LITERAL_253_BYTES   // first 253 bytes
                "12"                       // + 2 = 255 bytes
                , "file.in");
  printMessage(getLoc(254), SourceMgr::DK_Error, "message", std::nullopt,
               std::nullopt);
  EXPECT_EQ("file.in:24:2: error: message\n"
            "12\n"
            " ^\n",
            Output);
}

TEST_F(SourceMgrTest, LocationPastEndOf255ByteBuffer) {
  setMainBuffer(STRING_LITERAL_253_BYTES   // first 253 bytes
                "12"                       // + 2 = 255 bytes
                , "file.in");
  printMessage(getLoc(255), SourceMgr::DK_Error, "message", std::nullopt,
               std::nullopt);
  EXPECT_EQ("file.in:24:3: error: message\n"
            "12\n"
            "  ^\n",
            Output);
}

TEST_F(SourceMgrTest, LocationBeforeEndOf255ByteBufferEndingInNewline) {
  setMainBuffer(STRING_LITERAL_253_BYTES   // first 253 bytes
                "1\n"                      // + 2 = 255 bytes
                , "file.in");
  printMessage(getLoc(253), SourceMgr::DK_Error, "message", std::nullopt,
               std::nullopt);
  EXPECT_EQ("file.in:24:1: error: message\n"
            "1\n"
            "^\n",
            Output);
}

TEST_F(SourceMgrTest, LocationAtEndOf255ByteBufferEndingInNewline) {
  setMainBuffer(STRING_LITERAL_253_BYTES   // first 253 bytes
                "1\n"                      // + 2 = 255 bytes
                , "file.in");
  printMessage(getLoc(254), SourceMgr::DK_Error, "message", std::nullopt,
               std::nullopt);
  EXPECT_EQ("file.in:24:2: error: message\n"
            "1\n"
            " ^\n",
            Output);
}

TEST_F(SourceMgrTest, LocationPastEndOf255ByteBufferEndingInNewline) {
  setMainBuffer(STRING_LITERAL_253_BYTES   // first 253 bytes
                "1\n"                      // + 2 = 255 bytes
                , "file.in");
  printMessage(getLoc(255), SourceMgr::DK_Error, "message", std::nullopt,
               std::nullopt);
  EXPECT_EQ("file.in:25:1: error: message\n"
            "\n"
            "^\n",
            Output);
}

//===----------------------------------------------------------------------===//
// 256-byte buffer tests
//===----------------------------------------------------------------------===//

TEST_F(SourceMgrTest, LocationBeforeEndOf256ByteBuffer) {
  setMainBuffer(STRING_LITERAL_253_BYTES   // first 253 bytes
                "123"                      // + 3 = 256 bytes
                , "file.in");
  printMessage(getLoc(254), SourceMgr::DK_Error, "message", std::nullopt,
               std::nullopt);
  EXPECT_EQ("file.in:24:2: error: message\n"
            "123\n"
            " ^\n",
            Output);
}

TEST_F(SourceMgrTest, LocationAtEndOf256ByteBuffer) {
  setMainBuffer(STRING_LITERAL_253_BYTES   // first 253 bytes
                "123"                      // + 3 = 256 bytes
                , "file.in");
  printMessage(getLoc(255), SourceMgr::DK_Error, "message", std::nullopt,
               std::nullopt);
  EXPECT_EQ("file.in:24:3: error: message\n"
            "123\n"
            "  ^\n",
            Output);
}

TEST_F(SourceMgrTest, LocationPastEndOf256ByteBuffer) {
  setMainBuffer(STRING_LITERAL_253_BYTES   // first 253 bytes
                "123"                      // + 3 = 256 bytes
                , "file.in");
  printMessage(getLoc(256), SourceMgr::DK_Error, "message", std::nullopt,
               std::nullopt);
  EXPECT_EQ("file.in:24:4: error: message\n"
            "123\n"
            "   ^\n",
            Output);
}

TEST_F(SourceMgrTest, LocationBeforeEndOf256ByteBufferEndingInNewline) {
  setMainBuffer(STRING_LITERAL_253_BYTES   // first 253 bytes
                "12\n"                     // + 3 = 256 bytes
                , "file.in");
  printMessage(getLoc(254), SourceMgr::DK_Error, "message", std::nullopt,
               std::nullopt);
  EXPECT_EQ("file.in:24:2: error: message\n"
            "12\n"
            " ^\n",
            Output);
}

TEST_F(SourceMgrTest, LocationAtEndOf256ByteBufferEndingInNewline) {
  setMainBuffer(STRING_LITERAL_253_BYTES   // first 253 bytes
                "12\n"                     // + 3 = 256 bytes
                , "file.in");
  printMessage(getLoc(255), SourceMgr::DK_Error, "message", std::nullopt,
               std::nullopt);
  EXPECT_EQ("file.in:24:3: error: message\n"
            "12\n"
            "  ^\n",
            Output);
}

TEST_F(SourceMgrTest, LocationPastEndOf256ByteBufferEndingInNewline) {
  setMainBuffer(STRING_LITERAL_253_BYTES   // first 253 bytes
                "12\n"                     // + 3 = 256 bytes
                , "file.in");
  printMessage(getLoc(256), SourceMgr::DK_Error, "message", std::nullopt,
               std::nullopt);
  EXPECT_EQ("file.in:25:1: error: message\n"
            "\n"
            "^\n",
            Output);
}

//===----------------------------------------------------------------------===//
// 257-byte buffer tests
//===----------------------------------------------------------------------===//

TEST_F(SourceMgrTest, LocationBeforeEndOf257ByteBuffer) {
  setMainBuffer(STRING_LITERAL_253_BYTES   // first 253 bytes
                "1234"                     // + 4 = 257 bytes
                , "file.in");
  printMessage(getLoc(255), SourceMgr::DK_Error, "message", std::nullopt,
               std::nullopt);
  EXPECT_EQ("file.in:24:3: error: message\n"
            "1234\n"
            "  ^\n",
            Output);
}

TEST_F(SourceMgrTest, LocationAtEndOf257ByteBuffer) {
  setMainBuffer(STRING_LITERAL_253_BYTES   // first 253 bytes
                "1234"                     // + 4 = 257 bytes
                , "file.in");
  printMessage(getLoc(256), SourceMgr::DK_Error, "message", std::nullopt,
               std::nullopt);
  EXPECT_EQ("file.in:24:4: error: message\n"
            "1234\n"
            "   ^\n",
            Output);
}

TEST_F(SourceMgrTest, LocationPastEndOf257ByteBuffer) {
  setMainBuffer(STRING_LITERAL_253_BYTES   // first 253 bytes
                "1234"                     // + 4 = 257 bytes
                , "file.in");
  printMessage(getLoc(257), SourceMgr::DK_Error, "message", std::nullopt,
               std::nullopt);
  EXPECT_EQ("file.in:24:5: error: message\n"
            "1234\n"
            "    ^\n",
            Output);
}

TEST_F(SourceMgrTest, LocationBeforeEndOf257ByteBufferEndingInNewline) {
  setMainBuffer(STRING_LITERAL_253_BYTES   // first 253 bytes
                "123\n"                    // + 4 = 257 bytes
                , "file.in");
  printMessage(getLoc(255), SourceMgr::DK_Error, "message", std::nullopt,
               std::nullopt);
  EXPECT_EQ("file.in:24:3: error: message\n"
            "123\n"
            "  ^\n",
            Output);
}

TEST_F(SourceMgrTest, LocationAtEndOf257ByteBufferEndingInNewline) {
  setMainBuffer(STRING_LITERAL_253_BYTES   // first 253 bytes
                "123\n"                    // + 4 = 257 bytes
                , "file.in");
  printMessage(getLoc(256), SourceMgr::DK_Error, "message", std::nullopt,
               std::nullopt);
  EXPECT_EQ("file.in:24:4: error: message\n"
            "123\n"
            "   ^\n",
            Output);
}

TEST_F(SourceMgrTest, LocationPastEndOf257ByteBufferEndingInNewline) {
  setMainBuffer(STRING_LITERAL_253_BYTES   // first 253 bytes
                "123\n"                    // + 4 = 257 bytes
                , "file.in");
  printMessage(getLoc(257), SourceMgr::DK_Error, "message", std::nullopt,
               std::nullopt);
  EXPECT_EQ("file.in:25:1: error: message\n"
            "\n"
            "^\n",
            Output);
}

TEST_F(SourceMgrTest, BasicRange) {
  setMainBuffer("aaa bbb\nccc ddd\n", "file.in");
  printMessage(getLoc(4), SourceMgr::DK_Error, "message", getRange(4, 3),
               std::nullopt);

  EXPECT_EQ("file.in:1:5: error: message\n"
            "aaa bbb\n"
            "    ^~~\n",
            Output);
}

TEST_F(SourceMgrTest, RangeWithTab) {
  setMainBuffer("aaa\tbbb\nccc ddd\n", "file.in");
  printMessage(getLoc(4), SourceMgr::DK_Error, "message", getRange(3, 3),
               std::nullopt);

  EXPECT_EQ("file.in:1:5: error: message\n"
            "aaa     bbb\n"
            "   ~~~~~^~\n",
            Output);
}

TEST_F(SourceMgrTest, MultiLineRange) {
  setMainBuffer("aaa bbb\nccc ddd\n", "file.in");
  printMessage(getLoc(4), SourceMgr::DK_Error, "message", getRange(4, 7),
               std::nullopt);

  EXPECT_EQ("file.in:1:5: error: message\n"
            "aaa bbb\n"
            "    ^~~\n",
            Output);
}

TEST_F(SourceMgrTest, MultipleRanges) {
  setMainBuffer("aaa bbb\nccc ddd\n", "file.in");
  SMRange Ranges[] = { getRange(0, 3), getRange(4, 3) };
  printMessage(getLoc(4), SourceMgr::DK_Error, "message", Ranges, std::nullopt);

  EXPECT_EQ("file.in:1:5: error: message\n"
            "aaa bbb\n"
            "~~~ ^~~\n",
            Output);
}

TEST_F(SourceMgrTest, OverlappingRanges) {
  setMainBuffer("aaa bbb\nccc ddd\n", "file.in");
  SMRange Ranges[] = { getRange(0, 3), getRange(2, 4) };
  printMessage(getLoc(4), SourceMgr::DK_Error, "message", Ranges, std::nullopt);

  EXPECT_EQ("file.in:1:5: error: message\n"
            "aaa bbb\n"
            "~~~~^~\n",
            Output);
}

TEST_F(SourceMgrTest, BasicFixit) {
  setMainBuffer("aaa bbb\nccc ddd\n", "file.in");
  printMessage(getLoc(4), SourceMgr::DK_Error, "message", std::nullopt,
               ArrayRef(SMFixIt(getRange(4, 3), "zzz")));

  EXPECT_EQ("file.in:1:5: error: message\n"
            "aaa bbb\n"
            "    ^~~\n"
            "    zzz\n",
            Output);
}

TEST_F(SourceMgrTest, FixitForTab) {
  setMainBuffer("aaa\tbbb\nccc ddd\n", "file.in");
  printMessage(getLoc(3), SourceMgr::DK_Error, "message", std::nullopt,
               ArrayRef(SMFixIt(getRange(3, 1), "zzz")));

  EXPECT_EQ("file.in:1:4: error: message\n"
            "aaa     bbb\n"
            "   ^^^^^\n"
            "   zzz\n",
            Output);
}

TEST_F(SourceMgrTest, PrintWithoutLoc) {
  raw_string_ostream OS(Output);
  auto Diag =
      llvm::SMDiagnostic("file.in", llvm::SourceMgr::DK_Error, "message");
  Diag.print(nullptr, OS);
  EXPECT_EQ("file.in: error: message\n", Output);
  Output.clear();
  Diag.print(nullptr, OS, false, false, false);
  EXPECT_EQ("message\n", Output);
}
