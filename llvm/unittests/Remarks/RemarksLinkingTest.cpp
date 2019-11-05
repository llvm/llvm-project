//===- unittest/Support/RemarksLinkingTest.cpp - Linking tests ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitcode/BitcodeAnalyzer.h"
#include "llvm/Remarks/RemarkLinker.h"
#include "llvm/Remarks/RemarkSerializer.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <string>

using namespace llvm;

static void serializeAndCheck(remarks::RemarkLinker &RL,
                              remarks::Format OutputFormat,
                              StringRef ExpectedOutput) {
  // 1. Create a serializer.
  // 2. Serialize all the remarks from the linker.
  // 3. Check that it matches the output.
  std::string Buf;
  raw_string_ostream OS(Buf);
  Error E = RL.serialize(OS, OutputFormat);
  EXPECT_FALSE(static_cast<bool>(E));

  // For bitstream, run it through the analyzer.
  if (OutputFormat == remarks::Format::Bitstream) {
    std::string AnalyzeBuf;
    raw_string_ostream AnalyzeOS(AnalyzeBuf);
    BCDumpOptions O(AnalyzeOS);
    O.ShowBinaryBlobs = true;
    BitcodeAnalyzer BA(OS.str());
    EXPECT_FALSE(BA.analyze(O)); // Expect no errors.
    EXPECT_EQ(AnalyzeOS.str(), ExpectedOutput);
  } else {
    EXPECT_EQ(OS.str(), ExpectedOutput);
  }
}

static void check(remarks::Format InputFormat, StringRef Input,
                  remarks::Format OutputFormat, StringRef ExpectedOutput) {
  remarks::RemarkLinker RL;
  EXPECT_FALSE(RL.link(Input, InputFormat));
  serializeAndCheck(RL, OutputFormat, ExpectedOutput);
}

static void check(remarks::Format InputFormat, StringRef Input,
                  remarks::Format InputFormat2, StringRef Input2,
                  remarks::Format OutputFormat, StringRef ExpectedOutput) {
  remarks::RemarkLinker RL;
  EXPECT_FALSE(RL.link(Input, InputFormat));
  EXPECT_FALSE(RL.link(Input2, InputFormat2));
  serializeAndCheck(RL, OutputFormat, ExpectedOutput);
}

TEST(Remarks, LinkingGoodYAML) {
  // One YAML remark.
  check(remarks::Format::YAML,
        "--- !Missed\n"
        "Pass:            inline\n"
        "Name:            NoDefinition\n"
        "DebugLoc:        { File: file.c, Line: 3, Column: 12 }\n"
        "Function:        foo\n"
        "...\n",
        remarks::Format::YAML,
        "--- !Missed\n"
        "Pass:            inline\n"
        "Name:            NoDefinition\n"
        "DebugLoc:        { File: file.c, Line: 3, Column: 12 }\n"
        "Function:        foo\n"
        "...\n");

  // Check that we don't keep remarks without debug locations.
  check(remarks::Format::YAML,
        "--- !Missed\n"
        "Pass:            inline\n"
        "Name:            NoDefinition\n"
        "Function:        foo\n"
        "...\n",
        remarks::Format::YAML, "");

  // Check that we deduplicate remarks.
  check(remarks::Format::YAML,
        "--- !Missed\n"
        "Pass:            inline\n"
        "Name:            NoDefinition\n"
        "DebugLoc:        { File: file.c, Line: 3, Column: 12 }\n"
        "Function:        foo\n"
        "...\n"
        "--- !Missed\n"
        "Pass:            inline\n"
        "Name:            NoDefinition\n"
        "DebugLoc:        { File: file.c, Line: 3, Column: 12 }\n"
        "Function:        foo\n"
        "...\n",
        remarks::Format::YAML,
        "--- !Missed\n"
        "Pass:            inline\n"
        "Name:            NoDefinition\n"
        "DebugLoc:        { File: file.c, Line: 3, Column: 12 }\n"
        "Function:        foo\n"
        "...\n");
}

TEST(Remarks, LinkingGoodBitstream) {
  // One YAML remark.
  check(remarks::Format::YAML,
        "--- !Missed\n"
        "Pass:            inline\n"
        "Name:            NoDefinition\n"
        "DebugLoc:        { File: file.c, Line: 3, Column: 12 }\n"
        "Function:        foo\n"
        "...\n",
        remarks::Format::Bitstream,
        "<BLOCKINFO_BLOCK/>\n"
        "<Meta BlockID=8 NumWords=12 BlockCodeSize=3>\n"
        "  <Container info codeid=1 abbrevid=4 op0=0 op1=2/>\n"
        "  <Remark version codeid=2 abbrevid=5 op0=0/>\n"
        "  <String table codeid=3 abbrevid=6/> blob data = "
        "'inline\\x00NoDefinition\\x00foo\\x00file.c\\x00'\n"
        "</Meta>\n"
        "<Remark BlockID=9 NumWords=4 BlockCodeSize=4>\n"
        "  <Remark header codeid=5 abbrevid=4 op0=2 op1=1 op2=0 op3=2/>\n"
        "  <Remark debug location codeid=6 abbrevid=5 op0=3 op1=3 op2=12/>\n"
        "</Remark>\n");

  // Check that we deduplicate remarks.
  check(remarks::Format::YAML,
        "--- !Missed\n"
        "Pass:            inline\n"
        "Name:            NoDefinition\n"
        "DebugLoc:        { File: file.c, Line: 3, Column: 12 }\n"
        "Function:        foo\n"
        "...\n"
        "--- !Missed\n"
        "Pass:            inline\n"
        "Name:            NoDefinition\n"
        "DebugLoc:        { File: file.c, Line: 3, Column: 12 }\n"
        "Function:        foo\n"
        "...\n",
        remarks::Format::Bitstream,
        "<BLOCKINFO_BLOCK/>\n"
        "<Meta BlockID=8 NumWords=12 BlockCodeSize=3>\n"
        "  <Container info codeid=1 abbrevid=4 op0=0 op1=2/>\n"
        "  <Remark version codeid=2 abbrevid=5 op0=0/>\n"
        "  <String table codeid=3 abbrevid=6/> blob data = "
        "'inline\\x00NoDefinition\\x00foo\\x00file.c\\x00'\n"
        "</Meta>\n"
        "<Remark BlockID=9 NumWords=4 BlockCodeSize=4>\n"
        "  <Remark header codeid=5 abbrevid=4 op0=2 op1=1 op2=0 op3=2/>\n"
        "  <Remark debug location codeid=6 abbrevid=5 op0=3 op1=3 op2=12/>\n"
        "</Remark>\n");
}

TEST(Remarks, LinkingGoodStrTab) {
  // Check that remarks from different entries use the same strtab.
  check(remarks::Format::YAML,
        "--- !Missed\n"
        "Pass:            inline\n"
        "Name:            NoDefinition\n"
        "DebugLoc:        { File: file.c, Line: 3, Column: 12 }\n"
        "Function:        foo\n"
        "...\n",
        remarks::Format::YAML,
        "--- !Passed\n"
        "Pass:            inline\n"
        "Name:            Ok\n"
        "DebugLoc:        { File: file.c, Line: 3, Column: 12 }\n"
        "Function:        foo\n"
        "...\n",
        remarks::Format::YAMLStrTab,
        StringRef("REMARKS\0\0\0\0\0\0\0\0\0\x22\0\0\0\0\0\0\0"
                  "inline\0NoDefinition\0foo\0file.c\0Ok\0"
                  "--- !Passed\n"
                  "Pass:            0\n"
                  "Name:            4\n"
                  "DebugLoc:        { File: 3, Line: 3, Column: 12 }\n"
                  "Function:        2\n"
                  "...\n"
                  "--- !Missed\n"
                  "Pass:            0\n"
                  "Name:            1\n"
                  "DebugLoc:        { File: 3, Line: 3, Column: 12 }\n"
                  "Function:        2\n"
                  "...\n",
                  304));
}

// Check that we propagate parsing errors.
TEST(Remarks, LinkingError) {
  remarks::RemarkLinker RL;
  {
    Error E = RL.link("badyaml", remarks::Format::YAML);
    EXPECT_TRUE(static_cast<bool>(E));
    EXPECT_EQ(toString(std::move(E)),
              "YAML:1:1: error: document root is not of mapping type.\n"
              "\n"
              "badyaml\n"
              "^~~~~~~\n"
              "\n");
  }

  {
    // Check that the prepend path is propagated and fails with the full path.
    RL.setExternalFilePrependPath("/baddir/");
    Error E = RL.link(
        StringRef("REMARKS\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0badfile.opt.yaml",
                  40),
        remarks::Format::YAMLStrTab);
    EXPECT_TRUE(static_cast<bool>(E));
    std::string ErrorMessage = toString(std::move(E));
    EXPECT_EQ(StringRef(ErrorMessage).lower(),
              StringRef("'/baddir/badfile.opt.yaml': No such file or directory")
                  .lower());
  }
}
