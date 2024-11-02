//===- unittest/Support/BitstreamRemarksSerializerTest.cpp ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitcode/BitcodeAnalyzer.h"
#include "llvm/Remarks/BitstreamRemarkSerializer.h"
#include "llvm/Remarks/Remark.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <string>

// We need to supprt Windows paths as well. In order to have paths with the same
// length, use a different path according to the platform.
#ifdef _WIN32
#define EXTERNALFILETESTPATH "C:/externalfi"
#else
#define EXTERNALFILETESTPATH "/externalfile"
#endif

using namespace llvm;

static void checkAnalyze(StringRef Input, StringRef Expected) {
  std::string OutputBuf;
  raw_string_ostream OutputOS(OutputBuf);
  BCDumpOptions O(OutputOS);
  O.ShowBinaryBlobs = true;
  BitcodeAnalyzer BA(Input);
  EXPECT_FALSE(BA.analyze(O)); // Expect no errors.
  EXPECT_EQ(OutputOS.str(), Expected);
}

static void check(remarks::SerializerMode Mode, const remarks::Remark &R,
                  StringRef ExpectedR, Optional<StringRef> ExpectedMeta,
                  Optional<remarks::StringTable> StrTab) {
  // Emit the remark.
  std::string InputBuf;
  raw_string_ostream InputOS(InputBuf);
  Expected<std::unique_ptr<remarks::RemarkSerializer>> MaybeSerializer = [&] {
    if (StrTab)
      return createRemarkSerializer(remarks::Format::Bitstream, Mode, InputOS,
                                    std::move(*StrTab));
    else
      return createRemarkSerializer(remarks::Format::Bitstream, Mode, InputOS);
  }();
  EXPECT_FALSE(errorToBool(MaybeSerializer.takeError()));
  std::unique_ptr<remarks::RemarkSerializer> Serializer =
      std::move(*MaybeSerializer);
  Serializer->emit(R);

  // Analyze the serialized remark.
  checkAnalyze(InputOS.str(), ExpectedR);

  // Analyze the serialized metadata if it's not in standalone mode.
  if (ExpectedMeta) {
    std::string MetaBuf;
    raw_string_ostream MetaOS(MetaBuf);
    std::unique_ptr<remarks::MetaSerializer> MetaSerializer =
        Serializer->metaSerializer(MetaOS, StringRef(EXTERNALFILETESTPATH));
    MetaSerializer->emit();
    checkAnalyze(MetaOS.str(), *ExpectedMeta);
  }
}

static void check(const remarks::Remark &R, StringRef ExpectedR,
                  StringRef ExpectedMeta,
                  Optional<remarks::StringTable> StrTab = std::nullopt) {
  return check(remarks::SerializerMode::Separate, R, ExpectedR, ExpectedMeta,
               std::move(StrTab));
}

static void
checkStandalone(const remarks::Remark &R, StringRef ExpectedR,
                Optional<remarks::StringTable> StrTab = std::nullopt) {
  return check(remarks::SerializerMode::Standalone, R, ExpectedR,
               /*ExpectedMeta=*/std::nullopt, std::move(StrTab));
}

TEST(BitstreamRemarkSerializer, SeparateRemarkFileNoOptionals) {
  remarks::Remark R;
  R.RemarkType = remarks::Type::Missed;
  R.PassName = "pass";
  R.RemarkName = "remark";
  R.FunctionName = "function";
  check(R,
        "<BLOCKINFO_BLOCK/>\n"
        "<Meta BlockID=8 NumWords=3 BlockCodeSize=3>\n"
        "  <Container info codeid=1 abbrevid=4 op0=0 op1=1/>\n"
        "  <Remark version codeid=2 abbrevid=5 op0=0/>\n"
        "</Meta>\n"
        "<Remark BlockID=9 NumWords=1 BlockCodeSize=4>\n"
        "  <Remark header codeid=5 abbrevid=4 op0=2 op1=0 op2=1 op3=2/>\n"
        "</Remark>\n",
        "<BLOCKINFO_BLOCK/>\n"
        "<Meta BlockID=8 NumWords=14 BlockCodeSize=3>\n"
        "  <Container info codeid=1 abbrevid=4 op0=0 op1=0/>\n"
        "  <String table codeid=3 abbrevid=5/> blob data = "
        "'remark\\x00pass\\x00function\\x00'\n"
        "  <External File codeid=4 abbrevid=6/> blob data = "
        "'" EXTERNALFILETESTPATH"'\n"
        "</Meta>\n");
}

TEST(BitstreamRemarkSerializer, SeparateRemarkFileNoOptionalsSeparateStrTab) {
  remarks::StringTable StrTab;
  StrTab.add("function");
  StrTab.add("pass");
  StrTab.add("remark");
  remarks::Remark R;
  R.RemarkType = remarks::Type::Missed;
  R.PassName = "pass";
  R.RemarkName = "remark";
  R.FunctionName = "function";
  check(R,
        "<BLOCKINFO_BLOCK/>\n"
        "<Meta BlockID=8 NumWords=3 BlockCodeSize=3>\n"
        "  <Container info codeid=1 abbrevid=4 op0=0 op1=1/>\n"
        "  <Remark version codeid=2 abbrevid=5 op0=0/>\n"
        "</Meta>\n"
        "<Remark BlockID=9 NumWords=1 BlockCodeSize=4>\n"
        "  <Remark header codeid=5 abbrevid=4 op0=2 op1=2 op2=1 op3=0/>\n"
        "</Remark>\n",
        "<BLOCKINFO_BLOCK/>\n"
        "<Meta BlockID=8 NumWords=14 BlockCodeSize=3>\n"
        "  <Container info codeid=1 abbrevid=4 op0=0 op1=0/>\n"
        "  <String table codeid=3 abbrevid=5/> blob data = "
        "'function\\x00pass\\x00remark\\x00'\n"
        "  <External File codeid=4 abbrevid=6/> blob data = "
        "'" EXTERNALFILETESTPATH"'\n"
        "</Meta>\n",
        std::move(StrTab));
}

TEST(BitstreamRemarkSerializer, SeparateRemarkFileDebugLoc) {
  remarks::Remark R;
  R.RemarkType = remarks::Type::Missed;
  R.PassName = "pass";
  R.RemarkName = "remark";
  R.FunctionName = "function";
  R.Loc.emplace();
  R.Loc->SourceFilePath = "path";
  R.Loc->SourceLine = 99;
  R.Loc->SourceColumn = 55;
  check(R,
        "<BLOCKINFO_BLOCK/>\n"
        "<Meta BlockID=8 NumWords=3 BlockCodeSize=3>\n"
        "  <Container info codeid=1 abbrevid=4 op0=0 op1=1/>\n"
        "  <Remark version codeid=2 abbrevid=5 op0=0/>\n"
        "</Meta>\n"
        "<Remark BlockID=9 NumWords=4 BlockCodeSize=4>\n"
        "  <Remark header codeid=5 abbrevid=4 op0=2 op1=0 op2=1 op3=2/>\n"
        "  <Remark debug location codeid=6 abbrevid=5 op0=3 op1=99 op2=55/>\n"
        "</Remark>\n",
        "<BLOCKINFO_BLOCK/>\n"
        "<Meta BlockID=8 NumWords=15 BlockCodeSize=3>\n"
        "  <Container info codeid=1 abbrevid=4 op0=0 op1=0/>\n"
        "  <String table codeid=3 abbrevid=5/> blob data = "
        "'remark\\x00pass\\x00function\\x00path\\x00'\n"
        "  <External File codeid=4 abbrevid=6/> blob data = "
        "'" EXTERNALFILETESTPATH"'\n"
        "</Meta>\n");
}

TEST(BitstreamRemarkSerializer, SeparateRemarkFileHotness) {
  remarks::Remark R;
  R.RemarkType = remarks::Type::Missed;
  R.PassName = "pass";
  R.RemarkName = "remark";
  R.FunctionName = "function";
  R.Hotness.emplace(999999999);
  check(R,
        "<BLOCKINFO_BLOCK/>\n"
        "<Meta BlockID=8 NumWords=3 BlockCodeSize=3>\n"
        "  <Container info codeid=1 abbrevid=4 op0=0 op1=1/>\n"
        "  <Remark version codeid=2 abbrevid=5 op0=0/>\n"
        "</Meta>\n"
        "<Remark BlockID=9 NumWords=3 BlockCodeSize=4>\n"
        "  <Remark header codeid=5 abbrevid=4 op0=2 op1=0 op2=1 op3=2/>\n"
        "  <Remark hotness codeid=7 abbrevid=6 op0=999999999/>\n"
        "</Remark>\n",
        "<BLOCKINFO_BLOCK/>\n"
        "<Meta BlockID=8 NumWords=14 BlockCodeSize=3>\n"
        "  <Container info codeid=1 abbrevid=4 op0=0 op1=0/>\n"
        "  <String table codeid=3 abbrevid=5/> blob data = "
        "'remark\\x00pass\\x00function\\x00'\n"
        "  <External File codeid=4 abbrevid=6/> blob data = "
        "'" EXTERNALFILETESTPATH"'\n"
        "</Meta>\n");
}

TEST(BitstreamRemarkSerializer, SeparateRemarkFileArgNoDebugLoc) {
  remarks::Remark R;
  R.RemarkType = remarks::Type::Missed;
  R.PassName = "pass";
  R.RemarkName = "remark";
  R.FunctionName = "function";
  R.Args.emplace_back();
  R.Args.back().Key = "key";
  R.Args.back().Val = "value";
  check(R,
        "<BLOCKINFO_BLOCK/>\n"
        "<Meta BlockID=8 NumWords=3 BlockCodeSize=3>\n"
        "  <Container info codeid=1 abbrevid=4 op0=0 op1=1/>\n"
        "  <Remark version codeid=2 abbrevid=5 op0=0/>\n"
        "</Meta>\n"
        "<Remark BlockID=9 NumWords=2 BlockCodeSize=4>\n"
        "  <Remark header codeid=5 abbrevid=4 op0=2 op1=0 op2=1 op3=2/>\n"
        "  <Argument codeid=9 abbrevid=8 op0=3 op1=4/>\n"
        "</Remark>\n",
        "<BLOCKINFO_BLOCK/>\n"
        "<Meta BlockID=8 NumWords=16 BlockCodeSize=3>\n"
        "  <Container info codeid=1 abbrevid=4 op0=0 op1=0/>\n"
        "  <String table codeid=3 abbrevid=5/> blob data = "
        "'remark\\x00pass\\x00function\\x00key\\x00value\\x00'\n"
        "  <External File codeid=4 abbrevid=6/> blob data = "
        "'" EXTERNALFILETESTPATH"'\n"
        "</Meta>\n");
}

TEST(BitstreamRemarkSerializer, SeparateRemarkFileArgDebugLoc) {
  remarks::Remark R;
  R.RemarkType = remarks::Type::Missed;
  R.PassName = "pass";
  R.RemarkName = "remark";
  R.FunctionName = "function";
  R.Args.emplace_back();
  R.Args.back().Key = "key";
  R.Args.back().Val = "value";
  R.Args.back().Loc.emplace();
  R.Args.back().Loc->SourceFilePath = "path";
  R.Args.back().Loc->SourceLine = 99;
  R.Args.back().Loc->SourceColumn = 55;
  check(R,
        "<BLOCKINFO_BLOCK/>\n"
        "<Meta BlockID=8 NumWords=3 BlockCodeSize=3>\n"
        "  <Container info codeid=1 abbrevid=4 op0=0 op1=1/>\n"
        "  <Remark version codeid=2 abbrevid=5 op0=0/>\n"
        "</Meta>\n"
        "<Remark BlockID=9 NumWords=4 BlockCodeSize=4>\n"
        "  <Remark header codeid=5 abbrevid=4 op0=2 op1=0 op2=1 op3=2/>\n"
        "  <Argument with debug location codeid=8 abbrevid=7 op0=3 op1=4 op2=5 "
        "op3=99 op4=55/>\n"
        "</Remark>\n",
        "<BLOCKINFO_BLOCK/>\n"
        "<Meta BlockID=8 NumWords=17 BlockCodeSize=3>\n"
        "  <Container info codeid=1 abbrevid=4 op0=0 op1=0/>\n"
        "  <String table codeid=3 abbrevid=5/> blob data = "
        "'remark\\x00pass\\x00function\\x00key\\x00value\\x00path\\x00'\n"
        "  <External File codeid=4 abbrevid=6/> blob data = "
        "'" EXTERNALFILETESTPATH"'\n"
        "</Meta>\n");
}

TEST(BitstreamRemarkSerializer, SeparateRemarkFileAll) {
  remarks::Remark R;
  R.RemarkType = remarks::Type::Missed;
  R.PassName = "pass";
  R.RemarkName = "remark";
  R.FunctionName = "function";
  R.Loc.emplace();
  R.Loc->SourceFilePath = "path";
  R.Loc->SourceLine = 99;
  R.Loc->SourceColumn = 55;
  R.Hotness.emplace(999999999);
  R.Args.emplace_back();
  R.Args.back().Key = "key";
  R.Args.back().Val = "value";
  R.Args.back().Loc.emplace();
  R.Args.back().Loc->SourceFilePath = "argpath";
  R.Args.back().Loc->SourceLine = 11;
  R.Args.back().Loc->SourceColumn = 66;
  check(R,
        "<BLOCKINFO_BLOCK/>\n"
        "<Meta BlockID=8 NumWords=3 BlockCodeSize=3>\n"
        "  <Container info codeid=1 abbrevid=4 op0=0 op1=1/>\n"
        "  <Remark version codeid=2 abbrevid=5 op0=0/>\n"
        "</Meta>\n"
        "<Remark BlockID=9 NumWords=8 BlockCodeSize=4>\n"
        "  <Remark header codeid=5 abbrevid=4 op0=2 op1=0 op2=1 op3=2/>\n"
        "  <Remark debug location codeid=6 abbrevid=5 op0=3 op1=99 op2=55/>\n"
        "  <Remark hotness codeid=7 abbrevid=6 op0=999999999/>\n"
        "  <Argument with debug location codeid=8 abbrevid=7 op0=4 op1=5 op2=6 "
        "op3=11 op4=66/>\n"
        "</Remark>\n",
        "<BLOCKINFO_BLOCK/>\n"
        "<Meta BlockID=8 NumWords=19 BlockCodeSize=3>\n"
        "  <Container info codeid=1 abbrevid=4 op0=0 op1=0/>\n"
        "  <String table codeid=3 abbrevid=5/> blob data = "
        "'remark\\x00pass\\x00function\\x00path\\x00key\\x00value\\x00argpa"
        "th\\x00'\n  <External File codeid=4 abbrevid=6/> blob data = "
        "'" EXTERNALFILETESTPATH"'\n"
        "</Meta>\n");
}

TEST(BitstreamRemarkSerializer, Standalone) {
  // Pre-populate the string table.
  remarks::StringTable StrTab;
  StrTab.add("pass");
  StrTab.add("remark");
  StrTab.add("function");
  StrTab.add("path");
  StrTab.add("key");
  StrTab.add("value");
  StrTab.add("argpath");
  remarks::Remark R;
  R.RemarkType = remarks::Type::Missed;
  R.PassName = "pass";
  R.RemarkName = "remark";
  R.FunctionName = "function";
  R.Loc.emplace();
  R.Loc->SourceFilePath = "path";
  R.Loc->SourceLine = 99;
  R.Loc->SourceColumn = 55;
  R.Hotness.emplace(999999999);
  R.Args.emplace_back();
  R.Args.back().Key = "key";
  R.Args.back().Val = "value";
  R.Args.back().Loc.emplace();
  R.Args.back().Loc->SourceFilePath = "argpath";
  R.Args.back().Loc->SourceLine = 11;
  R.Args.back().Loc->SourceColumn = 66;
  checkStandalone(
      R,
      "<BLOCKINFO_BLOCK/>\n"
      "<Meta BlockID=8 NumWords=15 BlockCodeSize=3>\n"
      "  <Container info codeid=1 abbrevid=4 op0=0 op1=2/>\n"
      "  <Remark version codeid=2 abbrevid=5 op0=0/>\n"
      "  <String table codeid=3 abbrevid=6/> blob data = "
      "'pass\\x00remark\\x00function\\x00path\\x00key\\x00value\\x00argpath\\x0"
      "0'\n"
      "</Meta>\n"
      "<Remark BlockID=9 NumWords=8 BlockCodeSize=4>\n"
      "  <Remark header codeid=5 abbrevid=4 op0=2 op1=1 op2=0 op3=2/>\n"
      "  <Remark debug location codeid=6 abbrevid=5 op0=3 op1=99 op2=55/>\n"
      "  <Remark hotness codeid=7 abbrevid=6 op0=999999999/>\n"
      "  <Argument with debug location codeid=8 abbrevid=7 op0=4 op1=5 op2=6 "
      "op3=11 op4=66/>\n"
      "</Remark>\n",
      std::move(StrTab));
}
