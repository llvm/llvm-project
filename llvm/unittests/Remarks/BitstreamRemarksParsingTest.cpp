//===- unittests/Support/BitstreamRemarksParsingTest.cpp - Parsing tests --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Remarks.h"
#include "llvm/Remarks/Remark.h"
#include "llvm/Remarks/RemarkParser.h"
#include "llvm/Remarks/RemarkSerializer.h"
#include "gtest/gtest.h"

using namespace llvm;

template <size_t N> void parseGood(const char (&Buf)[N]) {
  // 1. Parse the YAML remark -> FromYAMLRemark
  // 2. Serialize it to bitstream -> BSStream
  // 3. Parse it back -> FromBSRemark
  // 4. Compare the remark objects
  //
  // This testing methodology has the drawback of relying on both the YAML
  // remark parser and the bitstream remark serializer. It does simplify
  // testing a lot, since working directly with bitstream is not that easy.

  // 1.
  Expected<std::unique_ptr<remarks::RemarkParser>> MaybeParser =
      remarks::createRemarkParser(remarks::Format::YAML, {Buf, N - 1});
  EXPECT_FALSE(errorToBool(MaybeParser.takeError()));
  EXPECT_TRUE(*MaybeParser != nullptr);

  std::unique_ptr<remarks::Remark> FromYAMLRemark = nullptr;
  remarks::RemarkParser &Parser = **MaybeParser;
  Expected<std::unique_ptr<remarks::Remark>> Remark = Parser.next();
  EXPECT_FALSE(errorToBool(Remark.takeError())); // Check for parsing errors.
  EXPECT_TRUE(*Remark != nullptr);               // At least one remark.
  // Keep the previous remark around.
  FromYAMLRemark = std::move(*Remark);
  Remark = Parser.next();
  Error E = Remark.takeError();
  EXPECT_TRUE(E.isA<remarks::EndOfFileError>());
  EXPECT_TRUE(errorToBool(std::move(E))); // Check for parsing errors.

  // 2.
  remarks::StringTable BSStrTab;
  BSStrTab.internalize(*FromYAMLRemark);
  std::string BSBuf;
  raw_string_ostream BSStream(BSBuf);
  Expected<std::unique_ptr<remarks::RemarkSerializer>> BSSerializer =
      remarks::createRemarkSerializer(remarks::Format::Bitstream,
                                      remarks::SerializerMode::Standalone,
                                      BSStream, std::move(BSStrTab));
  EXPECT_FALSE(errorToBool(BSSerializer.takeError()));
  (*BSSerializer)->emit(*FromYAMLRemark);

  // 3.
  Expected<std::unique_ptr<remarks::RemarkParser>> MaybeBSParser =
      remarks::createRemarkParser(remarks::Format::Bitstream, BSStream.str());
  EXPECT_FALSE(errorToBool(MaybeBSParser.takeError()));
  EXPECT_TRUE(*MaybeBSParser != nullptr);

  std::unique_ptr<remarks::Remark> FromBSRemark = nullptr;
  remarks::RemarkParser &BSParser = **MaybeBSParser;
  Expected<std::unique_ptr<remarks::Remark>> BSRemark = BSParser.next();
  EXPECT_FALSE(errorToBool(BSRemark.takeError())); // Check for parsing errors.
  EXPECT_TRUE(*BSRemark != nullptr);               // At least one remark.
  // Keep the previous remark around.
  FromBSRemark = std::move(*BSRemark);
  BSRemark = BSParser.next();
  Error BSE = BSRemark.takeError();
  EXPECT_TRUE(BSE.isA<remarks::EndOfFileError>());
  EXPECT_TRUE(errorToBool(std::move(BSE))); // Check for parsing errors.

  EXPECT_EQ(*FromYAMLRemark, *FromBSRemark);
}

TEST(BitstreamRemarks, ParsingGood) {
  parseGood("\n"
            "--- !Missed\n"
            "Pass: inline\n"
            "Name: NoDefinition\n"
            "DebugLoc: { File: file.c, Line: 3, Column: 12 }\n"
            "Function: foo\n"
            "Args:\n"
            "  - Callee: bar\n"
            "  - String: ' will not be inlined into '\n"
            "  - Caller: foo\n"
            "    DebugLoc: { File: file.c, Line: 2, Column: 0 }\n"
            "  - String: ' because its definition is unavailable'\n"
            "");

  // No debug loc should also pass.
  parseGood("\n"
            "--- !Missed\n"
            "Pass: inline\n"
            "Name: NoDefinition\n"
            "Function: foo\n"
            "Args:\n"
            "  - Callee: bar\n"
            "  - String: ' will not be inlined into '\n"
            "  - Caller: foo\n"
            "    DebugLoc: { File: file.c, Line: 2, Column: 0 }\n"
            "  - String: ' because its definition is unavailable'\n"
            "");

  // No args is also ok.
  parseGood("\n"
            "--- !Missed\n"
            "Pass: inline\n"
            "Name: NoDefinition\n"
            "DebugLoc: { File: file.c, Line: 3, Column: 12 }\n"
            "Function: foo\n"
            "");
}

// Mandatory common part of a remark.
#define COMMON_REMARK "\nPass: inline\nName: NoDefinition\nFunction: foo\n\n"
// Test all the types.
TEST(BitstreamRemarks, ParsingTypes) {
  // Type: Passed
  parseGood("--- !Passed" COMMON_REMARK);
  // Type: Missed
  parseGood("--- !Missed" COMMON_REMARK);
  // Type: Analysis
  parseGood("--- !Analysis" COMMON_REMARK);
  // Type: AnalysisFPCommute
  parseGood("--- !AnalysisFPCommute" COMMON_REMARK);
  // Type: AnalysisAliasing
  parseGood("--- !AnalysisAliasing" COMMON_REMARK);
  // Type: Failure
  parseGood("--- !Failure" COMMON_REMARK);
}
#undef COMMON_REMARK

static inline StringRef checkStr(StringRef Str, unsigned ExpectedLen) {
  const char *StrData = Str.data();
  unsigned StrLen = Str.size();
  EXPECT_EQ(StrLen, ExpectedLen);
  return StringRef(StrData, StrLen);
}

TEST(BitstreamRemarks, Contents) {
  StringRef Buf = "\n"
                  "--- !Missed\n"
                  "Pass: inline\n"
                  "Name: NoDefinition\n"
                  "DebugLoc: { File: file.c, Line: 3, Column: 12 }\n"
                  "Function: foo\n"
                  "Hotness: 4\n"
                  "Args:\n"
                  "  - Callee: bar\n"
                  "  - String: ' will not be inlined into '\n"
                  "  - Caller: foo\n"
                  "    DebugLoc: { File: file.c, Line: 2, Column: 0 }\n"
                  "  - String: ' because its definition is unavailable'\n"
                  "\n";

  Expected<std::unique_ptr<remarks::RemarkParser>> MaybeParser =
      remarks::createRemarkParser(remarks::Format::YAML, Buf);
  EXPECT_FALSE(errorToBool(MaybeParser.takeError()));
  EXPECT_TRUE(*MaybeParser != nullptr);

  remarks::RemarkParser &Parser = **MaybeParser;
  Expected<std::unique_ptr<remarks::Remark>> MaybeRemark = Parser.next();
  EXPECT_FALSE(
      errorToBool(MaybeRemark.takeError())); // Check for parsing errors.
  EXPECT_TRUE(*MaybeRemark != nullptr);      // At least one remark.

  const remarks::Remark &Remark = **MaybeRemark;
  EXPECT_EQ(Remark.RemarkType, remarks::Type::Missed);
  EXPECT_EQ(checkStr(Remark.PassName, 6), "inline");
  EXPECT_EQ(checkStr(Remark.RemarkName, 12), "NoDefinition");
  EXPECT_EQ(checkStr(Remark.FunctionName, 3), "foo");
  EXPECT_TRUE(Remark.Loc);
  const remarks::RemarkLocation &RL = *Remark.Loc;
  EXPECT_EQ(checkStr(RL.SourceFilePath, 6), "file.c");
  EXPECT_EQ(RL.SourceLine, 3U);
  EXPECT_EQ(RL.SourceColumn, 12U);
  EXPECT_TRUE(Remark.Hotness);
  EXPECT_EQ(*Remark.Hotness, 4U);
  EXPECT_EQ(Remark.Args.size(), 4U);

  unsigned ArgID = 0;
  for (const remarks::Argument &Arg : Remark.Args) {
    switch (ArgID) {
    case 0:
      EXPECT_EQ(checkStr(Arg.Key, 6), "Callee");
      EXPECT_EQ(checkStr(Arg.Val, 3), "bar");
      EXPECT_FALSE(Arg.Loc);
      break;
    case 1:
      EXPECT_EQ(checkStr(Arg.Key, 6), "String");
      EXPECT_EQ(checkStr(Arg.Val, 26), " will not be inlined into ");
      EXPECT_FALSE(Arg.Loc);
      break;
    case 2: {
      EXPECT_EQ(checkStr(Arg.Key, 6), "Caller");
      EXPECT_EQ(checkStr(Arg.Val, 3), "foo");
      EXPECT_TRUE(Arg.Loc);
      const remarks::RemarkLocation &RL = *Arg.Loc;
      EXPECT_EQ(checkStr(RL.SourceFilePath, 6), "file.c");
      EXPECT_EQ(RL.SourceLine, 2U);
      EXPECT_EQ(RL.SourceColumn, 0U);
      break;
    }
    case 3:
      EXPECT_EQ(checkStr(Arg.Key, 6), "String");
      EXPECT_EQ(checkStr(Arg.Val, 38),
                " because its definition is unavailable");
      EXPECT_FALSE(Arg.Loc);
      break;
    default:
      break;
    }
    ++ArgID;
  }

  MaybeRemark = Parser.next();
  Error E = MaybeRemark.takeError();
  EXPECT_TRUE(E.isA<remarks::EndOfFileError>());
  EXPECT_TRUE(errorToBool(std::move(E))); // Check for parsing errors.
}

static inline StringRef checkStr(LLVMRemarkStringRef Str,
                                 unsigned ExpectedLen) {
  const char *StrData = LLVMRemarkStringGetData(Str);
  unsigned StrLen = LLVMRemarkStringGetLen(Str);
  EXPECT_EQ(StrLen, ExpectedLen);
  return StringRef(StrData, StrLen);
}

TEST(BitstreamRemarks, ContentsCAPI) {
  remarks::StringTable BSStrTab;
  remarks::Remark ToSerializeRemark;
  ToSerializeRemark.RemarkType = remarks::Type::Missed;
  ToSerializeRemark.PassName = "inline";
  ToSerializeRemark.RemarkName = "NoDefinition";
  ToSerializeRemark.FunctionName = "foo";
  ToSerializeRemark.Loc = remarks::RemarkLocation{"file.c", 3, 12};
  ToSerializeRemark.Hotness = 0;
  ToSerializeRemark.Args.emplace_back();
  ToSerializeRemark.Args.back().Key = "Callee";
  ToSerializeRemark.Args.back().Val = "bar";
  ToSerializeRemark.Args.emplace_back();
  ToSerializeRemark.Args.back().Key = "String";
  ToSerializeRemark.Args.back().Val = " will not be inlined into ";
  ToSerializeRemark.Args.emplace_back();
  ToSerializeRemark.Args.back().Key = "Caller";
  ToSerializeRemark.Args.back().Val = "foo";
  ToSerializeRemark.Args.back().Loc = remarks::RemarkLocation{"file.c", 2, 0};
  ToSerializeRemark.Args.emplace_back();
  ToSerializeRemark.Args.back().Key = "String";
  ToSerializeRemark.Args.back().Val = " because its definition is unavailable";
  BSStrTab.internalize(ToSerializeRemark);
  std::string BSBuf;
  raw_string_ostream BSStream(BSBuf);
  Expected<std::unique_ptr<remarks::RemarkSerializer>> BSSerializer =
      remarks::createRemarkSerializer(remarks::Format::Bitstream,
                                      remarks::SerializerMode::Standalone,
                                      BSStream, std::move(BSStrTab));
  EXPECT_FALSE(errorToBool(BSSerializer.takeError()));
  (*BSSerializer)->emit(ToSerializeRemark);

  StringRef Buf = BSStream.str();
  LLVMRemarkParserRef Parser =
      LLVMRemarkParserCreateBitstream(Buf.data(), Buf.size());
  LLVMRemarkEntryRef Remark = LLVMRemarkParserGetNext(Parser);
  EXPECT_FALSE(Remark == nullptr);
  EXPECT_EQ(LLVMRemarkEntryGetType(Remark), LLVMRemarkTypeMissed);
  EXPECT_EQ(checkStr(LLVMRemarkEntryGetPassName(Remark), 6), "inline");
  EXPECT_EQ(checkStr(LLVMRemarkEntryGetRemarkName(Remark), 12), "NoDefinition");
  EXPECT_EQ(checkStr(LLVMRemarkEntryGetFunctionName(Remark), 3), "foo");
  LLVMRemarkDebugLocRef DL = LLVMRemarkEntryGetDebugLoc(Remark);
  EXPECT_EQ(checkStr(LLVMRemarkDebugLocGetSourceFilePath(DL), 6), "file.c");
  EXPECT_EQ(LLVMRemarkDebugLocGetSourceLine(DL), 3U);
  EXPECT_EQ(LLVMRemarkDebugLocGetSourceColumn(DL), 12U);
  EXPECT_EQ(LLVMRemarkEntryGetHotness(Remark), 0U);
  EXPECT_EQ(LLVMRemarkEntryGetNumArgs(Remark), 4U);

  unsigned ArgID = 0;
  LLVMRemarkArgRef Arg = LLVMRemarkEntryGetFirstArg(Remark);
  do {
    switch (ArgID) {
    case 0:
      EXPECT_EQ(checkStr(LLVMRemarkArgGetKey(Arg), 6), "Callee");
      EXPECT_EQ(checkStr(LLVMRemarkArgGetValue(Arg), 3), "bar");
      EXPECT_EQ(LLVMRemarkArgGetDebugLoc(Arg), nullptr);
      break;
    case 1:
      EXPECT_EQ(checkStr(LLVMRemarkArgGetKey(Arg), 6), "String");
      EXPECT_EQ(checkStr(LLVMRemarkArgGetValue(Arg), 26),
                " will not be inlined into ");
      EXPECT_EQ(LLVMRemarkArgGetDebugLoc(Arg), nullptr);
      break;
    case 2: {
      EXPECT_EQ(checkStr(LLVMRemarkArgGetKey(Arg), 6), "Caller");
      EXPECT_EQ(checkStr(LLVMRemarkArgGetValue(Arg), 3), "foo");
      LLVMRemarkDebugLocRef DL = LLVMRemarkArgGetDebugLoc(Arg);
      EXPECT_EQ(checkStr(LLVMRemarkDebugLocGetSourceFilePath(DL), 6), "file.c");
      EXPECT_EQ(LLVMRemarkDebugLocGetSourceLine(DL), 2U);
      EXPECT_EQ(LLVMRemarkDebugLocGetSourceColumn(DL), 0U);
      break;
    }
    case 3:
      EXPECT_EQ(checkStr(LLVMRemarkArgGetKey(Arg), 6), "String");
      EXPECT_EQ(checkStr(LLVMRemarkArgGetValue(Arg), 38),
                " because its definition is unavailable");
      EXPECT_EQ(LLVMRemarkArgGetDebugLoc(Arg), nullptr);
      break;
    default:
      break;
    }
    ++ArgID;
  } while ((Arg = LLVMRemarkEntryGetNextArg(Arg, Remark)));

  LLVMRemarkEntryDispose(Remark);

  EXPECT_EQ(LLVMRemarkParserGetNext(Parser), nullptr);

  EXPECT_FALSE(LLVMRemarkParserHasError(Parser));
  LLVMRemarkParserDispose(Parser);
}

static void parseBad(StringRef Input, const char *ErrorMsg) {
  Expected<std::unique_ptr<remarks::RemarkParser>> MaybeBSParser =
      remarks::createRemarkParser(remarks::Format::Bitstream, Input);
  EXPECT_FALSE(errorToBool(MaybeBSParser.takeError()));
  EXPECT_TRUE(*MaybeBSParser != nullptr);

  remarks::RemarkParser &BSParser = **MaybeBSParser;
  Expected<std::unique_ptr<remarks::Remark>> BSRemark = BSParser.next();
  EXPECT_EQ(ErrorMsg, toString(BSRemark.takeError())); // Expect an error.
}

TEST(BitstreamRemarks, ParsingEmpty) {
  parseBad(StringRef(), "End of file reached.");
}

TEST(BitstreamRemarks, ParsingBadMagic) {
  parseBad("KRMR", "Unknown magic number: expecting RMRK, got KRMR.");
}

// Testing malformed bitstream is not easy. We would need to replace bytes in
// the stream to create malformed and unknown records and blocks. There is no
// textual format for bitstream that can be decoded, modified and encoded
// back.

// FIXME: Add tests for the following error messages:
// * Error while parsing META_BLOCK: malformed record entry
// (RECORD_META_CONTAINER_INFO).
// * Error while parsing META_BLOCK: malformed record entry
// (RECORD_META_REMARK_VERSION).
// * Error while parsing META_BLOCK: malformed record entry
// (RECORD_META_STRTAB).
// * Error while parsing META_BLOCK: malformed record entry
// (RECORD_META_EXTERNAL_FILE).
// * Error while parsing META_BLOCK: unknown record entry (NUM).
// * Error while parsing REMARK_BLOCK: malformed record entry
// (RECORD_REMARK_HEADER).
// * Error while parsing REMARK_BLOCK: malformed record entry
// (RECORD_REMARK_DEBUG_LOC).
// * Error while parsing REMARK_BLOCK: malformed record entry
// (RECORD_REMARK_HOTNESS).
// * Error while parsing REMARK_BLOCK: malformed record entry
// (RECORD_REMARK_ARG_WITH_DEBUGLOC).
// * Error while parsing REMARK_BLOCK: malformed record entry
// (RECORD_REMARK_ARG_WITHOUT_DEBUGLOC).
// * Error while parsing REMARK_BLOCK: unknown record entry (NUM).
// * Error while parsing META_BLOCK: expecting [ENTER_SUBBLOCO, META_BLOCK,
// ...].
// * Error while entering META_BLOCK.
// * Error while parsing META_BLOCK: expecting records.
// * Error while parsing META_BLOCK: unterminated block.
// * Error while parsing REMARK_BLOCK: expecting [ENTER_SUBBLOCO, REMARK_BLOCK,
// ...].
// * Error while entering REMARK_BLOCK.
// * Error while parsing REMARK_BLOCK: expecting records.
// * Error while parsing REMARK_BLOCK: unterminated block.
// * Error while parsing BLOCKINFO_BLOCK: expecting [ENTER_SUBBLOCK,
// BLOCKINFO_BLOCK, ...].
// * Error while parsing BLOCKINFO_BLOCK.
// * Unexpected error while parsing bitstream.
// * Expecting META_BLOCK after the BLOCKINFO_BLOCK.
// * Error while parsing BLOCK_META: missing container version.
// * Error while parsing BLOCK_META: invalid container type.
// * Error while parsing BLOCK_META: missing container type.
// * Error while parsing BLOCK_META: missing string table.
// * Error while parsing BLOCK_META: missing remark version.
// * Error while parsing BLOCK_META: missing external file path.
// * Error while parsing external file's BLOCK_META: wrong container type.
// * Error while parsing external file's BLOCK_META: mismatching versions:
// original meta: NUM, external file meta: NUM.
// * Error while parsing BLOCK_REMARK: missing string table.
// * Error while parsing BLOCK_REMARK: missing remark type.
// * Error while parsing BLOCK_REMARK: unknown remark type.
// * Error while parsing BLOCK_REMARK: missing remark name.
// * Error while parsing BLOCK_REMARK: missing remark pass.
// * Error while parsing BLOCK_REMARK: missing remark function name.
// * Error while parsing BLOCK_REMARK: missing key in remark argument.
// * Error while parsing BLOCK_REMARK: missing value in remark argument.
