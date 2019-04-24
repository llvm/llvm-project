//===- unittest/Support/YAMLRemarksParsingTest.cpp - OptTable tests -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Remarks.h"
#include "llvm/Remarks/Remark.h"
#include "llvm/Remarks/RemarkParser.h"
#include "gtest/gtest.h"

using namespace llvm;

template <size_t N> void parseGood(const char (&Buf)[N]) {
  remarks::Parser Parser({Buf, N - 1});
  Expected<const remarks::Remark *> Remark = Parser.getNext();
  EXPECT_FALSE(errorToBool(Remark.takeError())); // Check for parsing errors.
  EXPECT_TRUE(*Remark != nullptr);               // At least one remark.
  Remark = Parser.getNext();
  EXPECT_FALSE(errorToBool(Remark.takeError())); // Check for parsing errors.
  EXPECT_TRUE(*Remark == nullptr); // Check that there are no more remarks.
}

template <size_t N>
bool parseExpectError(const char (&Buf)[N], const char *Error) {
  remarks::Parser Parser({Buf, N - 1});
  Expected<const remarks::Remark *> Remark = Parser.getNext();
  EXPECT_FALSE(Remark); // Expect an error here.

  std::string ErrorStr;
  raw_string_ostream Stream(ErrorStr);
  handleAllErrors(Remark.takeError(),
                  [&](const ErrorInfoBase &EIB) { EIB.log(Stream); });
  return StringRef(Stream.str()).contains(Error);
}

TEST(YAMLRemarks, ParsingEmpty) {
  EXPECT_TRUE(parseExpectError("\n\n", "document root is not of mapping type."));
}

TEST(YAMLRemarks, ParsingNotYAML) {
  EXPECT_TRUE(
      parseExpectError("\x01\x02\x03\x04\x05\x06", "not a valid YAML file."));
}

TEST(YAMLRemarks, ParsingGood) {
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

  // Different order.
  parseGood("\n"
            "--- !Missed\n"
            "DebugLoc: { Line: 3, Column: 12, File: file.c }\n"
            "Function: foo\n"
            "Name: NoDefinition\n"
            "Args:\n"
            "  - Callee: bar\n"
            "  - String: ' will not be inlined into '\n"
            "  - Caller: foo\n"
            "    DebugLoc: { File: file.c, Line: 2, Column: 0 }\n"
            "  - String: ' because its definition is unavailable'\n"
            "Pass: inline\n"
            "");
}

// Mandatory common part of a remark.
#define COMMON_REMARK "\nPass: inline\nName: NoDefinition\nFunction: foo\n\n"
// Test all the types.
TEST(YAMLRemarks, ParsingTypes) {
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

TEST(YAMLRemarks, ParsingMissingFields) {
  // No type.
  EXPECT_TRUE(parseExpectError("\n"
                   "---\n"
                   "Pass: inline\n"
                   "Name: NoDefinition\n"
                   "Function: foo\n"
                   "",
                   "expected a remark tag."));
  // No pass.
  EXPECT_TRUE(parseExpectError("\n"
                   "--- !Missed\n"
                   "Name: NoDefinition\n"
                   "Function: foo\n"
                   "",
                   "Type, Pass, Name or Function missing."));
  // No name.
  EXPECT_TRUE(parseExpectError("\n"
                   "--- !Missed\n"
                   "Pass: inline\n"
                   "Function: foo\n"
                   "",
                   "Type, Pass, Name or Function missing."));
  // No function.
  EXPECT_TRUE(parseExpectError("\n"
                   "--- !Missed\n"
                   "Pass: inline\n"
                   "Name: NoDefinition\n"
                   "",
                   "Type, Pass, Name or Function missing."));
  // Debug loc but no file.
  EXPECT_TRUE(parseExpectError("\n"
                   "--- !Missed\n"
                   "Pass: inline\n"
                   "Name: NoDefinition\n"
                   "Function: foo\n"
                   "DebugLoc: { Line: 3, Column: 12 }\n"
                   "",
                   "DebugLoc node incomplete."));
  // Debug loc but no line.
  EXPECT_TRUE(parseExpectError("\n"
                   "--- !Missed\n"
                   "Pass: inline\n"
                   "Name: NoDefinition\n"
                   "Function: foo\n"
                   "DebugLoc: { File: file.c, Column: 12 }\n"
                   "",
                   "DebugLoc node incomplete."));
  // Debug loc but no column.
  EXPECT_TRUE(parseExpectError("\n"
                   "--- !Missed\n"
                   "Pass: inline\n"
                   "Name: NoDefinition\n"
                   "Function: foo\n"
                   "DebugLoc: { File: file.c, Line: 3 }\n"
                   "",
                   "DebugLoc node incomplete."));
}

TEST(YAMLRemarks, ParsingWrongTypes) {
  // Wrong debug loc type.
  EXPECT_TRUE(parseExpectError("\n"
                   "--- !Missed\n"
                   "Pass: inline\n"
                   "Name: NoDefinition\n"
                   "Function: foo\n"
                   "DebugLoc: foo\n"
                   "",
                   "expected a value of mapping type."));
  // Wrong line type.
  EXPECT_TRUE(parseExpectError("\n"
                   "--- !Missed\n"
                   "Pass: inline\n"
                   "Name: NoDefinition\n"
                   "Function: foo\n"
                   "DebugLoc: { File: file.c, Line: b, Column: 12 }\n"
                   "",
                   "expected a value of integer type."));
  // Wrong column type.
  EXPECT_TRUE(parseExpectError("\n"
                   "--- !Missed\n"
                   "Pass: inline\n"
                   "Name: NoDefinition\n"
                   "Function: foo\n"
                   "DebugLoc: { File: file.c, Line: 3, Column: c }\n"
                   "",
                   "expected a value of integer type."));
  // Wrong args type.
  EXPECT_TRUE(parseExpectError("\n"
                   "--- !Missed\n"
                   "Pass: inline\n"
                   "Name: NoDefinition\n"
                   "Function: foo\n"
                   "Args: foo\n"
                   "",
                   "wrong value type for key."));
  // Wrong key type.
  EXPECT_TRUE(parseExpectError("\n"
                   "--- !Missed\n"
                   "{ A: a }: inline\n"
                   "Name: NoDefinition\n"
                   "Function: foo\n"
                   "",
                   "key is not a string."));
  // Debug loc with unknown entry.
  EXPECT_TRUE(parseExpectError("\n"
                   "--- !Missed\n"
                   "Pass: inline\n"
                   "Name: NoDefinition\n"
                   "Function: foo\n"
                   "DebugLoc: { File: file.c, Column: 12, Unknown: 12 }\n"
                   "",
                   "unknown entry in DebugLoc map."));
  // Unknown entry.
  EXPECT_TRUE(parseExpectError("\n"
                   "--- !Missed\n"
                   "Unknown: inline\n"
                   "",
                   "unknown key."));
  // Not a scalar.
  EXPECT_TRUE(parseExpectError("\n"
                   "--- !Missed\n"
                   "Pass: { File: a, Line: 1, Column: 2 }\n"
                   "Name: NoDefinition\n"
                   "Function: foo\n"
                   "",
                   "expected a value of scalar type."));
  // Not a string file in debug loc.
  EXPECT_TRUE(parseExpectError("\n"
                   "--- !Missed\n"
                   "Pass: inline\n"
                   "Name: NoDefinition\n"
                   "Function: foo\n"
                   "DebugLoc: { File: { a: b }, Column: 12, Line: 12 }\n"
                   "",
                   "expected a value of scalar type."));
  // Not a integer column in debug loc.
  EXPECT_TRUE(parseExpectError("\n"
                   "--- !Missed\n"
                   "Pass: inline\n"
                   "Name: NoDefinition\n"
                   "Function: foo\n"
                   "DebugLoc: { File: file.c, Column: { a: b }, Line: 12 }\n"
                   "",
                   "expected a value of scalar type."));
  // Not a integer line in debug loc.
  EXPECT_TRUE(parseExpectError("\n"
                   "--- !Missed\n"
                   "Pass: inline\n"
                   "Name: NoDefinition\n"
                   "Function: foo\n"
                   "DebugLoc: { File: file.c, Column: 12, Line: { a: b } }\n"
                   "",
                   "expected a value of scalar type."));
  // Not a mapping type value for args.
  EXPECT_TRUE(parseExpectError("\n"
                   "--- !Missed\n"
                   "Pass: inline\n"
                   "Name: NoDefinition\n"
                   "Function: foo\n"
                   "DebugLoc: { File: file.c, Column: 12, Line: { a: b } }\n"
                   "",
                   "expected a value of scalar type."));
}

TEST(YAMLRemarks, ParsingWrongArgs) {
  // Multiple debug locs per arg.
  EXPECT_TRUE(parseExpectError("\n"
                   "--- !Missed\n"
                   "Pass: inline\n"
                   "Name: NoDefinition\n"
                   "Function: foo\n"
                   "Args:\n"
                   "  - Str: string\n"
                   "    DebugLoc: { File: a, Line: 1, Column: 2 }\n"
                   "    DebugLoc: { File: a, Line: 1, Column: 2 }\n"
                   "",
                   "only one DebugLoc entry is allowed per argument."));
  // Multiple strings per arg.
  EXPECT_TRUE(parseExpectError("\n"
                   "--- !Missed\n"
                   "Pass: inline\n"
                   "Name: NoDefinition\n"
                   "Function: foo\n"
                   "Args:\n"
                   "  - Str: string\n"
                   "    Str2: string\n"
                   "    DebugLoc: { File: a, Line: 1, Column: 2 }\n"
                   "",
                   "only one string entry is allowed per argument."));
  // No arg value.
  EXPECT_TRUE(parseExpectError("\n"
                   "--- !Missed\n"
                   "Pass: inline\n"
                   "Name: NoDefinition\n"
                   "Function: foo\n"
                   "Args:\n"
                   "  - Callee: ''\n"
                   "  - DebugLoc: { File: a, Line: 1, Column: 2 }\n"
                   "",
                   "argument value is missing."));
  // No arg value.
  EXPECT_TRUE(parseExpectError("\n"
                   "--- !Missed\n"
                   "Pass: inline\n"
                   "Name: NoDefinition\n"
                   "Function: foo\n"
                   "Args:\n"
                   "  - DebugLoc: { File: a, Line: 1, Column: 2 }\n"
                   "",
                   "argument key is missing."));
}

static inline StringRef checkStr(StringRef Str, unsigned ExpectedLen) {
  const char *StrData = Str.data();
  unsigned StrLen = Str.size();
  EXPECT_EQ(StrLen, ExpectedLen);
  return StringRef(StrData, StrLen);
}

TEST(YAMLRemarks, Contents) {
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

  remarks::Parser Parser(Buf);
  Expected<const remarks::Remark *> RemarkOrErr = Parser.getNext();
  EXPECT_FALSE(errorToBool(RemarkOrErr.takeError()));
  EXPECT_TRUE(*RemarkOrErr != nullptr);

  const remarks::Remark &Remark = **RemarkOrErr;
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

  RemarkOrErr = Parser.getNext();
  EXPECT_FALSE(errorToBool(RemarkOrErr.takeError()));
  EXPECT_EQ(*RemarkOrErr, nullptr);
}

static inline StringRef checkStr(LLVMRemarkStringRef Str,
                                 unsigned ExpectedLen) {
  const char *StrData = LLVMRemarkStringGetData(Str);
  unsigned StrLen = LLVMRemarkStringGetLen(Str);
  EXPECT_EQ(StrLen, ExpectedLen);
  return StringRef(StrData, StrLen);
}

TEST(YAMLRemarks, ContentsCAPI) {
  StringRef Buf = "\n"
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
                  "\n";

  LLVMRemarkParserRef Parser =
      LLVMRemarkParserCreateYAML(Buf.data(), Buf.size());
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

  EXPECT_EQ(LLVMRemarkParserGetNext(Parser), nullptr);

  EXPECT_FALSE(LLVMRemarkParserHasError(Parser));
  LLVMRemarkParserDispose(Parser);
}

TEST(YAMLRemarks, ContentsStrTab) {
  StringRef Buf = "\n"
                  "--- !Missed\n"
                  "Pass: 0\n"
                  "Name: 1\n"
                  "DebugLoc: { File: 2, Line: 3, Column: 12 }\n"
                  "Function: 3\n"
                  "Hotness: 4\n"
                  "Args:\n"
                  "  - Callee: 5\n"
                  "  - String: 7\n"
                  "  - Caller: 3\n"
                  "    DebugLoc: { File: 2, Line: 2, Column: 0 }\n"
                  "  - String: 8\n"
                  "\n";

  StringRef StrTabBuf =
      StringRef("inline\0NoDefinition\0file.c\0foo\0Callee\0bar\0String\0 "
                "will not be inlined into \0 because its definition is "
                "unavailable",
                115);

  remarks::Parser Parser(Buf, StrTabBuf);
  Expected<const remarks::Remark *> RemarkOrErr = Parser.getNext();
  EXPECT_FALSE(errorToBool(RemarkOrErr.takeError()));
  EXPECT_TRUE(*RemarkOrErr != nullptr);

  const remarks::Remark &Remark = **RemarkOrErr;
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

  RemarkOrErr = Parser.getNext();
  EXPECT_FALSE(errorToBool(RemarkOrErr.takeError()));
  EXPECT_EQ(*RemarkOrErr, nullptr);
}

TEST(YAMLRemarks, ParsingBadStringTableIndex) {
  StringRef Buf = "\n"
                  "--- !Missed\n"
                  "Pass: 50\n"
                  "\n";

  StringRef StrTabBuf = StringRef("inline");

  remarks::Parser Parser(Buf, StrTabBuf);
  Expected<const remarks::Remark *> Remark = Parser.getNext();
  EXPECT_FALSE(Remark); // Expect an error here.

  std::string ErrorStr;
  raw_string_ostream Stream(ErrorStr);
  handleAllErrors(Remark.takeError(),
                  [&](const ErrorInfoBase &EIB) { EIB.log(Stream); });
  EXPECT_TRUE(
      StringRef(Stream.str())
          .contains("String with index 50 is out of bounds (size = 1)."));
}
