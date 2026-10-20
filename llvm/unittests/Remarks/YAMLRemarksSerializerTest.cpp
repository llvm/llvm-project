//===- unittest/Support/YAMLRemarksSerializerTest.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Remarks/Remark.h"
#include "llvm/Remarks/RemarkParser.h"
#include "llvm/Remarks/YAMLRemarkSerializer.h"
#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

// We need to supprt Windows paths as well. In order to have paths with the same
// length, use a different path according to the platform.
#ifdef _WIN32
#define EXTERNALFILETESTPATH "C:/externalfi"
#else
#define EXTERNALFILETESTPATH "/externalfile"
#endif

using namespace llvm;

static void check(remarks::Format SerializerFormat,
                  ArrayRef<remarks::Remark> Rs, StringRef ExpectedR,
                  std::optional<StringRef> ExpectedMeta,
                  std::optional<remarks::StringTable> StrTab = std::nullopt) {
  std::string Buf;
  raw_string_ostream OS(Buf);
  Expected<std::unique_ptr<remarks::RemarkSerializer>> MaybeS = [&] {
    if (StrTab)
      return createRemarkSerializer(SerializerFormat, OS, std::move(*StrTab));
    else
      return createRemarkSerializer(SerializerFormat, OS);
  }();
  EXPECT_FALSE(errorToBool(MaybeS.takeError()));
  std::unique_ptr<remarks::RemarkSerializer> S = std::move(*MaybeS);

  for (const remarks::Remark &R : Rs)
    S->emit(R);
  S->finalize();
  EXPECT_EQ(OS.str(), ExpectedR);

  if (ExpectedMeta) {
    Buf.clear();
    std::unique_ptr<remarks::MetaSerializer> MS =
        S->metaSerializer(OS, StringRef(EXTERNALFILETESTPATH));
    MS->emit();
    EXPECT_EQ(OS.str(), *ExpectedMeta);
  }
}

static void check(remarks::Format SerializerFormat, const remarks::Remark &R,
                  StringRef ExpectedR, StringRef ExpectedMeta,
                  std::optional<remarks::StringTable> StrTab = std::nullopt) {
  return check(SerializerFormat, ArrayRef(&R, &R + 1), ExpectedR, ExpectedMeta,
               std::move(StrTab));
}

static void
checkStandalone(remarks::Format SerializerFormat, const remarks::Remark &R,
                StringRef ExpectedR,
                std::optional<remarks::StringTable> StrTab = std::nullopt) {
  return check(SerializerFormat, ArrayRef(&R, &R + 1), ExpectedR,
               /*ExpectedMeta=*/std::nullopt, std::move(StrTab));
}

TEST(YAMLRemarks, SerializerRemark) {
  remarks::Remark R;
  R.RemarkType = remarks::Type::Missed;
  R.PassName = "pass";
  R.RemarkName = "name";
  R.FunctionName = "func";
  R.Loc = remarks::RemarkLocation{"path", 3, 4};
  R.Hotness = 5;
  R.Args.emplace_back();
  R.Args.back().Key = "key";
  R.Args.back().Val = "value";
  R.Args.emplace_back();
  R.Args.back().Key = "keydebug";
  R.Args.back().Val = "valuedebug";
  R.Args.back().Loc = remarks::RemarkLocation{"argpath", 6, 7};
  check(remarks::Format::YAML, R,
        "--- !Missed\n"
        "Pass:            pass\n"
        "Name:            name\n"
        "DebugLoc:        { File: path, Line: 3, Column: 4 }\n"
        "Function:        func\n"
        "Hotness:         5\n"
        "Args:\n"
        "  - key:             value\n"
        "  - keydebug:        valuedebug\n"
        "    DebugLoc:        { File: argpath, Line: 6, Column: 7 }\n"
        "...\n",
        StringRef("REMARKS\0"
                  "\0\0\0\0\0\0\0\0"
                  "\0\0\0\0\0\0\0\0" EXTERNALFILETESTPATH "\0",
                  38));
}

TEST(YAMLRemarks, SerializerRemarkStandalone) {
  remarks::Remark R;
  R.RemarkType = remarks::Type::Missed;
  R.PassName = "pass";
  R.RemarkName = "name";
  R.FunctionName = "func";
  R.Loc = remarks::RemarkLocation{"path", 3, 4};
  R.Hotness = 5;
  R.Args.emplace_back();
  R.Args.back().Key = "key";
  R.Args.back().Val = "value";
  R.Args.emplace_back();
  R.Args.back().Key = "keydebug";
  R.Args.back().Val = "valuedebug";
  R.Args.back().Loc = remarks::RemarkLocation{"argpath", 6, 7};
  checkStandalone(
      remarks::Format::YAML, R,
      StringRef("--- !Missed\n"
                "Pass:            pass\n"
                "Name:            name\n"
                "DebugLoc:        { File: path, Line: 3, Column: 4 }\n"
                "Function:        func\n"
                "Hotness:         5\n"
                "Args:\n"
                "  - key:             value\n"
                "  - keydebug:        valuedebug\n"
                "    DebugLoc:        { File: argpath, Line: 6, Column: 7 }\n"
                "...\n"));
}

TEST(YAMLRemarks, SerializerRemarkParsedStrTabStandaloneNoStrTab) {
  // Check that we don't use the string table even if it was provided.
  StringRef StrTab("pass\0name\0func\0path\0value\0valuedebug\0argpath\0", 45);
  remarks::ParsedStringTable ParsedStrTab(StrTab);
  remarks::StringTable PreFilledStrTab(ParsedStrTab);
  remarks::Remark R;
  R.RemarkType = remarks::Type::Missed;
  R.PassName = "pass";
  R.RemarkName = "name";
  R.FunctionName = "func";
  R.Loc = remarks::RemarkLocation{"path", 3, 4};
  R.Hotness = 5;
  R.Args.emplace_back();
  R.Args.back().Key = "key";
  R.Args.back().Val = "value";
  R.Args.emplace_back();
  R.Args.back().Key = "keydebug";
  R.Args.back().Val = "valuedebug";
  R.Args.back().Loc = remarks::RemarkLocation{"argpath", 6, 7};
  checkStandalone(
      remarks::Format::YAML, R,
      StringRef("--- !Missed\n"
                "Pass:            pass\n"
                "Name:            name\n"
                "DebugLoc:        { File: path, Line: 3, Column: 4 }\n"
                "Function:        func\n"
                "Hotness:         5\n"
                "Args:\n"
                "  - key:             value\n"
                "  - keydebug:        valuedebug\n"
                "    DebugLoc:        { File: argpath, Line: 6, Column: 7 }\n"
                "...\n"),
      std::move(PreFilledStrTab));
}

TEST(YAMLRemarks, SerializerRemarkStringRefOOBRead) {
  remarks::Remark R;
  R.RemarkType = remarks::Type::Missed;
  R.PassName = StringRef("passAAAA", 4);
  R.RemarkName = StringRef("nameAAAA", 4);
  R.FunctionName = StringRef("funcAAAA", 4);
  R.Loc = remarks::RemarkLocation{StringRef("pathAAAA", 4), 3, 4};
  R.Hotness = 5;
  R.Args.emplace_back();
  R.Args.back().Key = StringRef("keyAAAA", 3);
  R.Args.back().Val = StringRef("valueAAAA", 5);
  R.Args.emplace_back();
  R.Args.back().Key = StringRef("keydebugAAAA", 8);
  R.Args.back().Val = StringRef("valuedebugAAAA", 10);
  R.Args.back().Loc =
      remarks::RemarkLocation{StringRef("argpathAAAA", 7), 6, 7};
  checkStandalone(remarks::Format::YAML, R,
                  "--- !Missed\n"
                  "Pass:            pass\n"
                  "Name:            name\n"
                  "DebugLoc:        { File: path, Line: 3, Column: 4 }\n"
                  "Function:        func\n"
                  "Hotness:         5\n"
                  "Args:\n"
                  "  - key:             value\n"
                  "  - keydebug:        valuedebug\n"
                  "    DebugLoc:        { File: argpath, Line: 6, Column: 7 }\n"
                  "...\n");
}
