//===- unittest/Support/YAMLRemarksSerializerTest.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Remarks/Remark.h"
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

static void check(const remarks::Remark &R, StringRef ExpectedR,
                  StringRef ExpectedMeta, bool UseStrTab = false,
                  Optional<remarks::StringTable> StrTab = None) {
  std::string Buf;
  raw_string_ostream OS(Buf);
  Expected<std::unique_ptr<remarks::RemarkSerializer>> MaybeS = [&] {
    if (UseStrTab) {
      if (StrTab)
        return createRemarkSerializer(remarks::Format::YAMLStrTab, OS,
                                      std::move(*StrTab));
      else
        return createRemarkSerializer(remarks::Format::YAMLStrTab, OS);
    } else
      return createRemarkSerializer(remarks::Format::YAML, OS);
  }();
  EXPECT_FALSE(errorToBool(MaybeS.takeError()));
  std::unique_ptr<remarks::RemarkSerializer> S = std::move(*MaybeS);

  S->emit(R);
  EXPECT_EQ(OS.str(), ExpectedR);

  Buf.clear();
  std::unique_ptr<remarks::MetaSerializer> MS =
      S->metaSerializer(OS, StringRef(EXTERNALFILETESTPATH));
  MS->emit();
  EXPECT_EQ(OS.str(), ExpectedMeta);
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
  check(R,
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
                  "\0\0\0\0\0\0\0\0"
                  EXTERNALFILETESTPATH"\0",
                  38));
}

TEST(YAMLRemarks, SerializerRemarkStrTab) {
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
  check(R,
        "--- !Missed\n"
        "Pass:            0\n"
        "Name:            1\n"
        "DebugLoc:        { File: 3, Line: 3, Column: 4 }\n"
        "Function:        2\n"
        "Hotness:         5\n"
        "Args:\n"
        "  - key:             4\n"
        "  - keydebug:        5\n"
        "    DebugLoc:        { File: 6, Line: 6, Column: 7 }\n"
        "...\n",
        StringRef("REMARKS\0"
                  "\0\0\0\0\0\0\0\0"
                  "\x2d\0\0\0\0\0\0\0"
                  "pass\0name\0func\0path\0value\0valuedebug\0argpath\0"
                  EXTERNALFILETESTPATH"\0",
                  83),
        /*UseStrTab=*/true);
}

TEST(YAMLRemarks, SerializerRemarkParsedStrTab) {
  StringRef StrTab("pass\0name\0func\0path\0value\0valuedebug\0argpath\0", 45);
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
  check(R,
        "--- !Missed\n"
        "Pass:            0\n"
        "Name:            1\n"
        "DebugLoc:        { File: 3, Line: 3, Column: 4 }\n"
        "Function:        2\n"
        "Hotness:         5\n"
        "Args:\n"
        "  - key:             4\n"
        "  - keydebug:        5\n"
        "    DebugLoc:        { File: 6, Line: 6, Column: 7 }\n"
        "...\n",
        StringRef("REMARKS\0"
                  "\0\0\0\0\0\0\0\0"
                  "\x2d\0\0\0\0\0\0\0"
                  "pass\0name\0func\0path\0value\0valuedebug\0argpath\0"
                  EXTERNALFILETESTPATH"\0",
                  83),
        /*UseStrTab=*/true,
        remarks::StringTable(remarks::ParsedStringTable(StrTab)));
}
