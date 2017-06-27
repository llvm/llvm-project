//===-- StructuredDataTest.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Utility/Status.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/Utility/StructuredData.h"
#include "llvm/Support/Path.h"

extern const char *TestMainArgv0;

using namespace lldb;
using namespace lldb_private;

namespace {

class StructuredDataTest : public testing::Test {
public:
  static void SetUpTestCase() {
    s_inputs_folder = llvm::sys::path::parent_path(TestMainArgv0);
    llvm::sys::path::append(s_inputs_folder, "Inputs");
  }

protected:
  static llvm::SmallString<128> s_inputs_folder;
};
} // namespace

llvm::SmallString<128> StructuredDataTest::s_inputs_folder;

TEST_F(StructuredDataTest, StringDump) {
  std::pair<llvm::StringRef, llvm::StringRef> TestCases[] = {
      {R"(asdfg)", R"("asdfg")"},
      {R"(as"df)", R"("as\"df")"},
      {R"(as\df)", R"("as\\df")"},
  };
  for (auto P : TestCases) {
    StreamString S;
    const bool pretty_print = false;
    StructuredData::String(P.first).Dump(S, pretty_print);
    EXPECT_EQ(P.second, S.GetString());
  }
}

TEST_F(StructuredDataTest, ParseJSONFromFile) {
  Status status;
  auto object_sp = StructuredData::ParseJSONFromFile(
      FileSpec("non-existing-file.json", false), status);
  EXPECT_EQ(nullptr, object_sp);

  llvm::SmallString<128> input = s_inputs_folder;
  llvm::sys::path::append(input, "StructuredData-basic.json");
  object_sp = StructuredData::ParseJSONFromFile(FileSpec(input, false), status);
  ASSERT_NE(nullptr, object_sp);

  StreamString S;
  object_sp->Dump(S, false);
  EXPECT_EQ("[1,2,3]", S.GetString());
}
