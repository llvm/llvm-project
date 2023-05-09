//===-- FileSpecListTest.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Utility/FileSpecList.h"

using namespace lldb_private;

static FileSpec PosixSpec(llvm::StringRef path) {
  return FileSpec(path, FileSpec::Style::posix);
}

static FileSpec WindowsSpec(llvm::StringRef path) {
  return FileSpec(path, FileSpec::Style::windows);
}

TEST(FileSpecListTest, RelativePathMatchesPosix) {

  const FileSpec fullpath = PosixSpec("/build/src/main.cpp");
  const FileSpec relative = PosixSpec("./src/main.cpp");
  const FileSpec basename = PosixSpec("./main.cpp");
  const FileSpec full_wrong = PosixSpec("/other/wrong/main.cpp");
  const FileSpec rel_wrong = PosixSpec("./wrong/main.cpp");
  // Make sure these don't match "src/main.cpp" as we want to match full
  // directories only
  const FileSpec rel2_wrong = PosixSpec("asrc/main.cpp");
  const FileSpec rel3_wrong = PosixSpec("rc/main.cpp");

  FileSpecList files;
  files.Append(fullpath);
  files.Append(relative);
  files.Append(basename);
  files.Append(full_wrong);
  files.Append(rel_wrong);
  files.Append(rel2_wrong);
  files.Append(rel3_wrong);

  // Make sure the full path only matches the first entry
  EXPECT_EQ((size_t)0, files.FindCompatibleIndex(0, fullpath));
  EXPECT_EQ((size_t)1, files.FindCompatibleIndex(1, fullpath));
  EXPECT_EQ((size_t)2, files.FindCompatibleIndex(2, fullpath));
  EXPECT_EQ((size_t)UINT32_MAX, files.FindCompatibleIndex(3, fullpath));
  // Make sure the relative path matches the all of the entries that contain
  // the relative path
  EXPECT_EQ((size_t)0, files.FindCompatibleIndex(0, relative));
  EXPECT_EQ((size_t)1, files.FindCompatibleIndex(1, relative));
  EXPECT_EQ((size_t)2, files.FindCompatibleIndex(2, relative));
  EXPECT_EQ((size_t)UINT32_MAX, files.FindCompatibleIndex(3, relative));

  // Make sure looking file a file using the basename matches all entries
  EXPECT_EQ((size_t)0, files.FindCompatibleIndex(0, basename));
  EXPECT_EQ((size_t)1, files.FindCompatibleIndex(1, basename));
  EXPECT_EQ((size_t)2, files.FindCompatibleIndex(2, basename));
  EXPECT_EQ((size_t)3, files.FindCompatibleIndex(3, basename));
  EXPECT_EQ((size_t)4, files.FindCompatibleIndex(4, basename));
  EXPECT_EQ((size_t)5, files.FindCompatibleIndex(5, basename));
  EXPECT_EQ((size_t)6, files.FindCompatibleIndex(6, basename));

  // Make sure that paths that have a common suffix don't return values that
  // don't match on directory delimiters.
  EXPECT_EQ((size_t)2, files.FindCompatibleIndex(0, rel2_wrong));
  EXPECT_EQ((size_t)5, files.FindCompatibleIndex(3, rel2_wrong));
  EXPECT_EQ((size_t)UINT32_MAX, files.FindCompatibleIndex(6, rel2_wrong));

  EXPECT_EQ((size_t)2, files.FindCompatibleIndex(0, rel3_wrong));
  EXPECT_EQ((size_t)6, files.FindCompatibleIndex(3, rel3_wrong));
}

TEST(FileSpecListTest, RelativePathMatchesWindows) {

  const FileSpec fullpath = WindowsSpec(R"(C:\build\src\main.cpp)");
  const FileSpec relative = WindowsSpec(R"(.\src\main.cpp)");
  const FileSpec basename = WindowsSpec(R"(.\main.cpp)");
  const FileSpec full_wrong = WindowsSpec(R"(\other\wrong\main.cpp)");
  const FileSpec rel_wrong = WindowsSpec(R"(.\wrong\main.cpp)");
  // Make sure these don't match "src\main.cpp" as we want to match full
  // directories only
  const FileSpec rel2_wrong = WindowsSpec(R"(asrc\main.cpp)");
  const FileSpec rel3_wrong = WindowsSpec(R"("rc\main.cpp)");

  FileSpecList files;
  files.Append(fullpath);
  files.Append(relative);
  files.Append(basename);
  files.Append(full_wrong);
  files.Append(rel_wrong);
  files.Append(rel2_wrong);
  files.Append(rel3_wrong);

  // Make sure the full path only matches the first entry
  EXPECT_EQ((size_t)0, files.FindCompatibleIndex(0, fullpath));
  EXPECT_EQ((size_t)1, files.FindCompatibleIndex(1, fullpath));
  EXPECT_EQ((size_t)2, files.FindCompatibleIndex(2, fullpath));
  EXPECT_EQ((size_t)UINT32_MAX, files.FindCompatibleIndex(3, fullpath));
  // Make sure the relative path matches the all of the entries that contain
  // the relative path
  EXPECT_EQ((size_t)0, files.FindCompatibleIndex(0, relative));
  EXPECT_EQ((size_t)1, files.FindCompatibleIndex(1, relative));
  EXPECT_EQ((size_t)2, files.FindCompatibleIndex(2, relative));
  EXPECT_EQ((size_t)UINT32_MAX, files.FindCompatibleIndex(3, relative));

  // Make sure looking file a file using the basename matches all entries
  EXPECT_EQ((size_t)0, files.FindCompatibleIndex(0, basename));
  EXPECT_EQ((size_t)1, files.FindCompatibleIndex(1, basename));
  EXPECT_EQ((size_t)2, files.FindCompatibleIndex(2, basename));
  EXPECT_EQ((size_t)3, files.FindCompatibleIndex(3, basename));
  EXPECT_EQ((size_t)4, files.FindCompatibleIndex(4, basename));
  EXPECT_EQ((size_t)5, files.FindCompatibleIndex(5, basename));
  EXPECT_EQ((size_t)6, files.FindCompatibleIndex(6, basename));

  // Make sure that paths that have a common suffix don't return values that
  // don't match on directory delimiters.
  EXPECT_EQ((size_t)2, files.FindCompatibleIndex(0, rel2_wrong));
  EXPECT_EQ((size_t)5, files.FindCompatibleIndex(3, rel2_wrong));
  EXPECT_EQ((size_t)UINT32_MAX, files.FindCompatibleIndex(6, rel2_wrong));

  EXPECT_EQ((size_t)2, files.FindCompatibleIndex(0, rel3_wrong));
  EXPECT_EQ((size_t)6, files.FindCompatibleIndex(3, rel3_wrong));
}
