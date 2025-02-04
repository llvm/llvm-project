//===-- FileSpecListTest.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "MockSymlinkFileSystem.h"
#include "lldb/Utility/FileSpecList.h"
#include "lldb/Utility/RealpathPrefixes.h"

using namespace lldb_private;

static FileSpec PosixSpec(llvm::StringRef path) {
  return FileSpec(path, FileSpec::Style::posix);
}

static FileSpec WindowsSpec(llvm::StringRef path) {
  return FileSpec(path, FileSpec::Style::windows);
}

TEST(SupportFileListTest, RelativePathMatchesPosix) {

  const FileSpec fullpath = PosixSpec("/build/src/main.cpp");
  const FileSpec relative = PosixSpec("./src/main.cpp");
  const FileSpec basename = PosixSpec("./main.cpp");
  const FileSpec full_wrong = PosixSpec("/other/wrong/main.cpp");
  const FileSpec rel_wrong = PosixSpec("./wrong/main.cpp");
  // Make sure these don't match "src/main.cpp" as we want to match full
  // directories only
  const FileSpec rel2_wrong = PosixSpec("asrc/main.cpp");
  const FileSpec rel3_wrong = PosixSpec("rc/main.cpp");

  SupportFileList files;
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

TEST(SupportFileListTest, RelativePathMatchesWindows) {

  const FileSpec fullpath = WindowsSpec(R"(C:\build\src\main.cpp)");
  const FileSpec relative = WindowsSpec(R"(.\src\main.cpp)");
  const FileSpec basename = WindowsSpec(R"(.\main.cpp)");
  const FileSpec full_wrong = WindowsSpec(R"(\other\wrong\main.cpp)");
  const FileSpec rel_wrong = WindowsSpec(R"(.\wrong\main.cpp)");
  // Make sure these don't match "src\main.cpp" as we want to match full
  // directories only
  const FileSpec rel2_wrong = WindowsSpec(R"(asrc\main.cpp)");
  const FileSpec rel3_wrong = WindowsSpec(R"("rc\main.cpp)");

  SupportFileList files;
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

// Support file is a symlink to the breakpoint file.
// Absolute paths are used.
// A matching prefix is set.
// Should find it compatible.
TEST(SupportFileListTest, SymlinkedAbsolutePaths) {
  // Prepare FS
  llvm::IntrusiveRefCntPtr<MockSymlinkFileSystem> fs(new MockSymlinkFileSystem(
      PosixSpec("/symlink_dir/foo.h"), PosixSpec("/real_dir/foo.h"),
      FileSpec::Style::posix));

  // Prepare RealpathPrefixes
  FileSpecList file_spec_list;
  file_spec_list.Append(PosixSpec("/symlink_dir"));
  RealpathPrefixes prefixes(file_spec_list, fs);

  // Prepare support file list
  SupportFileList support_file_list;
  support_file_list.Append(PosixSpec("/symlink_dir/foo.h"));

  // Test
  size_t ret = support_file_list.FindCompatibleIndex(
      0, PosixSpec("/real_dir/foo.h"), &prefixes);
  EXPECT_EQ(ret, (size_t)0);
}

// Support file is a symlink to the breakpoint file.
// Absolute paths are used.
// A matching prefix is set, which is the root directory.
// Should find it compatible.
TEST(SupportFileListTest, RootDirectory) {
  // Prepare FS
  llvm::IntrusiveRefCntPtr<MockSymlinkFileSystem> fs(new MockSymlinkFileSystem(
      PosixSpec("/symlink_dir/foo.h"), PosixSpec("/real_dir/foo.h"),
      FileSpec::Style::posix));

  // Prepare RealpathPrefixes
  FileSpecList file_spec_list;
  file_spec_list.Append(PosixSpec("/"));
  RealpathPrefixes prefixes(file_spec_list, fs);

  // Prepare support file list
  SupportFileList support_file_list;
  support_file_list.Append(PosixSpec("/symlink_dir/foo.h"));

  // Test
  size_t ret = support_file_list.FindCompatibleIndex(
      0, PosixSpec("/real_dir/foo.h"), &prefixes);
  EXPECT_EQ(ret, (size_t)0);
}

// Support file is a symlink to the breakpoint file.
// Relative paths are used.
// A matching prefix is set.
// Should find it compatible.
TEST(SupportFileListTest, SymlinkedRelativePaths) {
  // Prepare FS
  llvm::IntrusiveRefCntPtr<MockSymlinkFileSystem> fs(new MockSymlinkFileSystem(
      PosixSpec("symlink_dir/foo.h"), PosixSpec("real_dir/foo.h"),
      FileSpec::Style::posix));

  // Prepare RealpathPrefixes
  FileSpecList file_spec_list;
  file_spec_list.Append(PosixSpec("symlink_dir"));
  RealpathPrefixes prefixes(file_spec_list, fs);

  // Prepare support file list
  SupportFileList support_file_list;
  support_file_list.Append(PosixSpec("symlink_dir/foo.h"));

  // Test
  size_t ret = support_file_list.FindCompatibleIndex(
      0, PosixSpec("real_dir/foo.h"), &prefixes);
  EXPECT_EQ(ret, (size_t)0);
}

// Support file is a symlink to the breakpoint file.
// A matching prefix is set.
// Input file only match basename and not directory.
// Should find it incompatible.
TEST(SupportFileListTest, RealpathOnlyMatchFileName) {
  // Prepare FS
  llvm::IntrusiveRefCntPtr<MockSymlinkFileSystem> fs(new MockSymlinkFileSystem(
      PosixSpec("symlink_dir/foo.h"), PosixSpec("real_dir/foo.h"),
      FileSpec::Style::posix));

  // Prepare RealpathPrefixes
  FileSpecList file_spec_list;
  file_spec_list.Append(PosixSpec("symlink_dir"));
  RealpathPrefixes prefixes(file_spec_list, fs);

  // Prepare support file list
  SupportFileList support_file_list;
  support_file_list.Append(PosixSpec("symlink_dir/foo.h"));

  // Test
  size_t ret = support_file_list.FindCompatibleIndex(
      0, PosixSpec("some_other_dir/foo.h"), &prefixes);
  EXPECT_EQ(ret, UINT32_MAX);
}

// Support file is a symlink to the breakpoint file.
// A prefix is set, which is a matching string prefix, but not a path prefix.
// Should find it incompatible.
TEST(SupportFileListTest, DirectoryMatchStringPrefixButNotWholeDirectoryName) {
  // Prepare FS
  llvm::IntrusiveRefCntPtr<MockSymlinkFileSystem> fs(new MockSymlinkFileSystem(
      PosixSpec("symlink_dir/foo.h"), PosixSpec("real_dir/foo.h"),
      FileSpec::Style::posix));

  // Prepare RealpathPrefixes
  FileSpecList file_spec_list;
  file_spec_list.Append(PosixSpec("symlink")); // This is a string prefix of the
                                               // symlink but not a path prefix.
  RealpathPrefixes prefixes(file_spec_list, fs);

  // Prepare support file list
  SupportFileList support_file_list;
  support_file_list.Append(PosixSpec("symlink_dir/foo.h"));

  // Test
  size_t ret = support_file_list.FindCompatibleIndex(
      0, PosixSpec("real_dir/foo.h"), &prefixes);
  EXPECT_EQ(ret, UINT32_MAX);
}

// Support file is a symlink to the breakpoint file.
// A matching prefix is set.
// However, the breakpoint is set with a partial path.
// Should find it compatible.
TEST(SupportFileListTest, PartialBreakpointPath) {
  // Prepare FS
  llvm::IntrusiveRefCntPtr<MockSymlinkFileSystem> fs(new MockSymlinkFileSystem(
      PosixSpec("symlink_dir/foo.h"), PosixSpec("/real_dir/foo.h"),
      FileSpec::Style::posix));

  // Prepare RealpathPrefixes
  FileSpecList file_spec_list;
  file_spec_list.Append(PosixSpec("symlink_dir"));
  RealpathPrefixes prefixes(file_spec_list, fs);

  // Prepare support file list
  SupportFileList support_file_list;
  support_file_list.Append(PosixSpec("symlink_dir/foo.h"));

  // Test
  size_t ret = support_file_list.FindCompatibleIndex(
      0, PosixSpec("real_dir/foo.h"), &prefixes);
  EXPECT_EQ(ret, (size_t)0);
}

// Support file is a symlink to the breakpoint file.
// A matching prefix is set.
// However, the basename is different between the symlink and its target.
// Should find it incompatible.
TEST(SupportFileListTest, DifferentBasename) {
  // Prepare FS
  llvm::IntrusiveRefCntPtr<MockSymlinkFileSystem> fs(new MockSymlinkFileSystem(
      PosixSpec("/symlink_dir/foo.h"), PosixSpec("/real_dir/bar.h"),
      FileSpec::Style::posix));

  // Prepare RealpathPrefixes
  FileSpecList file_spec_list;
  file_spec_list.Append(PosixSpec("/symlink_dir"));
  RealpathPrefixes prefixes(file_spec_list, fs);

  // Prepare support file list
  SupportFileList support_file_list;
  support_file_list.Append(PosixSpec("/symlink_dir/foo.h"));

  // Test
  size_t ret = support_file_list.FindCompatibleIndex(
      0, PosixSpec("real_dir/bar.h"), &prefixes);
  EXPECT_EQ(ret, UINT32_MAX);
}

// No prefixes are configured.
// The support file and the breakpoint file are different.
// Should find it incompatible.
TEST(SupportFileListTest, NoPrefixes) {
  // Prepare support file list
  SupportFileList support_file_list;
  support_file_list.Append(PosixSpec("/real_dir/bar.h"));

  // Test
  size_t ret = support_file_list.FindCompatibleIndex(
      0, PosixSpec("/real_dir/foo.h"), nullptr);
  EXPECT_EQ(ret, UINT32_MAX);
}

// No prefixes are configured.
// The support file and the breakpoint file are the same.
// Should find it compatible.
TEST(SupportFileListTest, SameFile) {
  // Prepare support file list
  SupportFileList support_file_list;
  support_file_list.Append(PosixSpec("/real_dir/foo.h"));

  // Test
  size_t ret = support_file_list.FindCompatibleIndex(
      0, PosixSpec("/real_dir/foo.h"), nullptr);
  EXPECT_EQ(ret, (size_t)0);
}
