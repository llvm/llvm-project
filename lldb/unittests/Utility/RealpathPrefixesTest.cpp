//===-- RealpathPrefixesTest.cpp
//--------------------------------------------------===//
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

// Should resolve a symlink which match an absolute prefix
TEST(RealpathPrefixesTest, MatchingAbsolutePrefix) {
  // Prepare FS
  llvm::IntrusiveRefCntPtr<MockSymlinkFileSystem> fs(new MockSymlinkFileSystem(
      PosixSpec("/dir1/link.h"), PosixSpec("/dir2/real.h"),
      FileSpec::Style::posix));

  // Prepare RealpathPrefixes
  FileSpecList file_spec_list;
  file_spec_list.Append(PosixSpec("/dir1"));
  RealpathPrefixes prefixes(file_spec_list, fs);

  // Test
  std::optional<FileSpec> ret =
      prefixes.ResolveSymlinks(PosixSpec("/dir1/link.h"));
  EXPECT_EQ(ret, PosixSpec("/dir2/real.h"));
}

// Should resolve a symlink which match a relative prefix
TEST(RealpathPrefixesTest, MatchingRelativePrefix) {
  // Prepare FS
  llvm::IntrusiveRefCntPtr<MockSymlinkFileSystem> fs(new MockSymlinkFileSystem(
      PosixSpec("dir1/link.h"), PosixSpec("dir2/real.h"),
      FileSpec::Style::posix));

  // Prepare RealpathPrefixes
  FileSpecList file_spec_list;
  file_spec_list.Append(PosixSpec("dir1"));
  RealpathPrefixes prefixes(file_spec_list, fs);

  // Test
  std::optional<FileSpec> ret =
      prefixes.ResolveSymlinks(PosixSpec("dir1/link.h"));
  EXPECT_EQ(ret, PosixSpec("dir2/real.h"));
}

// Should resolve in Windows and/or with a case-insensitive support file
TEST(RealpathPrefixesTest, WindowsAndCaseInsensitive) {
  // Prepare FS
  llvm::IntrusiveRefCntPtr<MockSymlinkFileSystem> fs(new MockSymlinkFileSystem(
      WindowsSpec("f:\\dir1\\link.h"), WindowsSpec("f:\\dir2\\real.h"),
      FileSpec::Style::windows));

  // Prepare RealpathPrefixes
  FileSpecList file_spec_list;
  file_spec_list.Append(WindowsSpec("f:\\dir1"));
  RealpathPrefixes prefixes(file_spec_list, fs);

  // Test
  std::optional<FileSpec> ret =
      prefixes.ResolveSymlinks(WindowsSpec("F:\\DIR1\\LINK.H"));
  EXPECT_EQ(ret, WindowsSpec("f:\\dir2\\real.h"));
}

// Should resolve a symlink when there is mixture of matching and mismatching
// prefixex
TEST(RealpathPrefixesTest, MatchingAndMismatchingPrefix) {
  // Prepare FS
  llvm::IntrusiveRefCntPtr<MockSymlinkFileSystem> fs(new MockSymlinkFileSystem(
      PosixSpec("/dir1/link.h"), PosixSpec("/dir2/real.h"),
      FileSpec::Style::posix));

  // Prepare RealpathPrefixes
  FileSpecList file_spec_list;
  file_spec_list.Append(PosixSpec("/fake/path1"));
  file_spec_list.Append(PosixSpec("/dir1")); // Matching prefix
  file_spec_list.Append(PosixSpec("/fake/path2"));
  RealpathPrefixes prefixes(file_spec_list, fs);

  // Test
  std::optional<FileSpec> ret =
      prefixes.ResolveSymlinks(PosixSpec("/dir1/link.h"));
  EXPECT_EQ(ret, PosixSpec("/dir2/real.h"));
}

// Should resolve a symlink when the prefixes matches after normalization
TEST(RealpathPrefixesTest, ComplexPrefixes) {
  // Prepare FS
  llvm::IntrusiveRefCntPtr<MockSymlinkFileSystem> fs(new MockSymlinkFileSystem(
      PosixSpec("dir1/link.h"), PosixSpec("dir2/real.h"),
      FileSpec::Style::posix));

  // Prepare RealpathPrefixes
  FileSpecList file_spec_list;
  file_spec_list.Append(
      PosixSpec("./dir1/foo/../bar/..")); // Equivalent to "/dir1"
  RealpathPrefixes prefixes(file_spec_list, fs);

  // Test
  std::optional<FileSpec> ret =
      prefixes.ResolveSymlinks(PosixSpec("dir1/link.h"));
  EXPECT_EQ(ret, PosixSpec("dir2/real.h"));
}

// Should not resolve a symlink which doesn't match any prefixes
TEST(RealpathPrefixesTest, MismatchingPrefixes) {
  // Prepare FS
  llvm::IntrusiveRefCntPtr<MockSymlinkFileSystem> fs(new MockSymlinkFileSystem(
      PosixSpec("/dir1/link.h"), PosixSpec("/dir2/real.h"),
      FileSpec::Style::posix));

  // Prepare RealpathPrefixes
  FileSpecList file_spec_list;
  file_spec_list.Append(PosixSpec("/dir3"));
  RealpathPrefixes prefixes(file_spec_list, fs);

  // Test
  std::optional<FileSpec> ret =
      prefixes.ResolveSymlinks(PosixSpec("/dir1/link.h"));
  EXPECT_EQ(ret, std::nullopt);
}

// Should not resolve a realpath
TEST(RealpathPrefixesTest, Realpath) {
  // Prepare FS
  llvm::IntrusiveRefCntPtr<MockSymlinkFileSystem> fs(
      new MockSymlinkFileSystem());

  // Prepare RealpathPrefixes
  FileSpecList file_spec_list;
  file_spec_list.Append(PosixSpec("/symlink_dir"));
  RealpathPrefixes prefixes(file_spec_list, fs);

  // Test
  std::optional<FileSpec> ret =
      prefixes.ResolveSymlinks(PosixSpec("/dir/real.h"));
  EXPECT_EQ(ret, std::nullopt);
}
