//===-- FileSpecTest.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Host/FileSpec.h"

using namespace lldb_private;

TEST(FileSpecTest, FileAndDirectoryComponents) {
  FileSpec fs_posix("/foo/bar", false, FileSpec::ePathSyntaxPosix);
  EXPECT_STREQ("/foo/bar", fs_posix.GetCString());
  EXPECT_STREQ("/foo", fs_posix.GetDirectory().GetCString());
  EXPECT_STREQ("bar", fs_posix.GetFilename().GetCString());

  FileSpec fs_windows("F:\\bar", false, FileSpec::ePathSyntaxWindows);
  EXPECT_STREQ("F:\\bar", fs_windows.GetCString());
  // EXPECT_STREQ("F:\\", fs_windows.GetDirectory().GetCString()); // It returns
  // "F:/"
  EXPECT_STREQ("bar", fs_windows.GetFilename().GetCString());

  FileSpec fs_posix_root("/", false, FileSpec::ePathSyntaxPosix);
  EXPECT_STREQ("/", fs_posix_root.GetCString());
  EXPECT_EQ(nullptr, fs_posix_root.GetDirectory().GetCString());
  EXPECT_STREQ("/", fs_posix_root.GetFilename().GetCString());

  FileSpec fs_windows_drive("F:", false, FileSpec::ePathSyntaxWindows);
  EXPECT_STREQ("F:", fs_windows_drive.GetCString());
  EXPECT_EQ(nullptr, fs_windows_drive.GetDirectory().GetCString());
  EXPECT_STREQ("F:", fs_windows_drive.GetFilename().GetCString());

  FileSpec fs_windows_root("F:\\", false, FileSpec::ePathSyntaxWindows);
  EXPECT_STREQ("F:\\", fs_windows_root.GetCString());
  EXPECT_STREQ("F:", fs_windows_root.GetDirectory().GetCString());
  // EXPECT_STREQ("\\", fs_windows_root.GetFilename().GetCString()); // It
  // returns "/"

  FileSpec fs_posix_long("/foo/bar/baz", false, FileSpec::ePathSyntaxPosix);
  EXPECT_STREQ("/foo/bar/baz", fs_posix_long.GetCString());
  EXPECT_STREQ("/foo/bar", fs_posix_long.GetDirectory().GetCString());
  EXPECT_STREQ("baz", fs_posix_long.GetFilename().GetCString());

  FileSpec fs_windows_long("F:\\bar\\baz", false, FileSpec::ePathSyntaxWindows);
  EXPECT_STREQ("F:\\bar\\baz", fs_windows_long.GetCString());
  // EXPECT_STREQ("F:\\bar", fs_windows_long.GetDirectory().GetCString()); // It
  // returns "F:/bar"
  EXPECT_STREQ("baz", fs_windows_long.GetFilename().GetCString());

  FileSpec fs_posix_trailing_slash("/foo/bar/", false,
                                   FileSpec::ePathSyntaxPosix);
  EXPECT_STREQ("/foo/bar/.", fs_posix_trailing_slash.GetCString());
  EXPECT_STREQ("/foo/bar", fs_posix_trailing_slash.GetDirectory().GetCString());
  EXPECT_STREQ(".", fs_posix_trailing_slash.GetFilename().GetCString());

  FileSpec fs_windows_trailing_slash("F:\\bar\\", false,
                                     FileSpec::ePathSyntaxWindows);
  EXPECT_STREQ("F:\\bar\\.", fs_windows_trailing_slash.GetCString());
  // EXPECT_STREQ("F:\\bar",
  // fs_windows_trailing_slash.GetDirectory().GetCString()); // It returns
  // "F:/bar"
  EXPECT_STREQ(".", fs_windows_trailing_slash.GetFilename().GetCString());
}

TEST(FileSpecTest, AppendPathComponent) {
  FileSpec fs_posix("/foo", false, FileSpec::ePathSyntaxPosix);
  fs_posix.AppendPathComponent("bar");
  EXPECT_STREQ("/foo/bar", fs_posix.GetCString());
  EXPECT_STREQ("/foo", fs_posix.GetDirectory().GetCString());
  EXPECT_STREQ("bar", fs_posix.GetFilename().GetCString());

  FileSpec fs_posix_2("/foo", false, FileSpec::ePathSyntaxPosix);
  fs_posix_2.AppendPathComponent("//bar/baz");
  EXPECT_STREQ("/foo/bar/baz", fs_posix_2.GetCString());
  EXPECT_STREQ("/foo/bar", fs_posix_2.GetDirectory().GetCString());
  EXPECT_STREQ("baz", fs_posix_2.GetFilename().GetCString());

  FileSpec fs_windows("F:\\bar", false, FileSpec::ePathSyntaxWindows);
  fs_windows.AppendPathComponent("baz");
  EXPECT_STREQ("F:\\bar\\baz", fs_windows.GetCString());
  // EXPECT_STREQ("F:\\bar", fs_windows.GetDirectory().GetCString()); // It
  // returns "F:/bar"
  EXPECT_STREQ("baz", fs_windows.GetFilename().GetCString());

  FileSpec fs_posix_root("/", false, FileSpec::ePathSyntaxPosix);
  fs_posix_root.AppendPathComponent("bar");
  EXPECT_STREQ("/bar", fs_posix_root.GetCString());
  EXPECT_STREQ("/", fs_posix_root.GetDirectory().GetCString());
  EXPECT_STREQ("bar", fs_posix_root.GetFilename().GetCString());

  FileSpec fs_windows_root("F:\\", false, FileSpec::ePathSyntaxWindows);
  fs_windows_root.AppendPathComponent("bar");
  EXPECT_STREQ("F:\\bar", fs_windows_root.GetCString());
  // EXPECT_STREQ("F:\\", fs_windows_root.GetDirectory().GetCString()); // It
  // returns "F:/"
  EXPECT_STREQ("bar", fs_windows_root.GetFilename().GetCString());
}

TEST(FileSpecTest, CopyByAppendingPathComponent) {
  FileSpec fs = FileSpec("/foo", false, FileSpec::ePathSyntaxPosix)
                    .CopyByAppendingPathComponent("bar");
  EXPECT_STREQ("/foo/bar", fs.GetCString());
  EXPECT_STREQ("/foo", fs.GetDirectory().GetCString());
  EXPECT_STREQ("bar", fs.GetFilename().GetCString());
}

static void Compare(const FileSpec &one, const FileSpec &two, bool full_match,
                    bool remove_backup_dots, bool result) {
  EXPECT_EQ(result, FileSpec::Equal(one, two, full_match, remove_backup_dots))
      << "File one: " << one.GetCString() << "\nFile two: " << two.GetCString()
      << "\nFull match: " << full_match
      << "\nRemove backup dots: " << remove_backup_dots;
}

TEST(FileSpecTest, EqualSeparator) {
  FileSpec backward("C:\\foo\\bar", false, FileSpec::ePathSyntaxWindows);
  FileSpec forward("C:/foo/bar", false, FileSpec::ePathSyntaxWindows);
  EXPECT_EQ(forward, backward);

  const bool full_match = true;
  const bool remove_backup_dots = true;
  const bool match = true;
  Compare(forward, backward, full_match, remove_backup_dots, match);
  Compare(forward, backward, full_match, !remove_backup_dots, match);
  Compare(forward, backward, !full_match, remove_backup_dots, match);
  Compare(forward, backward, !full_match, !remove_backup_dots, match);
}

#if 0
TEST(FileSpecTest, EqualDotsWindows) {
  const bool full_match = true;
  const bool remove_backup_dots = true;
  const bool match = true;
  std::pair<const char *, const char *> tests[] = {
      {R"(C:\foo\bar\baz)", R"(C:\foo\foo\..\bar\baz)"},
      {R"(C:\bar\baz)", R"(C:\foo\..\bar\baz)"},
      {R"(C:\bar\baz)", R"(C:/foo/../bar/baz)"},
      {R"(C:/bar/baz)", R"(C:\foo\..\bar\baz)"},
      {R"(C:\bar)", R"(C:\foo\..\bar)"},
  };

  for(const auto &test: tests) {
    FileSpec one(test.first, false, FileSpec::ePathSyntaxWindows);
    FileSpec two(test.second, false, FileSpec::ePathSyntaxWindows);
    EXPECT_NE(one, two);
    Compare(one, two, full_match, remove_backup_dots, match);
    Compare(one, two, full_match, !remove_backup_dots, !match);
    Compare(one, two, !full_match, remove_backup_dots, match);
    Compare(one, two, !full_match, !remove_backup_dots, match);
  }

}
#endif

TEST(FileSpecTest, EqualDotsPosix) {
  const bool full_match = true;
  const bool remove_backup_dots = true;
  const bool match = true;
  std::pair<const char *, const char *> tests[] = {
      {R"(/foo/bar/baz)", R"(/foo/foo/../bar/baz)"},
      {R"(/bar/baz)", R"(/foo/../bar/baz)"},
//      {R"(/bar)", R"(/foo/../bar)"},
  };

  for(const auto &test: tests) {
    FileSpec one(test.first, false, FileSpec::ePathSyntaxPosix);
    FileSpec two(test.second, false, FileSpec::ePathSyntaxPosix);
    EXPECT_NE(one, two);
    Compare(one, two, full_match, remove_backup_dots, match);
    Compare(one, two, full_match, !remove_backup_dots, !match);
    Compare(one, two, !full_match, remove_backup_dots, match);
//    Compare(one, two, !full_match, !remove_backup_dots, match);
  }

}

#if 0
TEST(FileSpecTest, EqualDotsPosixRoot) {
  const bool full_match = true;
  const bool remove_backup_dots = true;
  const bool match = true;
  std::pair<const char *, const char *> tests[] = {
      {R"(/)", R"(/..)"},
      {R"(/)", R"(/foo/..)"},
  };

  for(const auto &test: tests) {
    FileSpec one(test.first, false, FileSpec::ePathSyntaxPosix);
    FileSpec two(test.second, false, FileSpec::ePathSyntaxPosix);
    EXPECT_NE(one, two);
    Compare(one, two, full_match, remove_backup_dots, match);
    Compare(one, two, !full_match, remove_backup_dots, match);
  }
}
#endif
