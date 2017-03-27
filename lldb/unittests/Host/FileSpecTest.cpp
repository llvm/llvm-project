//===-- FileSpecTest.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Utility/FileSpec.h"

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

TEST(FileSpecTest, PrependPathComponent) {
  FileSpec fs_posix("foo", false, FileSpec::ePathSyntaxPosix);
  fs_posix.PrependPathComponent("/bar");
  EXPECT_STREQ("/bar/foo", fs_posix.GetCString());

  FileSpec fs_posix_2("foo/bar", false, FileSpec::ePathSyntaxPosix);
  fs_posix_2.PrependPathComponent("/baz");
  EXPECT_STREQ("/baz/foo/bar", fs_posix_2.GetCString());

  FileSpec fs_windows("baz", false, FileSpec::ePathSyntaxWindows);
  fs_windows.PrependPathComponent("F:\\bar");
  EXPECT_STREQ("F:\\bar\\baz", fs_windows.GetCString());

  FileSpec fs_posix_root("bar", false, FileSpec::ePathSyntaxPosix);
  fs_posix_root.PrependPathComponent("/");
  EXPECT_STREQ("/bar", fs_posix_root.GetCString());

  FileSpec fs_windows_root("bar", false, FileSpec::ePathSyntaxWindows);
  fs_windows_root.PrependPathComponent("F:\\");
  EXPECT_STREQ("F:\\bar", fs_windows_root.GetCString());
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
      {R"(C:\foo\bar)", R"(C:\foo\.\bar)"},
      {R"(C:\foo\bar)", R"(C:\foo\bar\.)"},
  };

  for(const auto &test: tests) {
    FileSpec one(test.first, false, FileSpec::ePathSyntaxWindows);
    FileSpec two(test.second, false, FileSpec::ePathSyntaxWindows);
    EXPECT_NE(one, two);
    Compare(one, two, full_match, remove_backup_dots, match);
    Compare(one, two, full_match, !remove_backup_dots, !match);
    Compare(one, two, !full_match, remove_backup_dots, match);
    Compare(one, two, !full_match, !remove_backup_dots, !match);
  }

}

TEST(FileSpecTest, EqualDotsPosix) {
  const bool full_match = true;
  const bool remove_backup_dots = true;
  const bool match = true;
  std::pair<const char *, const char *> tests[] = {
      {R"(/foo/bar/baz)", R"(/foo/foo/../bar/baz)"},
      {R"(/bar/baz)", R"(/foo/../bar/baz)"},
      {R"(/bar)", R"(/foo/../bar)"},
      {R"(/foo/bar)", R"(/foo/./bar)"},
      {R"(/foo/bar)", R"(/foo/bar/.)"},
  };

  for(const auto &test: tests) {
    FileSpec one(test.first, false, FileSpec::ePathSyntaxPosix);
    FileSpec two(test.second, false, FileSpec::ePathSyntaxPosix);
    EXPECT_NE(one, two);
    Compare(one, two, full_match, remove_backup_dots, match);
    Compare(one, two, full_match, !remove_backup_dots, !match);
    Compare(one, two, !full_match, remove_backup_dots, match);
    Compare(one, two, !full_match, !remove_backup_dots, !match);
  }

}

TEST(FileSpecTest, EqualDotsPosixRoot) {
  const bool full_match = true;
  const bool remove_backup_dots = true;
  const bool match = true;
  std::pair<const char *, const char *> tests[] = {
      {R"(/)", R"(/..)"}, {R"(/)", R"(/.)"}, {R"(/)", R"(/foo/..)"},
  };

  for(const auto &test: tests) {
    FileSpec one(test.first, false, FileSpec::ePathSyntaxPosix);
    FileSpec two(test.second, false, FileSpec::ePathSyntaxPosix);
    EXPECT_NE(one, two);
    Compare(one, two, full_match, remove_backup_dots, match);
    Compare(one, two, full_match, !remove_backup_dots, !match);
    Compare(one, two, !full_match, remove_backup_dots, !match);
    Compare(one, two, !full_match, !remove_backup_dots, !match);
  }
}

TEST(FileSpecTest, GetNormalizedPath) {
  std::pair<const char *, const char *> posix_tests[] = {
      {"/foo/.././bar", "/bar"},
      {"/foo/./../bar", "/bar"},
      {"/foo/../bar", "/bar"},
      {"/foo/./bar", "/foo/bar"},
      {"/foo/..", "/"},
      {"/foo/.", "/foo"},
      {"/foo//bar", "/foo/bar"},
      {"/foo//bar/baz", "/foo/bar/baz"},
      {"/foo//bar/./baz", "/foo/bar/baz"},
      {"/./foo", "/foo"},
      {"/", "/"},
      {"//", "//"},
      {"//net", "//net"},
      {"/..", "/"},
      {"/.", "/"},
      {"..", ".."},
      {".", "."},
      {"../..", "../.."},
      {"foo/..", "."},
      {"foo/../bar", "bar"},
      {"../foo/..", ".."},
      {"./foo", "foo"},
  };
  for (auto test : posix_tests) {
    EXPECT_EQ(test.second,
              FileSpec(test.first, false, FileSpec::ePathSyntaxPosix)
                  .GetNormalizedPath()
                  .GetPath());
  }

  std::pair<const char *, const char *> windows_tests[] = {
      {R"(c:\bar\..\bar)", R"(c:\bar)"},
      {R"(c:\bar\.\bar)", R"(c:\bar\bar)"},
      {R"(c:\bar\..)", R"(c:\)"},
      {R"(c:\bar\.)", R"(c:\bar)"},
      {R"(c:\.\bar)", R"(c:\bar)"},
      {R"(\)", R"(\)"},
      //      {R"(\\)", R"(\\)"},
      //      {R"(\\net)", R"(\\net)"},
      {R"(c:\..)", R"(c:\)"},
      {R"(c:\.)", R"(c:\)"},
      {R"(\..)", R"(\)"},
      //      {R"(c:..)", R"(c:..)"},
      {R"(..)", R"(..)"},
      {R"(.)", R"(.)"},
      {R"(c:..\..)", R"(c:..\..)"},
      {R"(..\..)", R"(..\..)"},
      {R"(foo\..)", R"(.)"},
      {R"(foo\..\bar)", R"(bar)"},
      {R"(..\foo\..)", R"(..)"},
      {R"(.\foo)", R"(foo)"},
  };
  for (auto test : windows_tests) {
    EXPECT_EQ(test.second,
              FileSpec(test.first, false, FileSpec::ePathSyntaxWindows)
                  .GetNormalizedPath()
                  .GetPath())
        << "Original path: " << test.first;
  }
}

TEST(FileSpecTest, FormatFileSpec) {
  auto win = FileSpec::ePathSyntaxWindows;

  FileSpec F;
  EXPECT_EQ("(empty)", llvm::formatv("{0}", F).str());
  EXPECT_EQ("(empty)", llvm::formatv("{0:D}", F).str());
  EXPECT_EQ("(empty)", llvm::formatv("{0:F}", F).str());

  F = FileSpec("C:\\foo\\bar.txt", false, win);
  EXPECT_EQ("C:\\foo\\bar.txt", llvm::formatv("{0}", F).str());
  EXPECT_EQ("C:\\foo\\", llvm::formatv("{0:D}", F).str());
  EXPECT_EQ("bar.txt", llvm::formatv("{0:F}", F).str());

  F = FileSpec("foo\\bar.txt", false, win);
  EXPECT_EQ("foo\\bar.txt", llvm::formatv("{0}", F).str());
  EXPECT_EQ("foo\\", llvm::formatv("{0:D}", F).str());
  EXPECT_EQ("bar.txt", llvm::formatv("{0:F}", F).str());

  F = FileSpec("foo", false, win);
  EXPECT_EQ("foo", llvm::formatv("{0}", F).str());
  EXPECT_EQ("foo", llvm::formatv("{0:F}", F).str());
  EXPECT_EQ("(empty)", llvm::formatv("{0:D}", F).str());
}
