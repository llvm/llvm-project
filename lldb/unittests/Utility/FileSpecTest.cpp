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
  FileSpec fs_posix("/foo/bar", false, FileSpec::Style::posix);
  EXPECT_STREQ("/foo/bar", fs_posix.GetCString());
  EXPECT_STREQ("/foo", fs_posix.GetDirectory().GetCString());
  EXPECT_STREQ("bar", fs_posix.GetFilename().GetCString());

  FileSpec fs_windows("F:\\bar", false, FileSpec::Style::windows);
  EXPECT_STREQ("F:\\bar", fs_windows.GetCString());
  // EXPECT_STREQ("F:\\", fs_windows.GetDirectory().GetCString()); // It returns
  // "F:/"
  EXPECT_STREQ("bar", fs_windows.GetFilename().GetCString());

  FileSpec fs_posix_root("/", false, FileSpec::Style::posix);
  EXPECT_STREQ("/", fs_posix_root.GetCString());
  EXPECT_EQ(nullptr, fs_posix_root.GetDirectory().GetCString());
  EXPECT_STREQ("/", fs_posix_root.GetFilename().GetCString());

  FileSpec fs_windows_drive("F:", false, FileSpec::Style::windows);
  EXPECT_STREQ("F:", fs_windows_drive.GetCString());
  EXPECT_EQ(nullptr, fs_windows_drive.GetDirectory().GetCString());
  EXPECT_STREQ("F:", fs_windows_drive.GetFilename().GetCString());

  FileSpec fs_windows_root("F:\\", false, FileSpec::Style::windows);
  EXPECT_STREQ("F:\\", fs_windows_root.GetCString());
  EXPECT_STREQ("F:", fs_windows_root.GetDirectory().GetCString());
  // EXPECT_STREQ("\\", fs_windows_root.GetFilename().GetCString()); // It
  // returns "/"

  FileSpec fs_posix_long("/foo/bar/baz", false, FileSpec::Style::posix);
  EXPECT_STREQ("/foo/bar/baz", fs_posix_long.GetCString());
  EXPECT_STREQ("/foo/bar", fs_posix_long.GetDirectory().GetCString());
  EXPECT_STREQ("baz", fs_posix_long.GetFilename().GetCString());

  FileSpec fs_windows_long("F:\\bar\\baz", false, FileSpec::Style::windows);
  EXPECT_STREQ("F:\\bar\\baz", fs_windows_long.GetCString());
  // EXPECT_STREQ("F:\\bar", fs_windows_long.GetDirectory().GetCString()); // It
  // returns "F:/bar"
  EXPECT_STREQ("baz", fs_windows_long.GetFilename().GetCString());

  FileSpec fs_posix_trailing_slash("/foo/bar/", false, FileSpec::Style::posix);
  EXPECT_STREQ("/foo/bar", fs_posix_trailing_slash.GetCString());
  EXPECT_STREQ("/foo", fs_posix_trailing_slash.GetDirectory().GetCString());
  EXPECT_STREQ("bar", fs_posix_trailing_slash.GetFilename().GetCString());

  FileSpec fs_windows_trailing_slash("F:\\bar\\", false,
                                     FileSpec::Style::windows);
  EXPECT_STREQ("F:\\bar", fs_windows_trailing_slash.GetCString());
  EXPECT_STREQ("bar", fs_windows_trailing_slash.GetFilename().GetCString());
}

TEST(FileSpecTest, AppendPathComponent) {
  FileSpec fs_posix("/foo", false, FileSpec::Style::posix);
  fs_posix.AppendPathComponent("bar");
  EXPECT_STREQ("/foo/bar", fs_posix.GetCString());
  EXPECT_STREQ("/foo", fs_posix.GetDirectory().GetCString());
  EXPECT_STREQ("bar", fs_posix.GetFilename().GetCString());

  FileSpec fs_posix_2("/foo", false, FileSpec::Style::posix);
  fs_posix_2.AppendPathComponent("//bar/baz");
  EXPECT_STREQ("/foo/bar/baz", fs_posix_2.GetCString());
  EXPECT_STREQ("/foo/bar", fs_posix_2.GetDirectory().GetCString());
  EXPECT_STREQ("baz", fs_posix_2.GetFilename().GetCString());

  FileSpec fs_windows("F:\\bar", false, FileSpec::Style::windows);
  fs_windows.AppendPathComponent("baz");
  EXPECT_STREQ("F:\\bar\\baz", fs_windows.GetCString());
  // EXPECT_STREQ("F:\\bar", fs_windows.GetDirectory().GetCString()); // It
  // returns "F:/bar"
  EXPECT_STREQ("baz", fs_windows.GetFilename().GetCString());

  FileSpec fs_posix_root("/", false, FileSpec::Style::posix);
  fs_posix_root.AppendPathComponent("bar");
  EXPECT_STREQ("/bar", fs_posix_root.GetCString());
  EXPECT_STREQ("/", fs_posix_root.GetDirectory().GetCString());
  EXPECT_STREQ("bar", fs_posix_root.GetFilename().GetCString());

  FileSpec fs_windows_root("F:\\", false, FileSpec::Style::windows);
  fs_windows_root.AppendPathComponent("bar");
  EXPECT_STREQ("F:\\bar", fs_windows_root.GetCString());
  // EXPECT_STREQ("F:\\", fs_windows_root.GetDirectory().GetCString()); // It
  // returns "F:/"
  EXPECT_STREQ("bar", fs_windows_root.GetFilename().GetCString());
}

TEST(FileSpecTest, CopyByAppendingPathComponent) {
  FileSpec fs = FileSpec("/foo", false, FileSpec::Style::posix)
                    .CopyByAppendingPathComponent("bar");
  EXPECT_STREQ("/foo/bar", fs.GetCString());
  EXPECT_STREQ("/foo", fs.GetDirectory().GetCString());
  EXPECT_STREQ("bar", fs.GetFilename().GetCString());
}

TEST(FileSpecTest, PrependPathComponent) {
  FileSpec fs_posix("foo", false, FileSpec::Style::posix);
  fs_posix.PrependPathComponent("/bar");
  EXPECT_STREQ("/bar/foo", fs_posix.GetCString());

  FileSpec fs_posix_2("foo/bar", false, FileSpec::Style::posix);
  fs_posix_2.PrependPathComponent("/baz");
  EXPECT_STREQ("/baz/foo/bar", fs_posix_2.GetCString());

  FileSpec fs_windows("baz", false, FileSpec::Style::windows);
  fs_windows.PrependPathComponent("F:\\bar");
  EXPECT_STREQ("F:\\bar\\baz", fs_windows.GetCString());

  FileSpec fs_posix_root("bar", false, FileSpec::Style::posix);
  fs_posix_root.PrependPathComponent("/");
  EXPECT_STREQ("/bar", fs_posix_root.GetCString());

  FileSpec fs_windows_root("bar", false, FileSpec::Style::windows);
  fs_windows_root.PrependPathComponent("F:\\");
  EXPECT_STREQ("F:\\bar", fs_windows_root.GetCString());
}

TEST(FileSpecTest, EqualSeparator) {
  FileSpec backward("C:\\foo\\bar", false, FileSpec::Style::windows);
  FileSpec forward("C:/foo/bar", false, FileSpec::Style::windows);
  EXPECT_EQ(forward, backward);
}

TEST(FileSpecTest, EqualDotsWindows) {
  std::pair<const char *, const char *> tests[] = {
      {R"(C:\foo\bar\baz)", R"(C:\foo\foo\..\bar\baz)"},
      {R"(C:\bar\baz)", R"(C:\foo\..\bar\baz)"},
      {R"(C:\bar\baz)", R"(C:/foo/../bar/baz)"},
      {R"(C:/bar/baz)", R"(C:\foo\..\bar\baz)"},
      {R"(C:\bar)", R"(C:\foo\..\bar)"},
      {R"(C:\foo\bar)", R"(C:\foo\.\bar)"},
      {R"(C:\foo\bar)", R"(C:\foo\bar\.)"},
  };

  for (const auto &test : tests) {
    FileSpec one(test.first, false, FileSpec::Style::windows);
    FileSpec two(test.second, false, FileSpec::Style::windows);
    EXPECT_EQ(one, two);
  }
}

TEST(FileSpecTest, EqualDotsPosix) {
  std::pair<const char *, const char *> tests[] = {
      {R"(/foo/bar/baz)", R"(/foo/foo/../bar/baz)"},
      {R"(/bar/baz)", R"(/foo/../bar/baz)"},
      {R"(/bar)", R"(/foo/../bar)"},
      {R"(/foo/bar)", R"(/foo/./bar)"},
      {R"(/foo/bar)", R"(/foo/bar/.)"},
  };

  for (const auto &test : tests) {
    FileSpec one(test.first, false, FileSpec::Style::posix);
    FileSpec two(test.second, false, FileSpec::Style::posix);
    EXPECT_EQ(one, two);
  }
}

TEST(FileSpecTest, EqualDotsPosixRoot) {
  std::pair<const char *, const char *> tests[] = {
      {R"(/)", R"(/..)"},
      {R"(/)", R"(/.)"},
      {R"(/)", R"(/foo/..)"},
  };

  for (const auto &test : tests) {
    FileSpec one(test.first, false, FileSpec::Style::posix);
    FileSpec two(test.second, false, FileSpec::Style::posix);
    EXPECT_EQ(one, two);
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
      {"//", "/"},
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
      {"././foo", "foo"},
      {"../foo", "../foo"},
      {"../../foo", "../../foo"},
  };
  for (auto test : posix_tests) {
    SCOPED_TRACE(llvm::Twine("test.first = ") + test.first);
    EXPECT_EQ(test.second,
              FileSpec(test.first, false, FileSpec::Style::posix).GetPath());
  }

  std::pair<const char *, const char *> windows_tests[] = {
      {R"(c:\bar\..\bar)", R"(c:\bar)"},
      {R"(c:\bar\.\bar)", R"(c:\bar\bar)"},
      {R"(c:\bar\..)", R"(c:\)"},
      {R"(c:\bar\.)", R"(c:\bar)"},
      {R"(c:\.\bar)", R"(c:\bar)"},
      {R"(\)", R"(\)"},
      {R"(\\)", R"(\)"},
      {R"(\\net)", R"(\\net)"},
      {R"(c:\..)", R"(c:\)"},
      {R"(c:\.)", R"(c:\)"},
      // TODO: fix llvm::sys::path::remove_dots() to return "\" below.
      {R"(\..)", R"(\..)"},
      //      {R"(c:..)", R"(c:..)"},
      {R"(..)", R"(..)"},
      {R"(.)", R"(.)"},
      // TODO: fix llvm::sys::path::remove_dots() to return "c:\" below.
      {R"(c:..\..)", R"(c:\..\..)"},
      {R"(..\..)", R"(..\..)"},
      {R"(foo\..)", R"(.)"},
      {R"(foo\..\bar)", R"(bar)"},
      {R"(..\foo\..)", R"(..)"},
      {R"(.\foo)", R"(foo)"},
      {R"(.\.\foo)", R"(foo)"},
      {R"(..\foo)", R"(..\foo)"},
      {R"(..\..\foo)", R"(..\..\foo)"},
  };
  for (auto test : windows_tests) {
    EXPECT_EQ(test.second,
              FileSpec(test.first, false, FileSpec::Style::windows).GetPath())
        << "Original path: " << test.first;
  }
}

TEST(FileSpecTest, FormatFileSpec) {
  auto win = FileSpec::Style::windows;

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

TEST(FileSpecTest, IsRelative) {
  llvm::StringRef not_relative[] = {
    "/",
    "/a",
    "/a/",
    "/a/b",
    "/a/b/",
    "//",
    "//a",
    "//a/",
    "//a/b",
    "//a/b/",
    "~",
    "~/",
    "~/a",
    "~/a/",
    "~/a/b"
    "~/a/b/",
    "/foo/.",
    "/foo/..",
    "/foo/../",
    "/foo/../.",
  };
  for (const auto &path: not_relative) {
    FileSpec spec(path, false, FileSpec::Style::posix);
    EXPECT_FALSE(spec.IsRelative());
  }
  llvm::StringRef is_relative[] = {
    ".",
    "./",
    ".///",
    "a",
    "./a",
    "./a/",
    "./a/",
    "./a/b",
    "./a/b/",
    "../foo",
    "foo/bar.c",
    "./foo/bar.c"
  };
  for (const auto &path: is_relative) {
    FileSpec spec(path, false, FileSpec::Style::posix);
    EXPECT_TRUE(spec.IsRelative());
  }
}

