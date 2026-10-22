//===-- PathMappingListTest.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/PathMappingList.h"
#include "lldb/Utility/FileSpec.h"
#include "llvm/ADT/ArrayRef.h"
#include "gtest/gtest.h"
#include <utility>

using namespace lldb_private;

namespace {
struct Matches {
  FileSpec original;
  FileSpec remapped;
  Matches(const char *o, const char *r) : original(o), remapped(r) {}
  Matches(const char *o, llvm::sys::path::Style style, const char *r)
      : original(o, style), remapped(r) {}
};
} // namespace

static void TestPathMappings(const PathMappingList &map,
                             llvm::ArrayRef<Matches> matches,
                             llvm::ArrayRef<std::string> fails) {
  std::string actual_remapped;
  for (const auto &fail : fails) {
    SCOPED_TRACE(fail.c_str());
    EXPECT_FALSE(map.RemapPath(fail, actual_remapped))
        << "actual_remapped: " << actual_remapped.c_str();
  }
  for (const auto &match : matches) {
    SCOPED_TRACE(match.original.GetPath() + " -> " + match.remapped.GetPath());
    std::string orig_normalized = match.original.GetPath();
    EXPECT_TRUE(map.RemapPath(match.original.GetPath(), actual_remapped));
    EXPECT_EQ(FileSpec(actual_remapped), match.remapped);
    FileSpec unmapped_spec;
    EXPECT_TRUE(
        map.ReverseRemapPath(match.remapped, unmapped_spec).has_value());
    std::string unmapped_path = unmapped_spec.GetPath();
    EXPECT_EQ(unmapped_path, orig_normalized);
  }
}

TEST(PathMappingListTest, RelativeTests) {
  Matches matches[] = {
    {".", "/tmp"},
    {"./", "/tmp"},
    {"./////", "/tmp"},
    {"./foo.c", "/tmp/foo.c"},
    {"foo.c", "/tmp/foo.c"},
    {"./bar/foo.c", "/tmp/bar/foo.c"},
    {"bar/foo.c", "/tmp/bar/foo.c"},
  };
  std::string fails[] = {
#ifdef _WIN32
      "C:\\",
      "C:\\a",
#else
      "/a",
      "/",
#endif
  };
  PathMappingList map;
  map.Append(".", "/tmp", false);
  TestPathMappings(map, matches, fails);
  PathMappingList map2;
  map2.Append("", "/tmp", false);
  TestPathMappings(map, matches, fails);
}

TEST(PathMappingListTest, AbsoluteTests) {
  PathMappingList map;
  map.Append("/old", "/new", false);
  Matches matches[] = {
    {"/old", "/new"},
    {"/old/", "/new"},
    {"/old/foo/.", "/new/foo"},
    {"/old/foo.c", "/new/foo.c"},
    {"/old/foo.c/.", "/new/foo.c"},
    {"/old/./foo.c", "/new/foo.c"},
  };
  std::string fails[] = {
      "/foo", "/", "foo.c", "./foo.c", "../foo.c", "../bar/foo.c",
  };
  TestPathMappings(map, matches, fails);
}

TEST(PathMappingListTest, RemapRoot) {
  PathMappingList map;
  map.Append("/", "/new", false);
  Matches matches[] = {
    {"/old", "/new/old"},
    {"/old/", "/new/old"},
    {"/old/foo/.", "/new/old/foo"},
    {"/old/foo.c", "/new/old/foo.c"},
    {"/old/foo.c/.", "/new/old/foo.c"},
    {"/old/./foo.c", "/new/old/foo.c"},
  };
  std::string fails[] = {
      "foo.c",
      "./foo.c",
      "../foo.c",
      "../bar/foo.c",
  };
  TestPathMappings(map, matches, fails);
}

#ifndef _WIN32
TEST(PathMappingListTest, CrossPlatformTests) {
  PathMappingList map;
  map.Append(R"(C:\old)", "/new", false);
  Matches matches[] = {
    {R"(C:\old)", llvm::sys::path::Style::windows, "/new"},
    {R"(C:\old\)", llvm::sys::path::Style::windows, "/new"},
    {R"(C:\old\foo\.)", llvm::sys::path::Style::windows, "/new/foo"},
    {R"(C:\old\foo.c)", llvm::sys::path::Style::windows, "/new/foo.c"},
    {R"(C:\old\foo.c\.)", llvm::sys::path::Style::windows, "/new/foo.c"},
    {R"(C:\old\.\foo.c)", llvm::sys::path::Style::windows, "/new/foo.c"},
  };
  std::string fails[] = {
      "/foo", "/", "foo.c", "./foo.c", "../foo.c", "../bar/foo.c",
  };
  TestPathMappings(map, matches, fails);
}
#endif
