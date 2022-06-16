//===- unittests/Support/PrefixMapperTest.cpp - Prefix Mapper tests -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/PrefixMapper.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace llvm {
static inline void PrintTo(const MappedPrefix &P, std::ostream *OS) {
  *OS << ::testing::PrintToString(P.Old.str() + "=" + P.New.str());
}
} // end namespace llvm

namespace {

TEST(MappedPrefixTest, getInverse) {
  EXPECT_EQ((MappedPrefix{"b", "a"}), (MappedPrefix{"a", "b"}).getInverse());
  EXPECT_EQ((MappedPrefix{"a", "b"}), (MappedPrefix{"b", "a"}).getInverse());
}

TEST(MappedPrefixTest, getFromJoined) {
  auto split = MappedPrefix::getFromJoined;
  EXPECT_EQ(None, split(""));
  EXPECT_EQ(None, split("a"));
  EXPECT_EQ(None, split("abc"));
  EXPECT_EQ((MappedPrefix{"", ""}), split("="));
  EXPECT_EQ((MappedPrefix{"a", ""}), split("a="));
  EXPECT_EQ((MappedPrefix{"", "b"}), split("=b"));
  EXPECT_EQ((MappedPrefix{"a", "b"}), split("a=b"));
  EXPECT_EQ((MappedPrefix{"abc", "def"}), split("abc=def"));
  EXPECT_EQ((MappedPrefix{"", "="}), split("=="));
  EXPECT_EQ((MappedPrefix{"a", "b=c"}), split("a=b=c"));
}

TEST(MappedPrefixTest, transformJoined) {
  SmallVector<std::string> JoinedStrings = {
      "=", "a=", "=b", "a=b", "abc=def", "==", "a=b=c",
  };
  SmallVector<StringRef> JoinedRefs(JoinedStrings.begin(), JoinedStrings.end());

  MappedPrefix ExpectedSplit[] = {
      MappedPrefix{"", ""},       MappedPrefix{"a", ""},
      MappedPrefix{"", "b"},      MappedPrefix{"a", "b"},
      MappedPrefix{"abc", "def"}, MappedPrefix{"", "="},
      MappedPrefix{"a", "b=c"},
  };

  SmallVector<MappedPrefix> ComputedSplit;
  MappedPrefix::transformJoinedIfValid(JoinedStrings, ComputedSplit);
  EXPECT_EQ(makeArrayRef(ExpectedSplit), makeArrayRef(ComputedSplit));

  ComputedSplit.clear();
  MappedPrefix::transformJoinedIfValid(JoinedRefs, ComputedSplit);
  EXPECT_EQ(makeArrayRef(ExpectedSplit), makeArrayRef(ComputedSplit));
}

TEST(MappedPrefixTest, transformJoinedError) {
  SmallVector<std::string> JoinedStrings = {
      "old=new",
      "middle",
      "old-again=new-again",
      "after",
  };
  SmallVector<StringRef> JoinedRefs(JoinedStrings.begin(), JoinedStrings.end());

  SmallVector<MappedPrefix> Split;
  EXPECT_THAT_ERROR(MappedPrefix::transformJoined(JoinedStrings, Split),
                    FailedWithMessage("invalid prefix map: 'middle'"));
  ASSERT_EQ(0u, Split.size());
  EXPECT_THAT_ERROR(MappedPrefix::transformJoined(JoinedRefs, Split),
                    FailedWithMessage("invalid prefix map: 'middle'"));
  ASSERT_EQ(0u, Split.size());
}

TEST(MappedPrefixTest, transformJoinedIfValid) {
  SmallVector<std::string> JoinedStrings = {
      "", "a", "abc", "=", "a=", "=b", "a=b", "abc=def", "==", "a=b=c",
  };
  SmallVector<StringRef> JoinedRefs(JoinedStrings.begin(), JoinedStrings.end());

  MappedPrefix ExpectedSplit[] = {
      MappedPrefix{"", ""},       MappedPrefix{"a", ""},
      MappedPrefix{"", "b"},      MappedPrefix{"a", "b"},
      MappedPrefix{"abc", "def"}, MappedPrefix{"", "="},
      MappedPrefix{"a", "b=c"},
  };

  SmallVector<MappedPrefix> ComputedSplit;
  MappedPrefix::transformJoinedIfValid(JoinedStrings, ComputedSplit);
  EXPECT_EQ(makeArrayRef(ExpectedSplit), makeArrayRef(ComputedSplit));

  ComputedSplit.clear();
  MappedPrefix::transformJoinedIfValid(JoinedRefs, ComputedSplit);
  EXPECT_EQ(makeArrayRef(ExpectedSplit), makeArrayRef(ComputedSplit));
}

TEST(PrefixMapperTest, construct) {
  for (auto PathStyle : {
           sys::path::Style::posix,
           sys::path::Style::windows,
           sys::path::Style::native,
       })
    EXPECT_EQ(PathStyle, PrefixMapper(PathStyle).getPathStyle());
}

TEST(PrefixMapperTest, add) {
  PrefixMapper PM(sys::path::Style::posix);
  PM.add(MappedPrefix{"a", "b"});
  PM.add(MappedPrefix{"b", "a"});
  ASSERT_EQ(2u, PM.getMappings().size());
  ASSERT_EQ((MappedPrefix{"a", "b"}), PM.getMappings().front());
  ASSERT_EQ((MappedPrefix{"b", "a"}), PM.getMappings().back());
}

TEST(PrefixMapperTest, addRange) {
  PrefixMapper PM(sys::path::Style::posix);

  PM.add(MappedPrefix{"/old/before", "/new/before"});
  MappedPrefix Range[] = {
      {"/old/1", "/new/1"},
      {"/old/2", "/new/2"},
      {"/old/3", "/new/3"},
  };
  auto RangeRef = makeArrayRef(Range);
  PM.addRange(RangeRef);
  PM.add(MappedPrefix{"/old/after", "/new/after"});

  ASSERT_EQ(2u + RangeRef.size(), PM.getMappings().size());
  EXPECT_EQ((MappedPrefix{"/old/before", "/new/before"}),
            PM.getMappings().front());
  EXPECT_EQ((MappedPrefix{"/old/after", "/new/after"}),
            PM.getMappings().back());
  EXPECT_EQ(RangeRef, PM.getMappings().drop_front().drop_back());
}

static void checkPosix(sys::path::Style PathStyle) {
  PrefixMapper PM(PathStyle);
  MappedPrefix Mappings[] = {
      // Simple mappings.
      {"/old", "/new"},
      {"/shorter", "/this/is/longer"},
      {"/from/longer", "/shorter"},
      {"relative/old", "relative/new"},

      // Confirm that non-directory matches are skipped correctly.
      {"/old-almost", "/new-other"},

      // Confirm mappings don't run twice by adding a mapping that would rerun
      // on "/new".
      {"/new/reenter-check", "/reentered"},
  };
  PM.addRange(makeArrayRef(Mappings));

  // Not caught by anything.
  EXPECT_EQ("", PM.map(""));
  EXPECT_EQ("/x", PM.map("/x"));
  EXPECT_EQ("/new", PM.map("/new"));
  EXPECT_EQ("/new/nested", PM.map("/new/nested"));
  EXPECT_EQ("/old-in-prefix", PM.map("/old-in-prefix"));

  // Caught by simple mappings.
  EXPECT_EQ("/new", PM.map("/old"));
  EXPECT_EQ("/new/", PM.map("/old/"));
  EXPECT_EQ("/new/a", PM.map("/old/a"));
  EXPECT_EQ("/new/../x/y", PM.map("/old/../x/y"));
  EXPECT_EQ("relative/new", PM.map("relative/old"));
  EXPECT_EQ("/this/is/longer", PM.map("/shorter"));
  EXPECT_EQ("/this/is/longer/a", PM.map("/shorter/a"));
  EXPECT_EQ("/this/is/longer/with/long/nested/file",
            PM.map("/shorter/with/long/nested/file"));
  EXPECT_EQ("/shorter", PM.map("/from/longer"));
  EXPECT_EQ("/shorter/a", PM.map("/from/longer/a"));
  EXPECT_EQ("/shorter/with/long/nested/file",
            PM.map("/from/longer/with/long/nested/file"));

  // Check that "/old" doesn't catch "/old-almost".
  EXPECT_EQ("/new-other", PM.map("/old-almost"));
  EXPECT_EQ("/new-other/a", PM.map("/old-almost/a"));
}

TEST(PrefixMapperTest, mapPosix) { checkPosix(sys::path::Style::posix); }

static void checkWindows(sys::path::Style PathStyle) {
  PrefixMapper PM(PathStyle);
  MappedPrefix Mappings[] = {
      // Simple mappings.
      {"c:\\old", "c:\\new"},
      {"C:\\UPPER\\old", "c:\\UPPER\\new"},
      {"c:/forward/old", "C:/forward/new"},
      {"c:\\shorter", "c:\\this\\is\\longer"},
      {"c:\\from\\longer", "c:\\shorter"},
      {"relative\\old", "relative\\new"},

      // Confirm that non-directory matches are remapped correctly.
      {"c:\\old-almost", "c:\\new-other"},

      // Confirm mappings don't run twice, just the first that matches.
      {"c:\\new\\reenter-check", "c:\\reentered"},
  };
  PM.addRange(makeArrayRef(Mappings));

  // Not caught by anything.
  EXPECT_EQ("", PM.map(""));
  EXPECT_EQ("c:\\x", PM.map("c:\\x"));
  EXPECT_EQ("c:\\new", PM.map("c:\\new"));
  EXPECT_EQ("c:\\new\\nested", PM.map("c:\\new\\nested"));
  EXPECT_EQ("c:\\old-in-prefix", PM.map("c:\\old-in-prefix"));

  // Caught by simple mappings.
  EXPECT_EQ("c:\\new", PM.map("c:\\old"));
  EXPECT_EQ("c:\\new", PM.map("C:\\old"));
  EXPECT_EQ("c:\\new\\", PM.map("c:\\old\\"));
  EXPECT_EQ("c:\\new\\a", PM.map("c:\\old\\a"));
  EXPECT_EQ("c:\\new\\a", PM.map("C:\\old\\a"));
  EXPECT_EQ("c:\\new\\..\\x\\y", PM.map("c:\\old\\..\\x\\y"));
  EXPECT_EQ("c:\\UPPER\\new", PM.map("C:\\UPPER\\old"));
  EXPECT_EQ("C:/forward/new", PM.map("C:/forward/old"));
  EXPECT_EQ("C:/forward/new\\x/y", PM.map("C:/forward/old/x/y"));
  EXPECT_EQ("relative\\new", PM.map("relative\\old"));
  EXPECT_EQ("c:\\this\\is\\longer", PM.map("c:\\shorter"));
  EXPECT_EQ("c:\\this\\is\\longer\\a", PM.map("c:\\shorter\\a"));
  EXPECT_EQ("c:\\this\\is\\longer\\with\\long\\nested\\file",
            PM.map("c:\\shorter\\with\\long\\nested\\file"));
  EXPECT_EQ("c:\\shorter", PM.map("c:\\from\\longer"));
  EXPECT_EQ("c:\\shorter\\a", PM.map("c:\\from\\longer\\a"));
  EXPECT_EQ("c:\\shorter\\with\\long\\nested\\file",
            PM.map("c:\\from\\longer\\with\\long\\nested\\file"));

  // Check that "\\old" doesn't catch "\\old-almost".
  EXPECT_EQ("c:\\new-other", PM.map("c:\\old-almost"));
  EXPECT_EQ("c:\\new-other\\a", PM.map("c:\\old-almost\\a"));
}

TEST(PrefixMapperTest, mapWindows) { checkWindows(sys::path::Style::windows); }

TEST(PrefixMapperTest, mapNative) {
  if (sys::path::system_style() == sys::path::Style::posix)
    checkPosix(sys::path::Style::native);
  else
    checkWindows(sys::path::Style::native);
}

TEST(PrefixMapperTest, mapLifetime) {
  MappedPrefix Mappings[] = {
      {"/old", "/new"},
  };

  StringRef Input[] = {
      "/old/short",
      "/old/0123456789012345678901234567890123456789-long",
  };

  StringRef Expected[] = {
      "/new/short",
      "/new/0123456789012345678901234567890123456789-long",
  };

  SmallVector<StringRef> Default;
  SmallVector<StringRef> WithAlloc;

  PrefixMapper PMDefault(sys::path::Style::posix);
  BumpPtrAllocator Alloc;
  {
    // Shorter lifetime to confirm StringRefs live with Alloc.
    PrefixMapper PMWithAlloc(Alloc, sys::path::Style::posix);

    PMDefault.addRange(makeArrayRef(Mappings));
    PMWithAlloc.addRange(makeArrayRef(Mappings));
    for (StringRef Path : Input) {
      Default.push_back(PMDefault.map(Path));
      WithAlloc.push_back(PMWithAlloc.map(Path));
    }
    ASSERT_EQ(makeArrayRef(Expected), makeArrayRef(Default));
    ASSERT_EQ(makeArrayRef(Expected), makeArrayRef(WithAlloc));
  }

  // Check lifetime of returned StringRef. Sanitizers should crash with
  // use-after-free if there's a problem.
  ASSERT_EQ(makeArrayRef(Expected), makeArrayRef(Default));
  ASSERT_EQ(makeArrayRef(Expected), makeArrayRef(WithAlloc));
}

TEST(PrefixMapperTest, mapTwoArgs) {
  PrefixMapper PM(sys::path::Style::posix);
  MappedPrefix Mappings[] = {{"/old", "/new"}};
  PM.addRange(makeArrayRef(Mappings));

  MappedPrefix Tests[] = {
      {"/old", "/new"},     {"/old/x/y", "/new/x/y"},
      {"/old", "/new"},     {"relative/path", "relative/path"},
      {"/other", "/other"},
  };

  SmallString<128> OutputV;
  std::string OutputS;
  for (MappedPrefix M : Tests) {
    PM.map(M.Old, OutputV);
    PM.map(M.Old, OutputS);
    EXPECT_EQ(M.New, OutputV);
    EXPECT_EQ(M.New, OutputS);
  }
}

TEST(PrefixMapperTest, mapToString) {
  PrefixMapper PM(sys::path::Style::posix);
  MappedPrefix Mappings[] = {{"/old", "/new"}};
  PM.addRange(makeArrayRef(Mappings));

  MappedPrefix Tests[] = {
      {"/old", "/new"},     {"/old/x/y", "/new/x/y"},
      {"/old", "/new"},     {"relative/path", "relative/path"},
      {"/other", "/other"},
  };

  SmallString<128> Output;
  for (MappedPrefix M : Tests) {
    std::string S = PM.mapToString(M.Old);
    EXPECT_EQ(M.New, S);
  }
}

TEST(PrefixMapperTest, mapInPlace) {
  PrefixMapper PM(sys::path::Style::posix);
  MappedPrefix Mappings[] = {{"/old", "/new"}};
  PM.addRange(makeArrayRef(Mappings));

  MappedPrefix Tests[] = {
      {"/old", "/new"},     {"/old/x/y", "/new/x/y"},
      {"/old", "/new"},     {"relative/path", "relative/path"},
      {"/other", "/other"},
  };

  SmallString<128> V;
  std::string S;
  for (MappedPrefix M : Tests) {
    V = M.Old;
    S = M.Old.str();
    PM.mapInPlace(V);
    PM.mapInPlace(S);
    EXPECT_EQ(M.New, V);
    EXPECT_EQ(M.New, S);
  }
}

class GetDirectoryEntryFileSystem : public vfs::FileSystem {
public:
  ErrorOr<vfs::Status> status(const Twine &) override {
    return make_error_code(std::errc::operation_not_permitted);
  }
  vfs::directory_iterator dir_begin(const Twine &,
                                    std::error_code &EC) override {
    EC = make_error_code(std::errc::operation_not_permitted);
    return vfs::directory_iterator();
  }
  std::error_code setCurrentWorkingDirectory(const Twine &) override {
    return make_error_code(std::errc::operation_not_permitted);
  }
  ErrorOr<std::string> getCurrentWorkingDirectory() const override {
    return make_error_code(std::errc::operation_not_permitted);
  }
  ErrorOr<std::unique_ptr<vfs::File>> openFileForRead(const Twine &) override {
    return make_error_code(std::errc::operation_not_permitted);
  }

  Expected<const vfs::CachedDirectoryEntry *>
  getDirectoryEntry(const Twine &Path, bool) const override {
    auto I = Entries.find(Path.str());
    if (I == Entries.end())
      return createFileError(
          Path, make_error_code(std::errc::no_such_file_or_directory));
    return &I->second;
  }

  using CachedDirectoryEntry = vfs::CachedDirectoryEntry;
  StringMap<CachedDirectoryEntry> Entries = {
      {"relative", CachedDirectoryEntry("/real/path/1")},
      {"relative/", CachedDirectoryEntry("/real/path/1")},
      {"symlink/to/relative", CachedDirectoryEntry("/real/path/1")},
      {"relative/nested", CachedDirectoryEntry("/real/path/1/nested")},
      {"symlink/to/relative/nested",
       CachedDirectoryEntry("/real/path/1/nested")},
      {"/real/path/1", CachedDirectoryEntry("/real/path/1")},
      {"/real/path/1/", CachedDirectoryEntry("/real/path/1")},
      {"/real/path/1/nested", CachedDirectoryEntry("/real/path/1/nested")},
      {"/absolute", CachedDirectoryEntry("/real/path/2")},
      {"/absolute/", CachedDirectoryEntry("/real/path/2")},
      {"/absolute/nested", CachedDirectoryEntry("/real/path/2/nested")},
      {"/real/path/2", CachedDirectoryEntry("/real/path/2")},
      {"/real/path/2/", CachedDirectoryEntry("/real/path/2")},
      {"/real/path/2/nested", CachedDirectoryEntry("/real/path/2/nested")},
      {"/unmapped/file", CachedDirectoryEntry("/unmapped/file")},
      {"/unmapped/../unmapped/file", CachedDirectoryEntry("/unmapped/file")},
      {"/unmapped/symlink", CachedDirectoryEntry("/unmapped/symlink")},
      {"/unmapped/symlink/", CachedDirectoryEntry("/real/path/1")},
      {"/unmapped/symlink/nested", CachedDirectoryEntry("/real/path/1/nested")},
  };
};

TEST(TreePathPrefixMapperTest, construct) {
  auto FS = makeIntrusiveRefCnt<GetDirectoryEntryFileSystem>();

  for (auto PathStyle : {
           sys::path::Style::posix,
           sys::path::Style::windows,
           sys::path::Style::native,
       }) {
    EXPECT_EQ(PathStyle, TreePathPrefixMapper(FS, PathStyle).getPathStyle());
  }
}

TEST(TreePathPrefixMapperTest, add) {
  auto FS = makeIntrusiveRefCnt<GetDirectoryEntryFileSystem>();
  TreePathPrefixMapper PM(FS);

  EXPECT_THAT_ERROR(PM.add(MappedPrefix{"relative", "/new1"}), Succeeded());
  EXPECT_THAT_ERROR(PM.add(MappedPrefix{"/absolute", "/new2"}), Succeeded());
  ASSERT_EQ(2u, PM.getMappings().size());
  EXPECT_EQ((MappedPrefix{"/real/path/1", "/new1"}), PM.getMappings().front());
  EXPECT_EQ((MappedPrefix{"/real/path/2", "/new2"}), PM.getMappings().back());

  EXPECT_THAT_ERROR(PM.add(MappedPrefix{"missing", "/new"}), Failed());
  EXPECT_EQ(2u, PM.getMappings().size());
}

TEST(TreePathPrefixMapperTest, addRange) {
  auto FS = makeIntrusiveRefCnt<GetDirectoryEntryFileSystem>();
  TreePathPrefixMapper PM(FS);

  MappedPrefix BadMapping[] = {
      {"missing", "/new"},
  };
  MappedPrefix Mappings[] = {
      {"relative", "/new1"},
      {"/absolute", "/new2"},
  };
  EXPECT_THAT_ERROR(PM.addRange(makeArrayRef(Mappings)), Succeeded());
  ASSERT_EQ(2u, PM.getMappings().size());
  EXPECT_EQ((MappedPrefix{"/real/path/1", "/new1"}), PM.getMappings().front());
  EXPECT_EQ((MappedPrefix{"/real/path/2", "/new2"}), PM.getMappings().back());

  EXPECT_THAT_ERROR(PM.addRange(makeArrayRef(BadMapping)), Failed());
  EXPECT_EQ(2u, PM.getMappings().size());
}

TEST(TreePathPrefixMapperTest, addRangeIfValid) {
  auto FS = makeIntrusiveRefCnt<GetDirectoryEntryFileSystem>();
  TreePathPrefixMapper PM(FS);

  MappedPrefix Mappings[] = {
      {"missing-before", "/new"}, {"relative", "/new1"},
      {"missing", "/new"},        {"/absolute", "/new2"},
      {"missing-after", "/new"},
  };
  PM.addRangeIfValid(makeArrayRef(Mappings));
  ASSERT_EQ(2u, PM.getMappings().size());
  EXPECT_EQ((MappedPrefix{"/real/path/1", "/new1"}), PM.getMappings().front());
  EXPECT_EQ((MappedPrefix{"/real/path/2", "/new2"}), PM.getMappings().back());
}

TEST(TreePathPrefixMapperTest, addInverseRange) {
  auto FS = makeIntrusiveRefCnt<GetDirectoryEntryFileSystem>();
  TreePathPrefixMapper PM(FS);

  MappedPrefix BadMapping[] = {
      {"/new", "missing"},
  };
  MappedPrefix Mappings[] = {
      {"/new1", "relative"},
      {"/new2", "/absolute"},
  };
  EXPECT_THAT_ERROR(PM.addInverseRange(makeArrayRef(Mappings)), Succeeded());
  ASSERT_EQ(2u, PM.getMappings().size());
  EXPECT_EQ((MappedPrefix{"/real/path/1", "/new1"}), PM.getMappings().front());
  EXPECT_EQ((MappedPrefix{"/real/path/2", "/new2"}), PM.getMappings().back());

  EXPECT_THAT_ERROR(PM.addInverseRange(makeArrayRef(BadMapping)), Failed());
  EXPECT_EQ(2u, PM.getMappings().size());
}

TEST(TreePathPrefixMapperTest, addInverseRangeIfValid) {
  auto FS = makeIntrusiveRefCnt<GetDirectoryEntryFileSystem>();
  TreePathPrefixMapper PM(FS);

  MappedPrefix Mappings[] = {
      {"/new", "missing-before"}, {"/new1", "relative"},
      {"/new", "missing"},        {"/new2", "/absolute"},
      {"/new", "missing-after"},
  };
  PM.addInverseRangeIfValid(makeArrayRef(Mappings));
  ASSERT_EQ(2u, PM.getMappings().size());
  EXPECT_EQ((MappedPrefix{"/real/path/1", "/new1"}), PM.getMappings().front());
  EXPECT_EQ((MappedPrefix{"/real/path/2", "/new2"}), PM.getMappings().back());
}

struct MapState {
  BumpPtrAllocator Alloc;
  IntrusiveRefCntPtr<GetDirectoryEntryFileSystem> FS =
      makeIntrusiveRefCnt<GetDirectoryEntryFileSystem>();
  TreePathPrefixMapper PM;

  SmallVector<MappedPrefix> Tests = {
      {"", ""},
      {"/unmapped/file", "/unmapped/file"},
      {"/unmapped/../unmapped/file", "/unmapped/file"},
      {"relative", "/new1"},
      {"relative/nested", "/new1/nested"},
      {"symlink/to/relative", "/new1"},
      {"symlink/to/relative/nested", "/new1/nested"},
      {"/real/path/1", "/new1"},
      {"/real/path/1/nested", "/new1/nested"},
      {"/absolute", "/new2"},
      {"/absolute/nested", "/new2/nested"},
      {"/real/path/2", "/new2"},
      {"/real/path/2/nested", "/new2/nested"},
  };
  SmallVector<StringRef> FailedTests = {"missing", "/missing", "/relative"};
  MapState() : PM(FS, Alloc) {
    EXPECT_THAT_ERROR(PM.add(MappedPrefix{"relative", "/new1"}), Succeeded());
    EXPECT_THAT_ERROR(PM.add(MappedPrefix{"/absolute", "/new2"}), Succeeded());
  }
};

TEST(TreePathPrefixMapperTest, map) {
  MapState State;
  ASSERT_EQ(2u, State.PM.getMappings().size());

  SmallString<128> NotFoundV;
  std::string NotFoundS;
  SmallString<128> FoundV;
  std::string FoundS;
  for (StringRef S : State.FailedTests) {
    FoundV = "";
    FoundS = "";
    EXPECT_THAT_EXPECTED(State.PM.map(S), Failed());
    EXPECT_THAT_EXPECTED(State.PM.mapToString(S), Failed());
    EXPECT_THAT_ERROR(State.PM.map(S, NotFoundV), Failed());
    EXPECT_THAT_ERROR(State.PM.map(S, NotFoundS), Failed());
    EXPECT_EQ("", NotFoundV);
    EXPECT_EQ("", NotFoundS);
    EXPECT_EQ(None, State.PM.mapOrNone(S));
    EXPECT_EQ(None, State.PM.mapToStringOrNone(S));
    EXPECT_EQ(S, State.PM.mapOrOriginal(S));

    State.PM.mapOrOriginal(S, FoundV);
    State.PM.mapOrOriginal(S, FoundS);
    EXPECT_EQ(S, FoundV);
    EXPECT_EQ(S, FoundS);
  }

  for (MappedPrefix Map : State.Tests) {
    EXPECT_THAT_EXPECTED(State.PM.map(Map.Old), HasValue(Map.New));
    EXPECT_THAT_EXPECTED(State.PM.mapToString(Map.Old), HasValue(Map.New));
    EXPECT_THAT_ERROR(State.PM.map(Map.Old, FoundV), Succeeded());
    EXPECT_THAT_ERROR(State.PM.map(Map.Old, FoundS), Succeeded());
    EXPECT_EQ(Map.New, FoundV);
    EXPECT_EQ(Map.New, FoundS);
    EXPECT_EQ(Map.New, State.PM.mapOrNone(Map.Old));
    EXPECT_EQ(Map.New.str(), State.PM.mapToStringOrNone(Map.Old));
    EXPECT_EQ(Map.New, State.PM.mapOrOriginal(Map.Old));

    FoundV = "";
    FoundS = "";
    State.PM.mapOrOriginal(Map.Old, FoundV);
    State.PM.mapOrOriginal(Map.Old, FoundS);
    EXPECT_EQ(Map.New, FoundV);
    EXPECT_EQ(Map.New, FoundS);

    if (Map.Old.empty())
      continue;
    const vfs::CachedDirectoryEntry *Entry = nullptr;
    ASSERT_THAT_ERROR(
        State.FS->getDirectoryEntry(Map.Old, /*FollowSymlinks=*/false)
            .moveInto(Entry),
        Succeeded());
    FoundV = "";
    FoundS = "";
    State.PM.map(*Entry, FoundV);
    State.PM.map(*Entry, FoundS);
    EXPECT_EQ(Map.New, State.PM.map(*Entry));
    EXPECT_EQ(Map.New, State.PM.mapToString(*Entry));
    EXPECT_EQ(Map.New, FoundV);
    EXPECT_EQ(Map.New, FoundS);
  }
}

TEST(TreePathPrefixMapperTest, mapInPlace) {
  MapState State;
  ASSERT_EQ(2u, State.PM.getMappings().size());

  std::string FoundS;
  SmallString<128> FoundV;
  for (StringRef S : State.FailedTests) {
    FoundS = S.str();
    FoundV = S;
    EXPECT_THAT_ERROR(State.PM.mapInPlace(FoundS), Failed());
    EXPECT_THAT_ERROR(State.PM.mapInPlace(FoundV), Failed());
    EXPECT_EQ(S, FoundS);
    EXPECT_EQ(S, FoundV);
    State.PM.mapInPlaceOrClear(FoundS);
    State.PM.mapInPlaceOrClear(FoundV);
    EXPECT_EQ("", FoundS);
    EXPECT_EQ("", FoundV);
  }

  for (MappedPrefix Map : State.Tests) {
    FoundS = Map.Old.str();
    FoundV = Map.Old;
    EXPECT_THAT_ERROR(State.PM.mapInPlace(FoundS), Succeeded());
    EXPECT_THAT_ERROR(State.PM.mapInPlace(FoundV), Succeeded());
    EXPECT_EQ(Map.New, FoundS);
    EXPECT_EQ(Map.New, FoundV);

    FoundS = Map.Old.str();
    FoundV = Map.Old;
    State.PM.mapInPlaceOrClear(FoundS);
    State.PM.mapInPlaceOrClear(FoundV);
    EXPECT_EQ(Map.New, FoundS);
    EXPECT_EQ(Map.New, FoundV);
  }
}

} // end namespace
