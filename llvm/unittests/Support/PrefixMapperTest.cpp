//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Prefix Mapper tests.
///
//===----------------------------------------------------------------------===//

#include "llvm/Support/PrefixMapper.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace llvm {
static inline void PrintTo(const MappedPrefix &P, std::ostream *OS) {
  *OS << ::testing::PrintToString(P.Old + "=" + P.New);
}
} // end namespace llvm

namespace {

struct PrefixMapperWrapper {
  PrefixMapper PM;
  BumpPtrAllocator Alloc;
  StringSaver Saver;

  PrefixMapperWrapper(sys::path::Style PathStyle)
      : PM(PathStyle), Saver(Alloc) {}

  void add(const MappedPrefix &Mapping) { return PM.add(Mapping); }

  template <class RangeT> void addRange(const RangeT &Mappings) {
    return PM.addRange(Mappings);
  }

  StringRef map(StringRef Path) {
    SmallString<256> PathBuf;
    PM.map(Path, PathBuf);
    return Saver.save(PathBuf.str());
  }
};

TEST(MappedPrefixTest, getInverse) {
  EXPECT_EQ((MappedPrefix{"b", "a"}), (MappedPrefix{"a", "b"}).getInverse());
  EXPECT_EQ((MappedPrefix{"a", "b"}), (MappedPrefix{"b", "a"}).getInverse());
}

TEST(MappedPrefixTest, getFromJoined) {
  auto split = MappedPrefix::getFromJoined;
  EXPECT_EQ(std::nullopt, split(""));
  EXPECT_EQ(std::nullopt, split("a"));
  EXPECT_EQ(std::nullopt, split("abc"));
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
  EXPECT_EQ(ArrayRef(ExpectedSplit), ArrayRef(ComputedSplit));

  ComputedSplit.clear();
  MappedPrefix::transformJoinedIfValid(JoinedRefs, ComputedSplit);
  EXPECT_EQ(ArrayRef(ExpectedSplit), ArrayRef(ComputedSplit));
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
  EXPECT_EQ(ArrayRef(ExpectedSplit), ArrayRef(ComputedSplit));

  ComputedSplit.clear();
  MappedPrefix::transformJoinedIfValid(JoinedRefs, ComputedSplit);
  EXPECT_EQ(ArrayRef(ExpectedSplit), ArrayRef(ComputedSplit));
}

TEST(PrefixMapperTest, transformPairs) {
  SmallVector<std::pair<std::string, std::string>> PrefixMappings = {
      {"", ""}, {"a", "b"}, {"", "a"}, {"a", ""}, {"=a=b=", "=c=d==="}};
  MappedPrefix ExpectedSplit[] = {
      MappedPrefix{"", ""},
      MappedPrefix{"a", "b"},
      MappedPrefix{"", "a"},
      MappedPrefix{"a", ""},
      MappedPrefix{"=a=b=", "=c=d==="},
  };

  SmallVector<MappedPrefix> ComputedSplit;
  MappedPrefix::transformPairs(PrefixMappings, ComputedSplit);
  EXPECT_EQ(ArrayRef(ExpectedSplit), ArrayRef(ComputedSplit));
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
  auto RangeRef = ArrayRef(Range);
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
  PrefixMapperWrapper PM(PathStyle);
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
  PM.addRange(ArrayRef(Mappings));

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
  PrefixMapperWrapper PM(PathStyle);
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
  PM.addRange(ArrayRef(Mappings));

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
  if (sys::path::is_style_posix(sys::path::Style::native))
    checkPosix(sys::path::Style::native);
  else
    checkWindows(sys::path::Style::native);
}

TEST(PrefixMapperTest, mapTwoArgs) {
  PrefixMapper PM(sys::path::Style::posix);
  MappedPrefix Mappings[] = {{"/old", "/new"}};
  PM.addRange(ArrayRef(Mappings));

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
  PM.addRange(ArrayRef(Mappings));

  MappedPrefix Tests[] = {
      {"/old", "/new"},     {"/old/x/y", "/new/x/y"},
      {"/old", "/new"},     {"relative/path", "relative/path"},
      {"/other", "/other"},
  };

  for (MappedPrefix M : Tests) {
    std::string S = PM.mapToString(M.Old);
    EXPECT_EQ(M.New, S);
  }
}

TEST(PrefixMapperTest, mapInPlace) {
  PrefixMapper PM(sys::path::Style::posix);
  MappedPrefix Mappings[] = {{"/old", "/new"}};
  PM.addRange(ArrayRef(Mappings));

  MappedPrefix Tests[] = {
      {"/old", "/new"},     {"/old/x/y", "/new/x/y"},
      {"/old", "/new"},     {"relative/path", "relative/path"},
      {"/other", "/other"},
  };

  SmallString<128> V;
  std::string S;
  for (MappedPrefix M : Tests) {
    V = M.Old;
    S = M.Old;
    PM.mapInPlace(V);
    PM.mapInPlace(S);
    EXPECT_EQ(M.New, V);
    EXPECT_EQ(M.New, S);
  }
}

} // end namespace
