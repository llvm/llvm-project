//===- SpecialCaseListTest.cpp - Unit tests for SpecialCaseList -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/SpecialCaseList.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using testing::HasSubstr;
using testing::Pair;
using testing::StartsWith;
using namespace llvm;

namespace {

class SpecialCaseListTest : public ::testing::Test {
protected:
  std::unique_ptr<SpecialCaseList>
  makeSpecialCaseList(StringRef List, std::string &Error, int Version = 0) {
    auto S = List.str();
    if (Version)
      S = (Twine("#!special-case-list-v") + Twine(Version) + "\n" + S).str();
    std::unique_ptr<MemoryBuffer> MB = MemoryBuffer::getMemBuffer(S);
    return SpecialCaseList::create(MB.get(), Error);
  }

  std::unique_ptr<SpecialCaseList> makeSpecialCaseList(StringRef List,
                                                       int Version = 0) {
    std::string Error;
    auto SCL = makeSpecialCaseList(List, Error, Version);
    assert(SCL);
    assert(Error == "");
    return SCL;
  }

  std::string makeSpecialCaseListFile(StringRef Contents, int Version = 0) {
    int FD;
    SmallString<64> Path;
    sys::fs::createTemporaryFile("SpecialCaseListTest", "temp", FD, Path);
    raw_fd_ostream OF(FD, true, true);
    if (Version)
      OF << "#!special-case-list-v" << Version << "\n";
    OF << Contents;
    OF.close();
    return std::string(Path.str());
  }
};

TEST_F(SpecialCaseListTest, Basic) {
  std::unique_ptr<SpecialCaseList> SCL =
      makeSpecialCaseList("# This is a comment.\n"
                          "\n"
                          "src:hello\n"
                          "src:bye\n"
                          "src:hi=category\n"
                          "src:z*=category\n");
  EXPECT_TRUE(SCL->inSection("", "src", "hello"));
  EXPECT_TRUE(SCL->inSection("", "src", "bye"));
  EXPECT_TRUE(SCL->inSection("", "src", "hi", "category"));
  EXPECT_TRUE(SCL->inSection("", "src", "zzzz", "category"));
  EXPECT_FALSE(SCL->inSection("", "src", "hi"));
  EXPECT_FALSE(SCL->inSection("", "fun", "hello"));
  EXPECT_FALSE(SCL->inSection("", "src", "hello", "category"));

  EXPECT_THAT(SCL->inSectionBlame("", "src", "hello"), Pair(0u, 3u));
  EXPECT_THAT(SCL->inSectionBlame("", "src", "bye"), Pair(0u, 4u));
  EXPECT_THAT(SCL->inSectionBlame("", "src", "hi", "category"), Pair(0u, 5u));
  EXPECT_THAT(SCL->inSectionBlame("", "src", "zzzz", "category"), Pair(0u, 6u));
  EXPECT_THAT(SCL->inSectionBlame("", "src", "hi"), Pair(0u, 0u));
  EXPECT_THAT(SCL->inSectionBlame("", "fun", "hello"), Pair(0u, 0u));
  EXPECT_THAT(SCL->inSectionBlame("", "src", "hello", "category"),
              Pair(0u, 0u));
}

TEST_F(SpecialCaseListTest, CorrectErrorLineNumberWithBlankLine) {
  std::string Error;
  EXPECT_EQ(nullptr, makeSpecialCaseList("# This is a comment.\n"
                                         "\n"
                                         "[not valid\n",
                                         Error));
  EXPECT_THAT(Error, StartsWith("malformed section header on line 3:"));

  EXPECT_EQ(nullptr, makeSpecialCaseList("\n\n\n"
                                         "[not valid\n",
                                         Error));
  EXPECT_THAT(Error, StartsWith("malformed section header on line 4:"));
}

TEST_F(SpecialCaseListTest, SectionGlobErrorHandling) {
  std::string Error;
  EXPECT_EQ(makeSpecialCaseList("[address", Error), nullptr);
  EXPECT_THAT(Error, StartsWith("malformed section header "));

  EXPECT_EQ(makeSpecialCaseList("[[]", Error), nullptr);
  EXPECT_EQ(
      Error,
      "malformed section at line 1: '[': invalid glob pattern, unmatched '['");

  EXPECT_EQ(makeSpecialCaseList("src:=", Error), nullptr);
  EXPECT_THAT(Error, HasSubstr("Supplied glob was blank"));
}

TEST_F(SpecialCaseListTest, Section) {
  std::unique_ptr<SpecialCaseList> SCL = makeSpecialCaseList("src:global\n"
                                                             "[{sect1,sect2}]\n"
                                                             "src:test1\n"
                                                             "[sect3*]\n"
                                                             "src:test2\n");
  EXPECT_TRUE(SCL->inSection("arbitrary", "src", "global"));
  EXPECT_TRUE(SCL->inSection("", "src", "global"));
  EXPECT_TRUE(SCL->inSection("sect1", "src", "test1"));
  EXPECT_FALSE(SCL->inSection("sect1-arbitrary", "src", "test1"));
  EXPECT_FALSE(SCL->inSection("sect", "src", "test1"));
  EXPECT_FALSE(SCL->inSection("sect1", "src", "test2"));
  EXPECT_TRUE(SCL->inSection("sect2", "src", "test1"));
  EXPECT_TRUE(SCL->inSection("sect3", "src", "test2"));
  EXPECT_TRUE(SCL->inSection("sect3-arbitrary", "src", "test2"));
  EXPECT_FALSE(SCL->inSection("", "src", "test1"));
  EXPECT_FALSE(SCL->inSection("", "src", "test2"));
}

TEST_F(SpecialCaseListTest, GlobalInit) {
  std::unique_ptr<SpecialCaseList> SCL =
      makeSpecialCaseList("global:foo=init\n");
  EXPECT_FALSE(SCL->inSection("", "global", "foo"));
  EXPECT_FALSE(SCL->inSection("", "global", "bar"));
  EXPECT_TRUE(SCL->inSection("", "global", "foo", "init"));
  EXPECT_FALSE(SCL->inSection("", "global", "bar", "init"));

  SCL = makeSpecialCaseList("type:t2=init\n");
  EXPECT_FALSE(SCL->inSection("", "type", "t1"));
  EXPECT_FALSE(SCL->inSection("", "type", "t2"));
  EXPECT_FALSE(SCL->inSection("", "type", "t1", "init"));
  EXPECT_TRUE(SCL->inSection("", "type", "t2", "init"));

  SCL = makeSpecialCaseList("src:hello=init\n");
  EXPECT_FALSE(SCL->inSection("", "src", "hello"));
  EXPECT_FALSE(SCL->inSection("", "src", "bye"));
  EXPECT_TRUE(SCL->inSection("", "src", "hello", "init"));
  EXPECT_FALSE(SCL->inSection("", "src", "bye", "init"));
}

TEST_F(SpecialCaseListTest, Substring) {
  std::unique_ptr<SpecialCaseList> SCL = makeSpecialCaseList("src:hello\n"
                                                             "fun:foo\n"
                                                             "global:bar\n");
  EXPECT_FALSE(SCL->inSection("", "src", "othello"));
  EXPECT_FALSE(SCL->inSection("", "fun", "tomfoolery"));
  EXPECT_FALSE(SCL->inSection("", "global", "bartender"));

  SCL = makeSpecialCaseList("fun:*foo*\n");
  EXPECT_TRUE(SCL->inSection("", "fun", "tomfoolery"));
  EXPECT_TRUE(SCL->inSection("", "fun", "foobar"));
}

TEST_F(SpecialCaseListTest, InvalidSpecialCaseList) {
  std::string Error;
  EXPECT_EQ(nullptr, makeSpecialCaseList("badline", Error));
  EXPECT_EQ("malformed line 1: 'badline'", Error);
  EXPECT_EQ(nullptr, makeSpecialCaseList("src:bad[a-", Error));
  EXPECT_EQ(
      "malformed glob in line 1: 'bad[a-': invalid glob pattern, unmatched '['",
      Error);
  std::vector<std::string> Files(1, "unexisting");
  EXPECT_EQ(nullptr,
            SpecialCaseList::create(Files, *vfs::getRealFileSystem(), Error));
  EXPECT_THAT(Error, StartsWith("can't open file 'unexisting':"));
}

TEST_F(SpecialCaseListTest, EmptySpecialCaseList) {
  std::unique_ptr<SpecialCaseList> SCL = makeSpecialCaseList("");
  EXPECT_FALSE(SCL->inSection("", "foo", "bar"));
}

TEST_F(SpecialCaseListTest, MultipleExclusions) {
  std::vector<std::string> Files;
  Files.push_back(makeSpecialCaseListFile("src:bar\n"
                                          "src:*foo*\n"
                                          "src:ban=init\n"));
  Files.push_back(makeSpecialCaseListFile("src:baz\n"
                                          "src:*fog*\n"));
  auto SCL = SpecialCaseList::createOrDie(Files, *vfs::getRealFileSystem());
  EXPECT_TRUE(SCL->inSection("", "src", "bar"));
  EXPECT_TRUE(SCL->inSection("", "src", "baz"));
  EXPECT_FALSE(SCL->inSection("", "src", "ban"));
  EXPECT_TRUE(SCL->inSection("", "src", "ban", "init"));
  EXPECT_TRUE(SCL->inSection("", "src", "tomfoolery"));
  EXPECT_TRUE(SCL->inSection("", "src", "tomfoglery"));
  for (auto &Path : Files)
    sys::fs::remove(Path);
}

TEST_F(SpecialCaseListTest, NoTrigramsInRules) {
  std::unique_ptr<SpecialCaseList> SCL = makeSpecialCaseList("fun:b?r\n"
                                                             "fun:za*az\n");
  EXPECT_TRUE(SCL->inSection("", "fun", "bar"));
  EXPECT_FALSE(SCL->inSection("", "fun", "baz"));
  EXPECT_TRUE(SCL->inSection("", "fun", "zakaz"));
  EXPECT_FALSE(SCL->inSection("", "fun", "zaraza"));
}

TEST_F(SpecialCaseListTest, NoTrigramsInARule) {
  std::unique_ptr<SpecialCaseList> SCL = makeSpecialCaseList("fun:*bar*\n"
                                                             "fun:za*az\n");
  EXPECT_TRUE(SCL->inSection("", "fun", "abara"));
  EXPECT_FALSE(SCL->inSection("", "fun", "bor"));
  EXPECT_TRUE(SCL->inSection("", "fun", "zakaz"));
  EXPECT_FALSE(SCL->inSection("", "fun", "zaraza"));
}

TEST_F(SpecialCaseListTest, RepetitiveRule) {
  std::unique_ptr<SpecialCaseList> SCL = makeSpecialCaseList("fun:*bar*bar*bar*bar*\n"
                                                             "fun:bar*\n");
  EXPECT_TRUE(SCL->inSection("", "fun", "bara"));
  EXPECT_FALSE(SCL->inSection("", "fun", "abara"));
  EXPECT_TRUE(SCL->inSection("", "fun", "barbarbarbar"));
  EXPECT_TRUE(SCL->inSection("", "fun", "abarbarbarbar"));
  EXPECT_FALSE(SCL->inSection("", "fun", "abarbarbar"));
}

TEST_F(SpecialCaseListTest, SpecialSymbolRule) {
  std::unique_ptr<SpecialCaseList> SCL = makeSpecialCaseList("src:*c\\+\\+abi*\n");
  EXPECT_TRUE(SCL->inSection("", "src", "c++abi"));
  EXPECT_FALSE(SCL->inSection("", "src", "c\\+\\+abi"));
}

TEST_F(SpecialCaseListTest, PopularTrigram) {
  std::unique_ptr<SpecialCaseList> SCL = makeSpecialCaseList("fun:*aaaaaa*\n"
                                                             "fun:*aaaaa*\n"
                                                             "fun:*aaaa*\n"
                                                             "fun:*aaa*\n");
  EXPECT_TRUE(SCL->inSection("", "fun", "aaa"));
  EXPECT_TRUE(SCL->inSection("", "fun", "aaaa"));
  EXPECT_TRUE(SCL->inSection("", "fun", "aaaabbbaaa"));
}

TEST_F(SpecialCaseListTest, EscapedSymbols) {
  std::unique_ptr<SpecialCaseList> SCL = makeSpecialCaseList("src:*c\\+\\+abi*\n"
                                                             "src:*hello\\\\world*\n");
  EXPECT_TRUE(SCL->inSection("", "src", "dir/c++abi"));
  EXPECT_FALSE(SCL->inSection("", "src", "dir/c\\+\\+abi"));
  EXPECT_FALSE(SCL->inSection("", "src", "c\\+\\+abi"));
  EXPECT_TRUE(SCL->inSection("", "src", "C:\\hello\\world"));
  EXPECT_TRUE(SCL->inSection("", "src", "hello\\world"));
  EXPECT_FALSE(SCL->inSection("", "src", "hello\\\\world"));
}

TEST_F(SpecialCaseListTest, Version1) {
  std::unique_ptr<SpecialCaseList> SCL =
      makeSpecialCaseList("[sect1|sect2]\n"
                          // Does not match foo!
                          "fun:foo.*\n"
                          "fun:abc|def\n"
                          "fun:b.r\n",
                          /*Version=*/1);

  EXPECT_TRUE(SCL->inSection("sect1", "fun", "fooz"));
  EXPECT_TRUE(SCL->inSection("sect2", "fun", "fooz"));
  EXPECT_FALSE(SCL->inSection("sect3", "fun", "fooz"));

  // `foo.*` does not match `foo` because the pattern is translated to `foo..*`
  EXPECT_FALSE(SCL->inSection("sect1", "fun", "foo"));

  EXPECT_TRUE(SCL->inSection("sect1", "fun", "abc"));
  EXPECT_TRUE(SCL->inSection("sect2", "fun", "abc"));
  EXPECT_FALSE(SCL->inSection("sect3", "fun", "abc"));

  EXPECT_TRUE(SCL->inSection("sect1", "fun", "def"));
  EXPECT_TRUE(SCL->inSection("sect2", "fun", "def"));
  EXPECT_FALSE(SCL->inSection("sect3", "fun", "def"));

  EXPECT_TRUE(SCL->inSection("sect1", "fun", "bar"));
  EXPECT_TRUE(SCL->inSection("sect2", "fun", "bar"));
  EXPECT_FALSE(SCL->inSection("sect3", "fun", "bar"));
}

TEST_F(SpecialCaseListTest, Version2) {
  std::unique_ptr<SpecialCaseList> SCL = makeSpecialCaseList("[{sect1,sect2}]\n"
                                                             "fun:foo*\n"
                                                             "fun:{abc,def}\n"
                                                             "fun:b?r\n");
  EXPECT_TRUE(SCL->inSection("sect1", "fun", "fooz"));
  EXPECT_TRUE(SCL->inSection("sect2", "fun", "fooz"));
  EXPECT_FALSE(SCL->inSection("sect3", "fun", "fooz"));

  EXPECT_TRUE(SCL->inSection("sect1", "fun", "foo"));
  EXPECT_TRUE(SCL->inSection("sect2", "fun", "foo"));
  EXPECT_FALSE(SCL->inSection("sect3", "fun", "foo"));

  EXPECT_TRUE(SCL->inSection("sect1", "fun", "abc"));
  EXPECT_TRUE(SCL->inSection("sect2", "fun", "abc"));
  EXPECT_FALSE(SCL->inSection("sect3", "fun", "abc"));

  EXPECT_TRUE(SCL->inSection("sect1", "fun", "def"));
  EXPECT_TRUE(SCL->inSection("sect2", "fun", "def"));
  EXPECT_FALSE(SCL->inSection("sect3", "fun", "def"));

  EXPECT_TRUE(SCL->inSection("sect1", "fun", "bar"));
  EXPECT_TRUE(SCL->inSection("sect2", "fun", "bar"));
  EXPECT_FALSE(SCL->inSection("sect3", "fun", "bar"));
}

TEST_F(SpecialCaseListTest, DotSlash) {
  std::unique_ptr<SpecialCaseList> SCL2 = makeSpecialCaseList("[dot]\n"
                                                              "fun:./foo\n"
                                                              "src:./bar\n"
                                                              "[not]\n"
                                                              "fun:foo\n"
                                                              "src:bar\n");
  std::unique_ptr<SpecialCaseList> SCL3 = makeSpecialCaseList("[dot]\n"
                                                              "fun:./foo\n"
                                                              "src:./bar\n"
                                                              "[not]\n"
                                                              "fun:foo\n"
                                                              "src:bar\n",
                                                              /*Version=*/3);

  EXPECT_TRUE(SCL2->inSection("dot", "fun", "./foo"));
  EXPECT_TRUE(SCL3->inSection("dot", "fun", "./foo"));

  EXPECT_FALSE(SCL2->inSection("dot", "fun", "foo"));
  EXPECT_FALSE(SCL3->inSection("dot", "fun", "foo"));

  EXPECT_TRUE(SCL2->inSection("dot", "src", "./bar"));
  EXPECT_FALSE(SCL3->inSection("dot", "src", "./bar"));

  EXPECT_FALSE(SCL2->inSection("dot", "src", "bar"));
  EXPECT_FALSE(SCL3->inSection("dot", "src", "bar"));

  EXPECT_FALSE(SCL2->inSection("not", "fun", "./foo"));
  EXPECT_FALSE(SCL3->inSection("not", "fun", "./foo"));

  EXPECT_TRUE(SCL2->inSection("not", "fun", "foo"));
  EXPECT_TRUE(SCL3->inSection("not", "fun", "foo"));

  EXPECT_FALSE(SCL2->inSection("not", "src", "./bar"));
  EXPECT_TRUE(SCL3->inSection("not", "src", "./bar"));

  EXPECT_TRUE(SCL2->inSection("not", "src", "bar"));
  EXPECT_TRUE(SCL3->inSection("not", "src", "bar"));
}

TEST_F(SpecialCaseListTest, LinesInSection) {
  std::unique_ptr<SpecialCaseList> SCL = makeSpecialCaseList("fun:foo\n"
                                                             "fun:bar\n"
                                                             "fun:foo\n");
  EXPECT_THAT(SCL->inSectionBlame("sect1", "fun", "foo"), Pair(0u, 3u));
  EXPECT_THAT(SCL->inSectionBlame("sect1", "fun", "bar"), Pair(0u, 2u));
}

TEST_F(SpecialCaseListTest, LinesCrossSection) {
  std::unique_ptr<SpecialCaseList> SCL = makeSpecialCaseList("fun:foo\n"
                                                             "fun:bar\n"
                                                             "fun:foo\n"
                                                             "[sect1]\n"
                                                             "fun:bar\n");
  EXPECT_THAT(SCL->inSectionBlame("sect1", "fun", "foo"), Pair(0u, 3u));
  EXPECT_THAT(SCL->inSectionBlame("sect1", "fun", "bar"), Pair(0u, 5u));
}

TEST_F(SpecialCaseListTest, Blame) {
  std::unique_ptr<SpecialCaseList> SCL = makeSpecialCaseList("[sect1]\n"
                                                             "src:foo*\n"
                                                             "[sect1]\n"
                                                             "src:bar*\n"
                                                             "src:def\n"
                                                             "[sect2]\n"
                                                             "src:def\n"
                                                             "src:de*\n");
  EXPECT_TRUE(SCL->inSection("sect1", "src", "fooz"));
  EXPECT_TRUE(SCL->inSection("sect1", "src", "barz"));
  EXPECT_FALSE(SCL->inSection("sect2", "src", "fooz"));

  EXPECT_TRUE(SCL->inSection("sect2", "src", "def"));
  EXPECT_TRUE(SCL->inSection("sect1", "src", "def"));

  EXPECT_THAT(SCL->inSectionBlame("sect1", "src", "fooz"), Pair(0u, 2u));
  EXPECT_THAT(SCL->inSectionBlame("sect1", "src", "barz"), Pair(0u, 4u));
  EXPECT_THAT(SCL->inSectionBlame("sect1", "src", "def"), Pair(0u, 5u));
  EXPECT_THAT(SCL->inSectionBlame("sect2", "src", "def"), Pair(0u, 8u));
  EXPECT_THAT(SCL->inSectionBlame("sect2", "src", "dez"), Pair(0u, 8u));
}

TEST_F(SpecialCaseListTest, FileIdx) {
  std::vector<std::string> Files;
  Files.push_back(makeSpecialCaseListFile("src:bar\n"
                                          "src:*foo*\n"
                                          "src:ban=init\n"
                                          "src:baz\n"
                                          "src:*def\n"));
  Files.push_back(makeSpecialCaseListFile("src:baz\n"
                                          "src:car\n"
                                          "src:def*"));
  auto SCL = SpecialCaseList::createOrDie(Files, *vfs::getRealFileSystem());
  EXPECT_THAT(SCL->inSectionBlame("", "src", "bar"), Pair(0u, 1u));
  EXPECT_THAT(SCL->inSectionBlame("", "src", "fooaaaaaa"), Pair(0u, 2u));
  EXPECT_THAT(SCL->inSectionBlame("", "src", "ban", "init"), Pair(0u, 3u));
  EXPECT_THAT(SCL->inSectionBlame("", "src", "baz"), Pair(1u, 1u));
  EXPECT_THAT(SCL->inSectionBlame("", "src", "car"), Pair(1u, 2u));
  EXPECT_THAT(SCL->inSectionBlame("", "src", "aaaadef"), Pair(0u, 5u));
  EXPECT_THAT(SCL->inSectionBlame("", "src", "defaaaaa"), Pair(1u, 3u));
  for (auto &Path : Files)
    sys::fs::remove(Path);
}

} // namespace
