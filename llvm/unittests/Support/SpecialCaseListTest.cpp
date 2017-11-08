//===- SpecialCaseListTest.cpp - Unit tests for SpecialCaseList -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/SpecialCaseList.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class SpecialCaseListTest : public ::testing::Test {
protected:
  std::unique_ptr<SpecialCaseList> makeSpecialCaseList(StringRef List,
                                                       std::string &Error) {
    std::unique_ptr<MemoryBuffer> MB = MemoryBuffer::getMemBuffer(List);
    return SpecialCaseList::create(MB.get(), Error);
  }

  std::unique_ptr<SpecialCaseList> makeSpecialCaseList(StringRef List) {
    std::string Error;
    auto SCL = makeSpecialCaseList(List, Error);
    assert(SCL);
    assert(Error == "");
    return SCL;
  }

  std::string makeSpecialCaseListFile(StringRef Contents) {
    int FD;
    SmallString<64> Path;
    sys::fs::createTemporaryFile("SpecialCaseListTest", "temp", FD, Path);
    raw_fd_ostream OF(FD, true, true);
    OF << Contents;
    OF.close();
    return Path.str();
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

  EXPECT_EQ(3u, SCL->inSectionBlame("", "src", "hello"));
  EXPECT_EQ(4u, SCL->inSectionBlame("", "src", "bye"));
  EXPECT_EQ(5u, SCL->inSectionBlame("", "src", "hi", "category"));
  EXPECT_EQ(6u, SCL->inSectionBlame("", "src", "zzzz", "category"));
  EXPECT_EQ(0u, SCL->inSectionBlame("", "src", "hi"));
  EXPECT_EQ(0u, SCL->inSectionBlame("", "fun", "hello"));
  EXPECT_EQ(0u, SCL->inSectionBlame("", "src", "hello", "category"));
}

TEST_F(SpecialCaseListTest, CorrectErrorLineNumberWithBlankLine) {
  std::string Error;
  EXPECT_EQ(nullptr, makeSpecialCaseList("# This is a comment.\n"
                                         "\n"
                                         "[not valid\n",
                                         Error));
  EXPECT_TRUE(
      ((StringRef)Error).startswith("malformed section header on line 3:"));

  EXPECT_EQ(nullptr, makeSpecialCaseList("\n\n\n"
                                         "[not valid\n",
                                         Error));
  EXPECT_TRUE(
      ((StringRef)Error).startswith("malformed section header on line 4:"));
}

TEST_F(SpecialCaseListTest, SectionRegexErrorHandling) {
  std::string Error;
  EXPECT_EQ(makeSpecialCaseList("[address", Error), nullptr);
  EXPECT_TRUE(((StringRef)Error).startswith("malformed section header "));

  EXPECT_EQ(makeSpecialCaseList("[[]", Error), nullptr);
  EXPECT_TRUE(((StringRef)Error).startswith("malformed regex for section [: "));

  EXPECT_EQ(makeSpecialCaseList("src:=", Error), nullptr);
  EXPECT_TRUE(((StringRef)Error).endswith("Supplied regexp was blank"));
}

TEST_F(SpecialCaseListTest, Section) {
  std::unique_ptr<SpecialCaseList> SCL = makeSpecialCaseList("src:global\n"
                                                             "[sect1|sect2]\n"
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
  EXPECT_EQ("malformed regex in line 1: 'bad[a-': invalid character range",
            Error);
  EXPECT_EQ(nullptr, makeSpecialCaseList("src:a.c\n"
                                   "fun:fun(a\n",
                                   Error));
  EXPECT_EQ("malformed regex in line 2: 'fun(a': parentheses not balanced",
            Error);
  std::vector<std::string> Files(1, "unexisting");
  EXPECT_EQ(nullptr, SpecialCaseList::create(Files, Error));
  EXPECT_EQ(0U, Error.find("can't open file 'unexisting':"));
}

TEST_F(SpecialCaseListTest, EmptySpecialCaseList) {
  std::unique_ptr<SpecialCaseList> SCL = makeSpecialCaseList("");
  EXPECT_FALSE(SCL->inSection("", "foo", "bar"));
}

TEST_F(SpecialCaseListTest, MultipleBlacklists) {
  std::vector<std::string> Files;
  Files.push_back(makeSpecialCaseListFile("src:bar\n"
                                          "src:*foo*\n"
                                          "src:ban=init\n"));
  Files.push_back(makeSpecialCaseListFile("src:baz\n"
                                          "src:*fog*\n"));
  auto SCL = SpecialCaseList::createOrDie(Files);
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
  std::unique_ptr<SpecialCaseList> SCL = makeSpecialCaseList("fun:b.r\n"
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

}
