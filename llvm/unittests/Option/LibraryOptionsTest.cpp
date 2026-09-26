//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define OPTIONS_STRUCT_DECL
#include "LibraryOpts.inc"

#include "llvm/Option/ArgList.h"
#include "llvm/Option/LibraryOptions.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#define OPTIONS_STRUCT_DEFS
#include "LibraryOpts.inc"

using namespace llvm;
using test::TestLibraryOptions;

namespace {

// The struct -gen-opt-parser-defs generates: every spelling sets its member.
TEST(LibraryOptionsTest, Apply) {
  TestLibraryOptions O;
  EXPECT_FALSE(O.lib_enable);
  EXPECT_EQ(O.lib_count, 3u);
  EXPECT_EQ(O.lib_ratio, 0.5);
  EXPECT_EQ(O.Path, "p");

  auto Apply = [&](std::initializer_list<const char *> Argv) {
    unsigned MissingIndex, MissingCount;
    opt::InputArgList Args = TestLibraryOptions::optTable().ParseArgs(
        Argv, MissingIndex, MissingCount);
    std::vector<bool> Applied;
    for (const opt::Arg *A : Args)
      Applied.push_back(O.apply(*A));
    return Applied;
  };
  EXPECT_THAT(Apply({"-lib-enable", "--lib-count=7", "-lib-ratio", "0.25",
                     "-lib-path=a=b"}),
              testing::Each(true));
  EXPECT_TRUE(O.lib_enable);
  EXPECT_EQ(O.lib_count, 7u);
  EXPECT_EQ(O.lib_ratio, 0.25);
  EXPECT_EQ(O.Path, "a=b");
  EXPECT_THAT(Apply({"-lib-enable=0"}), testing::Each(true));
  EXPECT_FALSE(O.lib_enable);
  EXPECT_THAT(Apply({"-lib-enable=1"}), testing::Each(true));
  EXPECT_TRUE(O.lib_enable);

  // A rejected value leaves the member unchanged.
  EXPECT_THAT(Apply({"-lib-enable=2", "-lib-count=x", "-lib-ratio=y"}),
              testing::Each(false));
  EXPECT_TRUE(O.lib_enable);
  EXPECT_EQ(O.lib_count, 7u);
  EXPECT_EQ(O.lib_ratio, 0.25);
}

// What cl:: sees of the struct, without cl::.
TEST(LibraryOptionsTest, Parser) {
  opt::LibraryOptionsParser P(
      TestLibraryOptions::optTable,
      [](const opt::Arg &A) { return TestLibraryOptions::Global.apply(A); },
      [] { TestLibraryOptions::Global = TestLibraryOptions(); });

  std::vector<std::string> Rows;
  P.forEachOption([&](StringRef Spelling, StringRef MetaVar, StringRef Help,
                      bool Hidden) {
    Rows.push_back(
        (Spelling + "|" + MetaVar + "|" + Help + (Hidden ? "|h" : "")).str());
  });
  EXPECT_THAT(Rows, testing::ElementsAre(
                        "lib-count=|<value>|An unsigned", "lib-count||",
                        "lib-enable=|<value>|", "lib-enable||A bool|h",
                        "lib-path=|<value>|A string|h", "lib-path||",
                        "lib-ratio=|<value>|A double|h", "lib-ratio||"));

  auto Parse = [&](std::initializer_list<const char *> Argv) {
    unsigned Consumed = 0;
    std::string Err = toString(P.parse(Argv, Consumed));
    return std::to_string(Consumed) + " " + Err;
  };
  EXPECT_EQ(Parse({"-lib-count", "5"}), "2 ");
  EXPECT_EQ(TestLibraryOptions::Global.lib_count, 5u);
  EXPECT_EQ(Parse({"-lib-count=x", "-lib-enable"}),
            "1 invalid value 'x' in '-lib-count=x'");
  EXPECT_EQ(Parse({"-lib-count"}),
            "1 option '-lib-count' requires an argument");
  EXPECT_EQ(Parse({"-lib-other"}), "1 unknown argument '-lib-other'");
  P.reset();
  EXPECT_EQ(TestLibraryOptions::Global.lib_count, 3u);
}

// A static RegisterLibraryOptions connects the struct's Global to cl::.
TEST(LibraryOptionsTest, Register) {
  cl::ResetCommandLineParser();
  opt::RegisterLibraryOptions<TestLibraryOptions> Registration;
  const TestLibraryOptions &G = TestLibraryOptions::Global;
  std::string Path = "-lib-path=q";
  const char *Args[] = {"prog", "-lib-count", "5", "-lib-enable", Path.c_str()};
  EXPECT_TRUE(cl::ParseCommandLineOptions(std::size(Args), Args, "", &nulls()));
  EXPECT_EQ(G.lib_count, 5u);
  EXPECT_TRUE(G.lib_enable);
  // A StringRef member does not refer to the caller's argument.
  Path.assign(Path.size(), 'x');
  EXPECT_EQ(G.Path, "q");
  cl::ResetAllOptionOccurrences();
  EXPECT_EQ(G.lib_count, 3u);
  EXPECT_FALSE(G.lib_enable);
  EXPECT_EQ(G.Path, "p");
  cl::ResetCommandLineParser();
}

} // namespace
