//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <optional>

namespace test {
enum class Mode { A, B };
} // namespace test

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
  EXPECT_FALSE(O.enable);
  EXPECT_EQ(O.count, 3u);
  EXPECT_EQ(O.limit, std::nullopt);
  EXPECT_EQ(O.mode, test::Mode::A);
  EXPECT_EQ(O.override, std::nullopt);
  EXPECT_EQ(O.ratio, 0.5);
  EXPECT_EQ(O.tristate, std::nullopt);
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
  EXPECT_THAT(Apply({"-lib-enable", "--lib-count=7", "-lib-limit=0",
                     "-lib-mode", "b", "-lib-override", "-lib-ratio", "0.25",
                     "-lib-tristate=Disable", "-lib-path=a=b"}),
              testing::Each(true));
  EXPECT_TRUE(O.enable);
  EXPECT_EQ(O.count, 7u);
  EXPECT_EQ(O.limit, 0u);
  EXPECT_EQ(O.mode, test::Mode::B);
  EXPECT_EQ(O.override, true);
  EXPECT_EQ(O.ratio, 0.25);
  EXPECT_EQ(O.tristate, false);
  EXPECT_EQ(O.Path, "a=b");
  EXPECT_THAT(Apply({"-lib-enable=false"}), testing::Each(true));
  EXPECT_FALSE(O.enable);
  EXPECT_THAT(
      Apply({"-lib-enable=1", "-lib-override=false", "-lib-tristate=Enable"}),
      testing::Each(true));
  EXPECT_TRUE(O.enable);
  EXPECT_EQ(O.override, false);
  EXPECT_EQ(O.tristate, true);
  EXPECT_THAT(Apply({"-lib-tristate=Default"}), testing::Each(true));
  EXPECT_EQ(O.tristate, std::nullopt);

  // A rejected value leaves the member unchanged.
  EXPECT_THAT(Apply({"-lib-enable=2", "-lib-count=-1", "-lib-limit=x",
                     "-lib-mode=c", "-lib-override=y", "-lib-ratio=y"}),
              testing::Each(false));
  EXPECT_TRUE(O.enable);
  EXPECT_EQ(O.count, 7u);
  EXPECT_EQ(O.limit, 0u);
  EXPECT_EQ(O.mode, test::Mode::B);
  EXPECT_EQ(O.override, false);
  EXPECT_EQ(O.ratio, 0.25);
}

// What cl:: sees of the struct, without cl::.
TEST(LibraryOptionsTest, Parser) {
  opt::LibraryOptionsParser P(
      TestLibraryOptions::optTable,
      [](const opt::Arg &A) { return TestLibraryOptions::Global.apply(A); },
      [] { TestLibraryOptions::Global = TestLibraryOptions(); });

  std::vector<std::string> Rows;
  P.forEachOption([&](StringRef Spelling, StringRef MetaVar, StringRef Help) {
    Rows.push_back((Spelling + "|" + MetaVar + "|" + Help).str());
  });
  EXPECT_THAT(Rows,
              testing::ElementsAre(
                  "lib-count|=<value>|An unsigned", "lib-enable||A bool",
                  "lib-limit|=<value>|An optional", "lib-mode|=<a|b>|An enum",
                  "lib-override||An optional bool",
                  "lib-path|=<value>|A string", "lib-ratio|=<value>|A double",
                  "lib-tristate|=<Default|Enable|Disable>|A tri-state"));

  auto Parse = [&](std::initializer_list<const char *> Argv) {
    unsigned Consumed = 0;
    std::string Err = toString(P.parse(Argv, Consumed));
    return std::to_string(Consumed) + " " + Err;
  };
  EXPECT_EQ(Parse({"-lib-count", "5"}), "2 ");
  EXPECT_EQ(TestLibraryOptions::Global.count, 5u);
  EXPECT_EQ(Parse({"-lib-count=x", "-lib-enable"}),
            "1 invalid value 'x' in '-lib-count=x'");
  EXPECT_EQ(Parse({"-lib-count"}),
            "1 option '-lib-count' requires an argument");
  EXPECT_EQ(Parse({"-lib-other"}), "1 unknown argument '-lib-other'");
  EXPECT_EQ(Parse({"-lib-counts=1"}), "1 unknown argument '-lib-counts=1'");
  EXPECT_EQ(Parse({"-lib-enable", "0"}), "1 ");
  EXPECT_EQ(Parse({"-lib-count", "x"}),
            "2 invalid value 'x' in '-lib-count=x'");
  EXPECT_EQ(Parse({"-lib-enable=x"}), "1 invalid value 'x' in '-lib-enable=x'");
  P.reset();
  EXPECT_EQ(TestLibraryOptions::Global.count, 3u);
}

// A static RegisterLibraryOptions connects the struct's Global to cl::.
TEST(LibraryOptionsTest, Register) {
  cl::ResetCommandLineParser();
  opt::RegisterLibraryOptions<TestLibraryOptions> Registration;
  const TestLibraryOptions &G = TestLibraryOptions::Global;
  std::string Path = "-lib-path=q";
  const char *Args[] = {"prog", "-lib-count", "5", "-lib-enable", Path.c_str()};
  EXPECT_TRUE(cl::ParseCommandLineOptions(std::size(Args), Args, "", &nulls()));
  EXPECT_EQ(G.count, 5u);
  EXPECT_TRUE(G.enable);
  // A StringRef member does not refer to the caller's argument.
  Path.assign(Path.size(), 'x');
  EXPECT_EQ(G.Path, "q");
  cl::ResetAllOptionOccurrences();
  EXPECT_EQ(G.count, 3u);
  EXPECT_FALSE(G.enable);
  EXPECT_EQ(G.Path, "p");
  cl::ResetCommandLineParser();
}

} // namespace
