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
#include "llvm/Support/LibraryOptionsContext.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#define OPTIONS_STRUCT_DEFS
#include "LibraryOpts.inc"

using namespace llvm;
using test::TestLibraryOptions;

namespace {

// The struct -gen-opt-parser-defs generates: every spelling sets its member.
TEST(LibraryOptionsTest, Apply) {
  BumpPtrAllocator Alloc;
  TestLibraryOptions O;
  EXPECT_FALSE(O.enable);
  EXPECT_EQ(O.count, 3u);
  EXPECT_EQ(O.limit, std::nullopt);
  EXPECT_THAT(O.list, testing::IsEmpty());
  EXPECT_EQ(O.mode, test::Mode::A);
  EXPECT_THAT(O.modes, testing::IsEmpty());
  EXPECT_EQ(O.override, BoolOrDefault::Default);
  EXPECT_EQ(O.print, std::nullopt);
  EXPECT_EQ(O.ratio, 0.5);
  EXPECT_EQ(O.tristate, BoolOrDefault::Default);
  EXPECT_EQ(O.Path, "p");

  auto Apply = [&](std::initializer_list<const char *> Argv) {
    unsigned MissingIndex, MissingCount;
    opt::InputArgList Args = TestLibraryOptions::optTable().ParseArgs(
        Argv, MissingIndex, MissingCount);
    std::vector<bool> Applied;
    for (const opt::Arg *A : Args)
      Applied.push_back(O.apply(*A, Alloc));
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
  EXPECT_EQ(O.override, BoolOrDefault::True);
  EXPECT_EQ(O.ratio, 0.25);
  EXPECT_EQ(O.tristate, BoolOrDefault::False);
  EXPECT_EQ(O.Path, "a=b");
  EXPECT_THAT(Apply({"-lib-enable=false"}), testing::Each(true));
  EXPECT_FALSE(O.enable);
  EXPECT_THAT(
      Apply({"-lib-enable=1", "-lib-override=false", "-lib-tristate=Enable"}),
      testing::Each(true));
  EXPECT_TRUE(O.enable);
  EXPECT_EQ(O.override, BoolOrDefault::False);
  EXPECT_EQ(O.tristate, BoolOrDefault::True);
  EXPECT_THAT(Apply({"-lib-tristate=Default"}), testing::Each(true));
  EXPECT_EQ(O.tristate, BoolOrDefault::Default);
  EXPECT_THAT(Apply({"-lib-print=a"}), testing::Each(true));
  EXPECT_EQ(O.print, test::Mode::A);
  EXPECT_THAT(Apply({"-lib-print"}), testing::Each(true));
  EXPECT_EQ(O.print, test::Mode::B);

  // Each occurrence of a list appends its comma-separated values.
  EXPECT_THAT(Apply({"-lib-list=1,2", "-lib-list", "3"}), testing::Each(true));
  EXPECT_THAT(O.list, testing::ElementsAre(1u, 2u, 3u));
  EXPECT_THAT(Apply({"-lib-modes=b,a", "-lib-modes", "b"}),
              testing::Each(true));
  EXPECT_THAT(O.modes, testing::ElementsAre(test::Mode::B, test::Mode::A,
                                            test::Mode::B));

  // A rejected value leaves the member unchanged.
  EXPECT_THAT(
      Apply({"-lib-enable=2", "-lib-count=-1", "-lib-limit=x", "-lib-list=4,x",
             "-lib-list=", "-lib-mode=c", "-lib-modes=a,c", "-lib-override=y",
             "-lib-print=c", "-lib-ratio=y"}),
      testing::Each(false));
  EXPECT_TRUE(O.enable);
  EXPECT_EQ(O.count, 7u);
  EXPECT_EQ(O.limit, 0u);
  EXPECT_THAT(O.list, testing::ElementsAre(1u, 2u, 3u));
  EXPECT_EQ(O.mode, test::Mode::B);
  EXPECT_THAT(O.modes, testing::ElementsAre(test::Mode::B, test::Mode::A,
                                            test::Mode::B));
  EXPECT_EQ(O.override, BoolOrDefault::False);
  EXPECT_EQ(O.print, test::Mode::B);
  EXPECT_EQ(O.ratio, 0.25);
}

// An alias sets its target's member. A Flag alias supplies the value in its
// AliasArgs.
TEST(LibraryOptionsTest, Alias) {
  BumpPtrAllocator Alloc;
  TestLibraryOptions O;
  auto Apply = [&](std::initializer_list<const char *> Argv) {
    unsigned MissingIndex, MissingCount;
    opt::InputArgList Args = TestLibraryOptions::optTable().ParseArgs(
        Argv, MissingIndex, MissingCount);
    std::vector<bool> Applied;
    for (const opt::Arg *A : Args)
      Applied.push_back(O.apply(*A, Alloc));
    return Applied;
  };

  // A FlagOrEq alias of a FlagOrEq target, and a SeparateOrEq of a
  // SeparateOrEq.
  EXPECT_THAT(Apply({"-lib-old-enable", "--lib-old-count=9"}),
              testing::Each(true));
  EXPECT_TRUE(O.enable);
  EXPECT_EQ(O.count, 9u);
  EXPECT_THAT(Apply({"-lib-old-enable=false", "-lib-old-count", "4"}),
              testing::Each(true));
  EXPECT_FALSE(O.enable);
  EXPECT_EQ(O.count, 4u);
  EXPECT_THAT(Apply({"-lib-old-print"}), testing::Each(true));
  EXPECT_EQ(O.print, test::Mode::B);
  EXPECT_THAT(Apply({"-lib-old-print=a"}), testing::Each(true));
  EXPECT_EQ(O.print, test::Mode::A);

  // A Flag alias of a bool, BoolOrDefault, SeparateOrEq or enum target.
  EXPECT_THAT(Apply({"-lib-enable", "-lib-disable"}), testing::Each(true));
  EXPECT_FALSE(O.enable);
  EXPECT_THAT(Apply({"--lib-disable"}), testing::Each(true));
  EXPECT_THAT(Apply({"-lib-no-override"}), testing::Each(true));
  EXPECT_EQ(O.override, BoolOrDefault::False);
  EXPECT_THAT(Apply({"-lib-count-one"}), testing::Each(true));
  EXPECT_EQ(O.count, 1u);
  EXPECT_THAT(Apply({"-lib-mode-b"}), testing::Each(true));
  EXPECT_EQ(O.mode, test::Mode::B);

  // The Arg is the target's, whichever spelling was given.
  unsigned MissingIndex, MissingCount;
  opt::InputArgList Args = TestLibraryOptions::optTable().ParseArgs(
      {"-lib-disable", "-lib-old-count=2"}, MissingIndex, MissingCount);
  std::vector<std::string> Seen;
  for (const opt::Arg *A : Args)
    Seen.push_back((A->getOption().getName() + "=" + A->getValue()).str());
  EXPECT_THAT(Seen, testing::ElementsAre("lib-enable=false", "lib-count=2"));
}

TEST(LibraryOptionsTest, BoolOrDefault) {
  EXPECT_TRUE(valueOr(BoolOrDefault::Default, true));
  EXPECT_FALSE(valueOr(BoolOrDefault::Default, false));
  EXPECT_TRUE(valueOr(BoolOrDefault::True, false));
  EXPECT_FALSE(valueOr(BoolOrDefault::False, true));
}

// What cl:: sees of the struct, without cl::.
TEST(LibraryOptionsTest, Parser) {
  opt::LibraryOptionsParser P(std::in_place_type<TestLibraryOptions>);

  std::vector<std::string> Rows;
  P.forEachOption([&](StringRef Spelling, StringRef MetaVar, StringRef Help) {
    Rows.push_back((Spelling + "|" + MetaVar + "|" + Help).str());
  });
  // An alias is listed too, including a Flag, with the target's metavar only
  // for a kind that takes a value. -lib-old-not-in-build is absent with its
  // target.
  EXPECT_THAT(
      Rows,
      testing::ElementsAre(
          "lib-count-one||", "lib-count|=<value>|An unsigned",
          "lib-disable||Same as -lib-enable=false", "lib-enable||A bool",
          "lib-in-build||Available", "lib-limit|=<value>|An optional",
          "lib-list|=<value>|A list", "lib-mode-b||",
          "lib-modes|=<a|b>|An enum list", "lib-mode|=<a|b>|An enum",
          "lib-no-override||", "lib-old-count|=<value>|", "lib-old-enable||",
          "lib-old-in-build||", "lib-old-print||",
          "lib-override||An optional bool", "lib-path|=<value>|A string",
          "lib-print|[=<a|b>]|A flag or enum", "lib-ratio|=<value>|A double",
          "lib-tristate|=<Default|Enable|Disable>|A tri-state"));

  BumpPtrAllocator Alloc;
  auto Parse = [&](std::initializer_list<const char *> Argv) {
    unsigned Consumed = 0;
    std::string Err = toString(P.parse(P.global(), Argv, Consumed, Alloc));
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
  EXPECT_EQ(Parse({"-lib-in-build"}), "1 ");
  EXPECT_TRUE(TestLibraryOptions::Global.in_build);
  EXPECT_EQ(Parse({"-lib-not-in-build"}),
            "1 unknown argument '-lib-not-in-build'");
  EXPECT_EQ(Parse({"-lib-enable", "0"}), "1 ");
  // An alias dispatches like its target; a Flag alias takes no value.
  EXPECT_EQ(Parse({"-lib-old-count", "6"}), "2 ");
  EXPECT_EQ(TestLibraryOptions::Global.count, 6u);
  EXPECT_EQ(Parse({"-lib-old-count=x"}),
            "1 invalid value 'x' in '-lib-old-count=x'");
  EXPECT_EQ(Parse({"-lib-count-one", "7"}), "1 ");
  EXPECT_EQ(TestLibraryOptions::Global.count, 1u);
  EXPECT_EQ(Parse({"-lib-disable"}), "1 ");
  EXPECT_FALSE(TestLibraryOptions::Global.enable);
  EXPECT_EQ(Parse({"-lib-disable=true"}),
            "1 unknown argument '-lib-disable=true'");
  EXPECT_EQ(Parse({"-lib-old-in-build"}), "1 ");
  EXPECT_TRUE(TestLibraryOptions::Global.in_build);
  EXPECT_EQ(Parse({"-lib-old-not-in-build"}),
            "1 unknown argument '-lib-old-not-in-build'");
  EXPECT_EQ(Parse({"-lib-print", "a"}), "1 ");
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
  const char *Args[] = {"prog",        "-lib-count", "5",
                        "-lib-enable", Path.c_str(), "-lib-list=1,2"};
  EXPECT_TRUE(cl::ParseCommandLineOptions(std::size(Args), Args, "", &nulls()));
  EXPECT_EQ(G.count, 5u);
  EXPECT_TRUE(G.enable);
  EXPECT_THAT(G.list, testing::ElementsAre(1u, 2u));
  // A StringRef member does not refer to the caller's argument.
  Path.assign(Path.size(), 'x');
  EXPECT_EQ(G.Path, "q");
  cl::ResetAllOptionOccurrences();
  EXPECT_EQ(G.count, 3u);
  EXPECT_FALSE(G.enable);
  EXPECT_EQ(G.Path, "p");
  EXPECT_THAT(G.list, testing::IsEmpty());
  // Options parse again after a reset frees their storage.
  const char *Again[] = {"prog", "-lib-path=r", "-lib-list=3", "-lib-list=4"};
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(std::size(Again), Again, "", &nulls()));
  EXPECT_EQ(G.Path, "r");
  EXPECT_THAT(G.list, testing::ElementsAre(3u, 4u));
  cl::ResetCommandLineParser();
}

// parseCommandLineOptions sets the options of a library that is not
// per-context in its Global, which a context without values reads.
TEST(LibraryOptionsTest, ParseIntoGlobal) {
  cl::ResetCommandLineParser();
  opt::RegisterLibraryOptions<TestLibraryOptions> Registration;
  cl::LibraryOptionsContext V;
  const char *Args[] = {"prog", "-lib-count=5"};
  EXPECT_TRUE(cl::parseCommandLineOptions(V, Args, "", &nulls()));
  EXPECT_EQ(TestLibraryOptions::Global.count, 5u);
  EXPECT_EQ(&V.get<TestLibraryOptions>(), &TestLibraryOptions::Global);
  cl::ResetCommandLineParser();
}

// Values that override Global, as an LLVMContext holds them.
TEST(LibraryOptionsTest, Values) {
  cl::ResetCommandLineParser();
  opt::RegisterLibraryOptions<TestLibraryOptions> Registration;
  opt::LibraryOptionsParser L{std::in_place_type<TestLibraryOptions>};
  auto Parse = [&](cl::LibraryOptionsContext &V, ArrayRef<const char *> Args) {
    unsigned N;
    return toString(V.parse(L, Args, N));
  };
  cl::LibraryOptionsContext V;
  std::string Path = "-lib-path=q";
  EXPECT_EQ(Parse(V, {"-lib-count", "7"}), "");
  EXPECT_EQ(Parse(V, Path.c_str()), "");
  EXPECT_EQ(Parse(V, "--lib-list=1,2"), "");
  Path.assign(Path.size(), ' ');
  EXPECT_EQ(TestLibraryOptions::Global.count, 3u);
  EXPECT_EQ(Parse(V, "-lib-count=x"), "invalid value 'x' in '-lib-count=x'");

  // assign() replays the arguments, which outlive the caller's, and discards
  // the earlier values.
  cl::LibraryOptionsContext Copy;
  EXPECT_EQ(Parse(Copy, "-lib-list=9"), "");
  Copy.assign(V);
  const TestLibraryOptions &C = Copy.get<TestLibraryOptions>();
  EXPECT_EQ(C.count, 7u);
  EXPECT_EQ(C.Path, "q");
  EXPECT_THAT(C.list, testing::ElementsAre(1u, 2u));
  EXPECT_EQ(Copy.args().size(), V.args().size());
  cl::ResetCommandLineParser();
}

// Aliases in the arguments of a context, as -mllvm values are: the target's
// member is set, and assign() replays the spellings that were given.
TEST(LibraryOptionsTest, AliasValues) {
  cl::ResetCommandLineParser();
  opt::RegisterLibraryOptions<TestLibraryOptions> Registration;
  opt::LibraryOptionsParser L{std::in_place_type<TestLibraryOptions>};
  auto Parse = [&](cl::LibraryOptionsContext &V, ArrayRef<const char *> Args) {
    unsigned N;
    return toString(V.parse(L, Args, N));
  };
  cl::LibraryOptionsContext V;
  EXPECT_EQ(Parse(V, "-lib-enable"), "");
  EXPECT_EQ(Parse(V, "-lib-disable"), "");
  EXPECT_EQ(Parse(V, {"-lib-old-count", "4"}), "");
  EXPECT_EQ(Parse(V, "-lib-mode-b"), "");
  EXPECT_EQ(Parse(V, "-lib-no-override"), "");
  EXPECT_EQ(Parse(V, "-lib-disable=true"),
            "unknown argument '-lib-disable=true'");
  const TestLibraryOptions &O = V.get<TestLibraryOptions>();
  EXPECT_FALSE(O.enable);
  EXPECT_EQ(O.count, 4u);
  EXPECT_EQ(O.mode, test::Mode::B);
  EXPECT_EQ(O.override, BoolOrDefault::False);
  EXPECT_EQ(TestLibraryOptions::Global.count, 3u);

  cl::LibraryOptionsContext Copy;
  EXPECT_EQ(Parse(Copy, "-lib-count-one"), "");
  Copy.assign(V);
  const TestLibraryOptions &C = Copy.get<TestLibraryOptions>();
  EXPECT_FALSE(C.enable);
  EXPECT_EQ(C.count, 4u);
  EXPECT_EQ(C.mode, test::Mode::B);
  EXPECT_EQ(C.override, BoolOrDefault::False);
  std::vector<std::string> Replayed(Copy.args().begin(), Copy.args().end());
  EXPECT_THAT(Replayed, testing::ElementsAre(
                            "-lib-enable", "-lib-disable", "-lib-old-count",
                            "4", "-lib-mode-b", "-lib-no-override"));
  cl::ResetCommandLineParser();
}

// cl:: dispatches every spelling of an alias, a Flag included, to the struct,
// and lists the aliases that have help text in -help-hidden.
TEST(LibraryOptionsTest, AliasRegister) {
  cl::ResetCommandLineParser();
  opt::RegisterLibraryOptions<TestLibraryOptions> Registration;
  const TestLibraryOptions &G = TestLibraryOptions::Global;
  const char *Args[] = {
      "prog",         "-lib-old-count", "5",          "-lib-old-enable",
      "-lib-disable", "-lib-count-one", "-lib-mode-b"};
  EXPECT_TRUE(cl::ParseCommandLineOptions(std::size(Args), Args, "", &nulls()));
  EXPECT_EQ(G.count, 1u);
  EXPECT_FALSE(G.enable);
  EXPECT_EQ(G.mode, test::Mode::B);
  cl::ResetAllOptionOccurrences();
  EXPECT_EQ(G.mode, test::Mode::A);

  // A Flag alias takes no value, and the alias of an option that this build
  // excludes is unknown.
  std::string Errs;
  raw_string_ostream OS(Errs);
  const char *Bad[] = {"prog", "-lib-disable=true", "-lib-old-not-in-build"};
  EXPECT_FALSE(cl::ParseCommandLineOptions(std::size(Bad), Bad, "", &OS));
  EXPECT_EQ(Errs, "prog: unknown argument '-lib-disable=true'\n"
                  "prog: Unknown command line argument "
                  "'-lib-old-not-in-build'.  Try: 'prog --help'\n");
  cl::ResetCommandLineParser();
}

#if GTEST_HAS_DEATH_TEST
// An alias is indexed like any option, so a cl::opt cannot reuse its name.
TEST(LibraryOptionsTest, AliasConflict) {
  cl::ResetCommandLineParser();
  opt::RegisterLibraryOptions<TestLibraryOptions> Registration;
  const char *Args[] = {"prog"};
  EXPECT_DEATH(
      {
        cl::opt<bool> Clash("lib-disable");
        cl::ParseCommandLineOptions(std::size(Args), Args);
      },
      "Option 'lib-disable' registered more than once");
  cl::ResetCommandLineParser();
}
#endif

} // namespace
