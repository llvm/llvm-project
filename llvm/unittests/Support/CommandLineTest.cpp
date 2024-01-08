//===- llvm/unittest/Support/CommandLineTest.cpp - CommandLine tests ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/config.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <fstream>
#include <stdlib.h>
#include <string>

using namespace llvm;
using llvm::unittest::TempDir;
using llvm::unittest::TempFile;

namespace {

MATCHER(StringEquality, "Checks if two char* are equal as strings") {
  return std::string(std::get<0>(arg)) == std::string(std::get<1>(arg));
}

class TempEnvVar {
 public:
  TempEnvVar(const char *name, const char *value)
      : name(name) {
    const char *old_value = getenv(name);
    EXPECT_EQ(nullptr, old_value) << old_value;
#if HAVE_SETENV
    setenv(name, value, true);
#endif
  }

  ~TempEnvVar() {
#if HAVE_SETENV
    // Assume setenv and unsetenv come together.
    unsetenv(name);
#else
    (void)name; // Suppress -Wunused-private-field.
#endif
  }

 private:
  const char *const name;
};

template <typename T, typename Base = cl::opt<T>>
class StackOption : public Base {
public:
  template <class... Ts>
  explicit StackOption(Ts &&... Ms) : Base(std::forward<Ts>(Ms)...) {}

  ~StackOption() override { this->removeArgument(); }

  template <class DT> StackOption<T> &operator=(const DT &V) {
    Base::operator=(V);
    return *this;
  }
};

class StackSubCommand : public cl::SubCommand {
public:
  StackSubCommand(StringRef Name,
                  StringRef Description = StringRef())
      : SubCommand(Name, Description) {}

  StackSubCommand() : SubCommand() {}

  ~StackSubCommand() { unregisterSubCommand(); }
};


cl::OptionCategory TestCategory("Test Options", "Description");
TEST(CommandLineTest, ModifyExisitingOption) {
  StackOption<int> TestOption("test-option", cl::desc("old description"));

  static const char Description[] = "New description";
  static const char ArgString[] = "new-test-option";
  static const char ValueString[] = "Integer";

  StringMap<cl::Option *> &Map =
      cl::getRegisteredOptions(cl::SubCommand::getTopLevel());

  ASSERT_EQ(Map.count("test-option"), 1u) << "Could not find option in map.";

  cl::Option *Retrieved = Map["test-option"];
  ASSERT_EQ(&TestOption, Retrieved) << "Retrieved wrong option.";

  ASSERT_NE(Retrieved->Categories.end(),
            find_if(Retrieved->Categories,
                    [&](const llvm::cl::OptionCategory *Cat) {
                      return Cat == &cl::getGeneralCategory();
                    }))
      << "Incorrect default option category.";

  Retrieved->addCategory(TestCategory);
  ASSERT_NE(Retrieved->Categories.end(),
            find_if(Retrieved->Categories,
                    [&](const llvm::cl::OptionCategory *Cat) {
                      return Cat == &TestCategory;
                    }))
      << "Failed to modify option's option category.";

  Retrieved->setDescription(Description);
  ASSERT_STREQ(Retrieved->HelpStr.data(), Description)
      << "Changing option description failed.";

  Retrieved->setArgStr(ArgString);
  ASSERT_STREQ(ArgString, Retrieved->ArgStr.data())
      << "Failed to modify option's Argument string.";

  Retrieved->setValueStr(ValueString);
  ASSERT_STREQ(Retrieved->ValueStr.data(), ValueString)
      << "Failed to modify option's Value string.";

  Retrieved->setHiddenFlag(cl::Hidden);
  ASSERT_EQ(cl::Hidden, TestOption.getOptionHiddenFlag()) <<
    "Failed to modify option's hidden flag.";
}

TEST(CommandLineTest, UseOptionCategory) {
  StackOption<int> TestOption2("test-option", cl::cat(TestCategory));

  ASSERT_NE(TestOption2.Categories.end(),
            find_if(TestOption2.Categories,
                         [&](const llvm::cl::OptionCategory *Cat) {
                           return Cat == &TestCategory;
                         }))
      << "Failed to assign Option Category.";
}

TEST(CommandLineTest, UseMultipleCategories) {
  StackOption<int> TestOption2("test-option2", cl::cat(TestCategory),
                               cl::cat(cl::getGeneralCategory()),
                               cl::cat(cl::getGeneralCategory()));

  // Make sure cl::getGeneralCategory() wasn't added twice.
  ASSERT_EQ(TestOption2.Categories.size(), 2U);

  ASSERT_NE(TestOption2.Categories.end(),
            find_if(TestOption2.Categories,
                         [&](const llvm::cl::OptionCategory *Cat) {
                           return Cat == &TestCategory;
                         }))
      << "Failed to assign Option Category.";
  ASSERT_NE(TestOption2.Categories.end(),
            find_if(TestOption2.Categories,
                    [&](const llvm::cl::OptionCategory *Cat) {
                      return Cat == &cl::getGeneralCategory();
                    }))
      << "Failed to assign General Category.";

  cl::OptionCategory AnotherCategory("Additional test Options", "Description");
  StackOption<int> TestOption("test-option", cl::cat(TestCategory),
                              cl::cat(AnotherCategory));
  ASSERT_EQ(TestOption.Categories.end(),
            find_if(TestOption.Categories,
                    [&](const llvm::cl::OptionCategory *Cat) {
                      return Cat == &cl::getGeneralCategory();
                    }))
      << "Failed to remove General Category.";
  ASSERT_NE(TestOption.Categories.end(),
            find_if(TestOption.Categories,
                         [&](const llvm::cl::OptionCategory *Cat) {
                           return Cat == &TestCategory;
                         }))
      << "Failed to assign Option Category.";
  ASSERT_NE(TestOption.Categories.end(),
            find_if(TestOption.Categories,
                         [&](const llvm::cl::OptionCategory *Cat) {
                           return Cat == &AnotherCategory;
                         }))
      << "Failed to assign Another Category.";
}

typedef void ParserFunction(StringRef Source, StringSaver &Saver,
                            SmallVectorImpl<const char *> &NewArgv,
                            bool MarkEOLs);

void testCommandLineTokenizer(ParserFunction *parse, StringRef Input,
                              ArrayRef<const char *> Output,
                              bool MarkEOLs = false) {
  SmallVector<const char *, 0> Actual;
  BumpPtrAllocator A;
  StringSaver Saver(A);
  parse(Input, Saver, Actual, MarkEOLs);
  EXPECT_EQ(Output.size(), Actual.size());
  for (unsigned I = 0, E = Actual.size(); I != E; ++I) {
    if (I < Output.size()) {
      EXPECT_STREQ(Output[I], Actual[I]);
    }
  }
}

TEST(CommandLineTest, TokenizeGNUCommandLine) {
  const char Input[] =
      "foo\\ bar \"foo bar\" \'foo bar\' 'foo\\\\bar' -DFOO=bar\\(\\) "
      "foo\"bar\"baz C:\\\\src\\\\foo.cpp \"C:\\src\\foo.cpp\"";
  const char *const Output[] = {
      "foo bar",     "foo bar",   "foo bar",          "foo\\bar",
      "-DFOO=bar()", "foobarbaz", "C:\\src\\foo.cpp", "C:srcfoo.cpp"};
  testCommandLineTokenizer(cl::TokenizeGNUCommandLine, Input, Output);
}

TEST(CommandLineTest, TokenizeWindowsCommandLine1) {
  const char Input[] =
      R"(a\b c\\d e\\"f g" h\"i j\\\"k "lmn" o pqr "st \"u" \v)";
  const char *const Output[] = { "a\\b", "c\\\\d", "e\\f g", "h\"i", "j\\\"k",
                                 "lmn", "o", "pqr", "st \"u", "\\v" };
  testCommandLineTokenizer(cl::TokenizeWindowsCommandLine, Input, Output);
}

TEST(CommandLineTest, TokenizeWindowsCommandLine2) {
  const char Input[] = "clang -c -DFOO=\"\"\"ABC\"\"\" x.cpp";
  const char *const Output[] = { "clang", "-c", "-DFOO=\"ABC\"", "x.cpp"};
  testCommandLineTokenizer(cl::TokenizeWindowsCommandLine, Input, Output);
}

TEST(CommandLineTest, TokenizeWindowsCommandLineQuotedLastArgument) {
  // Whitespace at the end of the command line doesn't cause an empty last word
  const char Input0[] = R"(a b c d )";
  const char *const Output0[] = {"a", "b", "c", "d"};
  testCommandLineTokenizer(cl::TokenizeWindowsCommandLine, Input0, Output0);

  // But an explicit "" does
  const char Input1[] = R"(a b c d "")";
  const char *const Output1[] = {"a", "b", "c", "d", ""};
  testCommandLineTokenizer(cl::TokenizeWindowsCommandLine, Input1, Output1);

  // An unterminated quoted string is also emitted as an argument word, empty
  // or not
  const char Input2[] = R"(a b c d ")";
  const char *const Output2[] = {"a", "b", "c", "d", ""};
  testCommandLineTokenizer(cl::TokenizeWindowsCommandLine, Input2, Output2);
  const char Input3[] = R"(a b c d "text)";
  const char *const Output3[] = {"a", "b", "c", "d", "text"};
  testCommandLineTokenizer(cl::TokenizeWindowsCommandLine, Input3, Output3);
}

TEST(CommandLineTest, TokenizeWindowsCommandLineExeName) {
  const char Input1[] =
      R"("C:\Program Files\Whatever\"clang.exe z.c -DY=\"x\")";
  const char *const Output1[] = {"C:\\Program Files\\Whatever\\clang.exe",
                                 "z.c", "-DY=\"x\""};
  testCommandLineTokenizer(cl::TokenizeWindowsCommandLineFull, Input1, Output1);

  const char Input2[] = "\"a\\\"b c\\\"d\n\"e\\\"f g\\\"h\n";
  const char *const Output2[] = {"a\\b", "c\"d", nullptr,
                                 "e\\f", "g\"h", nullptr};
  testCommandLineTokenizer(cl::TokenizeWindowsCommandLineFull, Input2, Output2,
                           /*MarkEOLs=*/true);

  const char Input3[] = R"(\\server\share\subdir\clang.exe)";
  const char *const Output3[] = {"\\\\server\\share\\subdir\\clang.exe"};
  testCommandLineTokenizer(cl::TokenizeWindowsCommandLineFull, Input3, Output3);
}

TEST(CommandLineTest, TokenizeAndMarkEOLs) {
  // Clang uses EOL marking in response files to support options that consume
  // the rest of the arguments on the current line, but do not consume arguments
  // from subsequent lines. For example, given these rsp files contents:
  // /c /Zi /O2
  // /Oy- /link /debug /opt:ref
  // /Zc:ThreadsafeStatics-
  //
  // clang-cl needs to treat "/debug /opt:ref" as linker flags, and everything
  // else as compiler flags. The tokenizer inserts nullptr sentinels into the
  // output so that clang-cl can find the end of the current line.
  const char Input[] = "clang -Xclang foo\n\nfoo\"bar\"baz\n x.cpp\n";
  const char *const Output[] = {"clang", "-Xclang", "foo",
                                nullptr, nullptr,   "foobarbaz",
                                nullptr, "x.cpp",   nullptr};
  testCommandLineTokenizer(cl::TokenizeWindowsCommandLine, Input, Output,
                           /*MarkEOLs=*/true);
  testCommandLineTokenizer(cl::TokenizeGNUCommandLine, Input, Output,
                           /*MarkEOLs=*/true);
}

TEST(CommandLineTest, TokenizeConfigFile1) {
  const char *Input = "\\";
  const char *const Output[] = { "\\" };
  testCommandLineTokenizer(cl::tokenizeConfigFile, Input, Output);
}

TEST(CommandLineTest, TokenizeConfigFile2) {
  const char *Input = "\\abc";
  const char *const Output[] = { "abc" };
  testCommandLineTokenizer(cl::tokenizeConfigFile, Input, Output);
}

TEST(CommandLineTest, TokenizeConfigFile3) {
  const char *Input = "abc\\";
  const char *const Output[] = { "abc\\" };
  testCommandLineTokenizer(cl::tokenizeConfigFile, Input, Output);
}

TEST(CommandLineTest, TokenizeConfigFile4) {
  const char *Input = "abc\\\n123";
  const char *const Output[] = { "abc123" };
  testCommandLineTokenizer(cl::tokenizeConfigFile, Input, Output);
}

TEST(CommandLineTest, TokenizeConfigFile5) {
  const char *Input = "abc\\\r\n123";
  const char *const Output[] = { "abc123" };
  testCommandLineTokenizer(cl::tokenizeConfigFile, Input, Output);
}

TEST(CommandLineTest, TokenizeConfigFile6) {
  const char *Input = "abc\\\n";
  const char *const Output[] = { "abc" };
  testCommandLineTokenizer(cl::tokenizeConfigFile, Input, Output);
}

TEST(CommandLineTest, TokenizeConfigFile7) {
  const char *Input = "abc\\\r\n";
  const char *const Output[] = { "abc" };
  testCommandLineTokenizer(cl::tokenizeConfigFile, Input, Output);
}

TEST(CommandLineTest, TokenizeConfigFile8) {
  SmallVector<const char *, 0> Actual;
  BumpPtrAllocator A;
  StringSaver Saver(A);
  cl::tokenizeConfigFile("\\\n", Saver, Actual, /*MarkEOLs=*/false);
  EXPECT_TRUE(Actual.empty());
}

TEST(CommandLineTest, TokenizeConfigFile9) {
  SmallVector<const char *, 0> Actual;
  BumpPtrAllocator A;
  StringSaver Saver(A);
  cl::tokenizeConfigFile("\\\r\n", Saver, Actual, /*MarkEOLs=*/false);
  EXPECT_TRUE(Actual.empty());
}

TEST(CommandLineTest, TokenizeConfigFile10) {
  const char *Input = "\\\nabc";
  const char *const Output[] = { "abc" };
  testCommandLineTokenizer(cl::tokenizeConfigFile, Input, Output);
}

TEST(CommandLineTest, TokenizeConfigFile11) {
  const char *Input = "\\\r\nabc";
  const char *const Output[] = { "abc" };
  testCommandLineTokenizer(cl::tokenizeConfigFile, Input, Output);
}

TEST(CommandLineTest, AliasesWithArguments) {
  static const size_t ARGC = 3;
  const char *const Inputs[][ARGC] = {
    { "-tool", "-actual=x", "-extra" },
    { "-tool", "-actual", "x" },
    { "-tool", "-alias=x", "-extra" },
    { "-tool", "-alias", "x" }
  };

  for (size_t i = 0, e = std::size(Inputs); i < e; ++i) {
    StackOption<std::string> Actual("actual");
    StackOption<bool> Extra("extra");
    StackOption<std::string> Input(cl::Positional);

    cl::alias Alias("alias", llvm::cl::aliasopt(Actual));

    cl::ParseCommandLineOptions(ARGC, Inputs[i]);
    EXPECT_EQ("x", Actual);
    EXPECT_EQ(0, Input.getNumOccurrences());

    Alias.removeArgument();
  }
}

void testAliasRequired(int argc, const char *const *argv) {
  StackOption<std::string> Option("option", cl::Required);
  cl::alias Alias("o", llvm::cl::aliasopt(Option));

  cl::ParseCommandLineOptions(argc, argv);
  EXPECT_EQ("x", Option);
  EXPECT_EQ(1, Option.getNumOccurrences());

  Alias.removeArgument();
}

TEST(CommandLineTest, AliasRequired) {
  const char *opts1[] = { "-tool", "-option=x" };
  const char *opts2[] = { "-tool", "-o", "x" };
  testAliasRequired(std::size(opts1), opts1);
  testAliasRequired(std::size(opts2), opts2);
}

TEST(CommandLineTest, HideUnrelatedOptions) {
  StackOption<int> TestOption1("hide-option-1");
  StackOption<int> TestOption2("hide-option-2", cl::cat(TestCategory));

  cl::HideUnrelatedOptions(TestCategory);

  ASSERT_EQ(cl::ReallyHidden, TestOption1.getOptionHiddenFlag())
      << "Failed to hide extra option.";
  ASSERT_EQ(cl::NotHidden, TestOption2.getOptionHiddenFlag())
      << "Hid extra option that should be visable.";

  StringMap<cl::Option *> &Map =
      cl::getRegisteredOptions(cl::SubCommand::getTopLevel());
  ASSERT_TRUE(Map.count("help") == (size_t)0 ||
              cl::NotHidden == Map["help"]->getOptionHiddenFlag())
      << "Hid default option that should be visable.";
}

cl::OptionCategory TestCategory2("Test Options set 2", "Description");

TEST(CommandLineTest, HideUnrelatedOptionsMulti) {
  StackOption<int> TestOption1("multi-hide-option-1");
  StackOption<int> TestOption2("multi-hide-option-2", cl::cat(TestCategory));
  StackOption<int> TestOption3("multi-hide-option-3", cl::cat(TestCategory2));

  const cl::OptionCategory *VisibleCategories[] = {&TestCategory,
                                                   &TestCategory2};

  cl::HideUnrelatedOptions(ArrayRef(VisibleCategories));

  ASSERT_EQ(cl::ReallyHidden, TestOption1.getOptionHiddenFlag())
      << "Failed to hide extra option.";
  ASSERT_EQ(cl::NotHidden, TestOption2.getOptionHiddenFlag())
      << "Hid extra option that should be visable.";
  ASSERT_EQ(cl::NotHidden, TestOption3.getOptionHiddenFlag())
      << "Hid extra option that should be visable.";

  StringMap<cl::Option *> &Map =
      cl::getRegisteredOptions(cl::SubCommand::getTopLevel());
  ASSERT_TRUE(Map.count("help") == (size_t)0 ||
              cl::NotHidden == Map["help"]->getOptionHiddenFlag())
      << "Hid default option that should be visable.";
}

TEST(CommandLineTest, SetMultiValues) {
  StackOption<int> Option("option");
  const char *args[] = {"prog", "-option=1", "-option=2"};
  EXPECT_TRUE(cl::ParseCommandLineOptions(std::size(args), args, StringRef(),
                                          &llvm::nulls()));
  EXPECT_EQ(Option, 2);
}

TEST(CommandLineTest, SetValueInSubcategories) {
  cl::ResetCommandLineParser();

  StackSubCommand SC1("sc1", "First subcommand");
  StackSubCommand SC2("sc2", "Second subcommand");

  StackOption<bool> TopLevelOpt("top-level", cl::init(false));
  StackOption<bool> SC1Opt("sc1", cl::sub(SC1), cl::init(false));
  StackOption<bool> SC2Opt("sc2", cl::sub(SC2), cl::init(false));

  EXPECT_FALSE(TopLevelOpt);
  EXPECT_FALSE(SC1Opt);
  EXPECT_FALSE(SC2Opt);
  const char *args[] = {"prog", "-top-level"};
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args, StringRef(), &llvm::nulls()));
  EXPECT_TRUE(TopLevelOpt);
  EXPECT_FALSE(SC1Opt);
  EXPECT_FALSE(SC2Opt);

  TopLevelOpt = false;

  cl::ResetAllOptionOccurrences();
  EXPECT_FALSE(TopLevelOpt);
  EXPECT_FALSE(SC1Opt);
  EXPECT_FALSE(SC2Opt);
  const char *args2[] = {"prog", "sc1", "-sc1"};
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(3, args2, StringRef(), &llvm::nulls()));
  EXPECT_FALSE(TopLevelOpt);
  EXPECT_TRUE(SC1Opt);
  EXPECT_FALSE(SC2Opt);

  SC1Opt = false;

  cl::ResetAllOptionOccurrences();
  EXPECT_FALSE(TopLevelOpt);
  EXPECT_FALSE(SC1Opt);
  EXPECT_FALSE(SC2Opt);
  const char *args3[] = {"prog", "sc2", "-sc2"};
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(3, args3, StringRef(), &llvm::nulls()));
  EXPECT_FALSE(TopLevelOpt);
  EXPECT_FALSE(SC1Opt);
  EXPECT_TRUE(SC2Opt);
}

TEST(CommandLineTest, LookupFailsInWrongSubCommand) {
  cl::ResetCommandLineParser();

  StackSubCommand SC1("sc1", "First subcommand");
  StackSubCommand SC2("sc2", "Second subcommand");

  StackOption<bool> SC1Opt("sc1", cl::sub(SC1), cl::init(false));
  StackOption<bool> SC2Opt("sc2", cl::sub(SC2), cl::init(false));

  std::string Errs;
  raw_string_ostream OS(Errs);

  const char *args[] = {"prog", "sc1", "-sc2"};
  EXPECT_FALSE(cl::ParseCommandLineOptions(3, args, StringRef(), &OS));
  OS.flush();
  EXPECT_FALSE(Errs.empty());
}

TEST(CommandLineTest, TopLevelOptInSubcommand) {
  enum LiteralOptionEnum {
    foo,
    bar,
    baz,
  };

  cl::ResetCommandLineParser();

  // This is a top-level option and not associated with a subcommand.
  // A command line using subcommand should parse both subcommand options and
  // top-level options.  A valid use case is that users of llvm command line
  // tools should be able to specify top-level options defined in any library.
  StackOption<std::string> TopLevelOpt("str", cl::init("txt"),
                                       cl::desc("A top-level option."));

  StackSubCommand SC("sc", "Subcommand");
  StackOption<std::string> PositionalOpt(
      cl::Positional, cl::desc("positional argument test coverage"),
      cl::sub(SC));
  StackOption<LiteralOptionEnum> LiteralOpt(
      cl::desc("literal argument test coverage"), cl::sub(SC), cl::init(bar),
      cl::values(clEnumVal(foo, "foo"), clEnumVal(bar, "bar"),
                 clEnumVal(baz, "baz")));
  StackOption<bool> EnableOpt("enable", cl::sub(SC), cl::init(false));
  StackOption<int> ThresholdOpt("threshold", cl::sub(SC), cl::init(1));

  const char *PositionalOptVal = "input-file";
  const char *args[] = {"prog",    "sc",        PositionalOptVal,
                        "-enable", "--str=csv", "--threshold=2"};

  // cl::ParseCommandLineOptions returns true on success. Otherwise, it will
  // print the error message to stderr and exit in this setting (`Errs` ostream
  // is not set).
  ASSERT_TRUE(cl::ParseCommandLineOptions(sizeof(args) / sizeof(args[0]), args,
                                          StringRef()));
  EXPECT_STREQ(PositionalOpt.getValue().c_str(), PositionalOptVal);
  EXPECT_TRUE(EnableOpt);
  // Tests that the value of `str` option is `csv` as specified.
  EXPECT_STREQ(TopLevelOpt.getValue().c_str(), "csv");
  EXPECT_EQ(ThresholdOpt, 2);

  for (auto &[LiteralOptVal, WantLiteralOpt] :
       {std::pair{"--bar", bar}, {"--foo", foo}, {"--baz", baz}}) {
    const char *args[] = {"prog", "sc", LiteralOptVal};
    ASSERT_TRUE(cl::ParseCommandLineOptions(sizeof(args) / sizeof(args[0]),
                                            args, StringRef()));

    // Tests that literal options are parsed correctly.
    EXPECT_EQ(LiteralOpt, WantLiteralOpt);
  }
}

TEST(CommandLineTest, AddToAllSubCommands) {
  cl::ResetCommandLineParser();

  StackSubCommand SC1("sc1", "First subcommand");
  StackOption<bool> AllOpt("everywhere", cl::sub(cl::SubCommand::getAll()),
                           cl::init(false));
  StackSubCommand SC2("sc2", "Second subcommand");

  const char *args[] = {"prog", "-everywhere"};
  const char *args2[] = {"prog", "sc1", "-everywhere"};
  const char *args3[] = {"prog", "sc2", "-everywhere"};

  std::string Errs;
  raw_string_ostream OS(Errs);

  EXPECT_FALSE(AllOpt);
  EXPECT_TRUE(cl::ParseCommandLineOptions(2, args, StringRef(), &OS));
  EXPECT_TRUE(AllOpt);

  AllOpt = false;

  cl::ResetAllOptionOccurrences();
  EXPECT_FALSE(AllOpt);
  EXPECT_TRUE(cl::ParseCommandLineOptions(3, args2, StringRef(), &OS));
  EXPECT_TRUE(AllOpt);

  AllOpt = false;

  cl::ResetAllOptionOccurrences();
  EXPECT_FALSE(AllOpt);
  EXPECT_TRUE(cl::ParseCommandLineOptions(3, args3, StringRef(), &OS));
  EXPECT_TRUE(AllOpt);

  // Since all parsing succeeded, the error message should be empty.
  OS.flush();
  EXPECT_TRUE(Errs.empty());
}

TEST(CommandLineTest, ReparseCommandLineOptions) {
  cl::ResetCommandLineParser();

  StackOption<bool> TopLevelOpt(
      "top-level", cl::sub(cl::SubCommand::getTopLevel()), cl::init(false));

  const char *args[] = {"prog", "-top-level"};

  EXPECT_FALSE(TopLevelOpt);
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args, StringRef(), &llvm::nulls()));
  EXPECT_TRUE(TopLevelOpt);

  TopLevelOpt = false;

  cl::ResetAllOptionOccurrences();
  EXPECT_FALSE(TopLevelOpt);
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args, StringRef(), &llvm::nulls()));
  EXPECT_TRUE(TopLevelOpt);
}

TEST(CommandLineTest, RemoveFromRegularSubCommand) {
  cl::ResetCommandLineParser();

  StackSubCommand SC("sc", "Subcommand");
  StackOption<bool> RemoveOption("remove-option", cl::sub(SC), cl::init(false));
  StackOption<bool> KeepOption("keep-option", cl::sub(SC), cl::init(false));

  const char *args[] = {"prog", "sc", "-remove-option"};

  std::string Errs;
  raw_string_ostream OS(Errs);

  EXPECT_FALSE(RemoveOption);
  EXPECT_TRUE(cl::ParseCommandLineOptions(3, args, StringRef(), &OS));
  EXPECT_TRUE(RemoveOption);
  OS.flush();
  EXPECT_TRUE(Errs.empty());

  RemoveOption.removeArgument();

  cl::ResetAllOptionOccurrences();
  EXPECT_FALSE(cl::ParseCommandLineOptions(3, args, StringRef(), &OS));
  OS.flush();
  EXPECT_FALSE(Errs.empty());
}

TEST(CommandLineTest, RemoveFromTopLevelSubCommand) {
  cl::ResetCommandLineParser();

  StackOption<bool> TopLevelRemove("top-level-remove",
                                   cl::sub(cl::SubCommand::getTopLevel()),
                                   cl::init(false));
  StackOption<bool> TopLevelKeep("top-level-keep",
                                 cl::sub(cl::SubCommand::getTopLevel()),
                                 cl::init(false));

  const char *args[] = {"prog", "-top-level-remove"};

  EXPECT_FALSE(TopLevelRemove);
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args, StringRef(), &llvm::nulls()));
  EXPECT_TRUE(TopLevelRemove);

  TopLevelRemove.removeArgument();

  cl::ResetAllOptionOccurrences();
  EXPECT_FALSE(
      cl::ParseCommandLineOptions(2, args, StringRef(), &llvm::nulls()));
}

TEST(CommandLineTest, RemoveFromAllSubCommands) {
  cl::ResetCommandLineParser();

  StackSubCommand SC1("sc1", "First Subcommand");
  StackSubCommand SC2("sc2", "Second Subcommand");
  StackOption<bool> RemoveOption(
      "remove-option", cl::sub(cl::SubCommand::getAll()), cl::init(false));
  StackOption<bool> KeepOption("keep-option", cl::sub(cl::SubCommand::getAll()),
                               cl::init(false));

  const char *args0[] = {"prog", "-remove-option"};
  const char *args1[] = {"prog", "sc1", "-remove-option"};
  const char *args2[] = {"prog", "sc2", "-remove-option"};

  // It should work for all subcommands including the top-level.
  EXPECT_FALSE(RemoveOption);
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args0, StringRef(), &llvm::nulls()));
  EXPECT_TRUE(RemoveOption);

  RemoveOption = false;

  cl::ResetAllOptionOccurrences();
  EXPECT_FALSE(RemoveOption);
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(3, args1, StringRef(), &llvm::nulls()));
  EXPECT_TRUE(RemoveOption);

  RemoveOption = false;

  cl::ResetAllOptionOccurrences();
  EXPECT_FALSE(RemoveOption);
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(3, args2, StringRef(), &llvm::nulls()));
  EXPECT_TRUE(RemoveOption);

  RemoveOption.removeArgument();

  // It should not work for any subcommands including the top-level.
  cl::ResetAllOptionOccurrences();
  EXPECT_FALSE(
      cl::ParseCommandLineOptions(2, args0, StringRef(), &llvm::nulls()));
  cl::ResetAllOptionOccurrences();
  EXPECT_FALSE(
      cl::ParseCommandLineOptions(3, args1, StringRef(), &llvm::nulls()));
  cl::ResetAllOptionOccurrences();
  EXPECT_FALSE(
      cl::ParseCommandLineOptions(3, args2, StringRef(), &llvm::nulls()));
}

TEST(CommandLineTest, GetRegisteredSubcommands) {
  cl::ResetCommandLineParser();

  StackSubCommand SC1("sc1", "First Subcommand");
  StackOption<bool> Opt1("opt1", cl::sub(SC1), cl::init(false));
  StackSubCommand SC2("sc2", "Second subcommand");
  StackOption<bool> Opt2("opt2", cl::sub(SC2), cl::init(false));

  const char *args0[] = {"prog", "sc1"};
  const char *args1[] = {"prog", "sc2"};

  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args0, StringRef(), &llvm::nulls()));
  EXPECT_FALSE(Opt1);
  EXPECT_FALSE(Opt2);
  for (auto *S : cl::getRegisteredSubcommands()) {
    if (*S) {
      EXPECT_EQ("sc1", S->getName());
    }
  }

  cl::ResetAllOptionOccurrences();
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args1, StringRef(), &llvm::nulls()));
  EXPECT_FALSE(Opt1);
  EXPECT_FALSE(Opt2);
  for (auto *S : cl::getRegisteredSubcommands()) {
    if (*S) {
      EXPECT_EQ("sc2", S->getName());
    }
  }
}

TEST(CommandLineTest, DefaultOptions) {
  cl::ResetCommandLineParser();

  StackOption<std::string> Bar("bar", cl::sub(cl::SubCommand::getAll()),
                               cl::DefaultOption);
  StackOption<std::string, cl::alias> Bar_Alias(
      "b", cl::desc("Alias for -bar"), cl::aliasopt(Bar), cl::DefaultOption);

  StackOption<bool> Foo("foo", cl::init(false),
                        cl::sub(cl::SubCommand::getAll()), cl::DefaultOption);
  StackOption<bool, cl::alias> Foo_Alias("f", cl::desc("Alias for -foo"),
                                         cl::aliasopt(Foo), cl::DefaultOption);

  StackSubCommand SC1("sc1", "First Subcommand");
  // Override "-b" and change type in sc1 SubCommand.
  StackOption<bool> SC1_B("b", cl::sub(SC1), cl::init(false));
  StackSubCommand SC2("sc2", "Second subcommand");
  // Override "-foo" and change type in sc2 SubCommand.  Note that this does not
  // affect "-f" alias, which continues to work correctly.
  StackOption<std::string> SC2_Foo("foo", cl::sub(SC2));

  const char *args0[] = {"prog", "-b", "args0 bar string", "-f"};
  EXPECT_TRUE(cl::ParseCommandLineOptions(std::size(args0), args0,
                                          StringRef(), &llvm::nulls()));
  EXPECT_EQ(Bar, "args0 bar string");
  EXPECT_TRUE(Foo);
  EXPECT_FALSE(SC1_B);
  EXPECT_TRUE(SC2_Foo.empty());

  cl::ResetAllOptionOccurrences();

  const char *args1[] = {"prog", "sc1", "-b", "-bar", "args1 bar string", "-f"};
  EXPECT_TRUE(cl::ParseCommandLineOptions(std::size(args1), args1,
                                          StringRef(), &llvm::nulls()));
  EXPECT_EQ(Bar, "args1 bar string");
  EXPECT_TRUE(Foo);
  EXPECT_TRUE(SC1_B);
  EXPECT_TRUE(SC2_Foo.empty());
  for (auto *S : cl::getRegisteredSubcommands()) {
    if (*S) {
      EXPECT_EQ("sc1", S->getName());
    }
  }

  cl::ResetAllOptionOccurrences();

  const char *args2[] = {"prog", "sc2", "-b", "args2 bar string",
                         "-f", "-foo", "foo string"};
  EXPECT_TRUE(cl::ParseCommandLineOptions(std::size(args2), args2,
                                          StringRef(), &llvm::nulls()));
  EXPECT_EQ(Bar, "args2 bar string");
  EXPECT_TRUE(Foo);
  EXPECT_FALSE(SC1_B);
  EXPECT_EQ(SC2_Foo, "foo string");
  for (auto *S : cl::getRegisteredSubcommands()) {
    if (*S) {
      EXPECT_EQ("sc2", S->getName());
    }
  }
  cl::ResetCommandLineParser();
}

TEST(CommandLineTest, ArgumentLimit) {
  std::string args(32 * 4096, 'a');
  EXPECT_FALSE(llvm::sys::commandLineFitsWithinSystemLimits("cl", args.data()));
  std::string args2(256, 'a');
  EXPECT_TRUE(llvm::sys::commandLineFitsWithinSystemLimits("cl", args2.data()));
}

TEST(CommandLineTest, ArgumentLimitWindows) {
  if (!Triple(sys::getProcessTriple()).isOSWindows())
    GTEST_SKIP();
  // We use 32000 as a limit for command line length. Program name ('cl'),
  // separating spaces and termination null character occupy 5 symbols.
  std::string long_arg(32000 - 5, 'b');
  EXPECT_TRUE(
      llvm::sys::commandLineFitsWithinSystemLimits("cl", long_arg.data()));
  long_arg += 'b';
  EXPECT_FALSE(
      llvm::sys::commandLineFitsWithinSystemLimits("cl", long_arg.data()));
}

TEST(CommandLineTest, ResponseFileWindows) {
  if (!Triple(sys::getProcessTriple()).isOSWindows())
    GTEST_SKIP();

  StackOption<std::string, cl::list<std::string>> InputFilenames(
      cl::Positional, cl::desc("<input files>"));
  StackOption<bool> TopLevelOpt("top-level", cl::init(false));

  // Create response file.
  TempFile ResponseFile("resp-", ".txt",
                        "-top-level\npath\\dir\\file1\npath/dir/file2",
                        /*Unique*/ true);

  llvm::SmallString<128> RspOpt;
  RspOpt.append(1, '@');
  RspOpt.append(ResponseFile.path());
  const char *args[] = {"prog", RspOpt.c_str()};
  EXPECT_FALSE(TopLevelOpt);
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args, StringRef(), &llvm::nulls()));
  EXPECT_TRUE(TopLevelOpt);
  EXPECT_EQ(InputFilenames[0], "path\\dir\\file1");
  EXPECT_EQ(InputFilenames[1], "path/dir/file2");
}

TEST(CommandLineTest, ResponseFiles) {
  vfs::InMemoryFileSystem FS;
#ifdef _WIN32
  const char *TestRoot = "C:\\";
#else
  const char *TestRoot = "/";
#endif
  FS.setCurrentWorkingDirectory(TestRoot);

  // Create included response file of first level.
  llvm::StringRef IncludedFileName = "resp1";
  FS.addFile(IncludedFileName, 0,
             llvm::MemoryBuffer::getMemBuffer("-option_1 -option_2\n"
                                              "@incdir/resp2\n"
                                              "-option_3=abcd\n"
                                              "@incdir/resp3\n"
                                              "-option_4=efjk\n"));

  // Directory for included file.
  llvm::StringRef IncDir = "incdir";

  // Create included response file of second level.
  llvm::SmallString<128> IncludedFileName2;
  llvm::sys::path::append(IncludedFileName2, IncDir, "resp2");
  FS.addFile(IncludedFileName2, 0,
             MemoryBuffer::getMemBuffer("-option_21 -option_22\n"
                                        "-option_23=abcd\n"));

  // Create second included response file of second level.
  llvm::SmallString<128> IncludedFileName3;
  llvm::sys::path::append(IncludedFileName3, IncDir, "resp3");
  FS.addFile(IncludedFileName3, 0,
             MemoryBuffer::getMemBuffer("-option_31 -option_32\n"
                                        "-option_33=abcd\n"));

  // Prepare 'file' with reference to response file.
  SmallString<128> IncRef;
  IncRef.append(1, '@');
  IncRef.append(IncludedFileName);
  llvm::SmallVector<const char *, 4> Argv = {"test/test", "-flag_1",
                                             IncRef.c_str(), "-flag_2"};

  // Expand response files.
  llvm::BumpPtrAllocator A;
  llvm::cl::ExpansionContext ECtx(A, llvm::cl::TokenizeGNUCommandLine);
  ECtx.setVFS(&FS).setCurrentDir(TestRoot).setRelativeNames(true);
  ASSERT_FALSE((bool)ECtx.expandResponseFiles(Argv));
  EXPECT_THAT(Argv, testing::Pointwise(
                        StringEquality(),
                        {"test/test", "-flag_1", "-option_1", "-option_2",
                         "-option_21", "-option_22", "-option_23=abcd",
                         "-option_3=abcd", "-option_31", "-option_32",
                         "-option_33=abcd", "-option_4=efjk", "-flag_2"}));
}

TEST(CommandLineTest, RecursiveResponseFiles) {
  vfs::InMemoryFileSystem FS;
#ifdef _WIN32
  const char *TestRoot = "C:\\";
#else
  const char *TestRoot = "/";
#endif
  FS.setCurrentWorkingDirectory(TestRoot);

  StringRef SelfFilePath = "self.rsp";
  std::string SelfFileRef = ("@" + SelfFilePath).str();

  StringRef NestedFilePath = "nested.rsp";
  std::string NestedFileRef = ("@" + NestedFilePath).str();

  StringRef FlagFilePath = "flag.rsp";
  std::string FlagFileRef = ("@" + FlagFilePath).str();

  std::string SelfFileContents;
  raw_string_ostream SelfFile(SelfFileContents);
  SelfFile << "-option_1\n";
  SelfFile << FlagFileRef << "\n";
  SelfFile << NestedFileRef << "\n";
  SelfFile << SelfFileRef << "\n";
  FS.addFile(SelfFilePath, 0, MemoryBuffer::getMemBuffer(SelfFile.str()));

  std::string NestedFileContents;
  raw_string_ostream NestedFile(NestedFileContents);
  NestedFile << "-option_2\n";
  NestedFile << FlagFileRef << "\n";
  NestedFile << SelfFileRef << "\n";
  NestedFile << NestedFileRef << "\n";
  FS.addFile(NestedFilePath, 0, MemoryBuffer::getMemBuffer(NestedFile.str()));

  std::string FlagFileContents;
  raw_string_ostream FlagFile(FlagFileContents);
  FlagFile << "-option_x\n";
  FS.addFile(FlagFilePath, 0, MemoryBuffer::getMemBuffer(FlagFile.str()));

  // Ensure:
  // Recursive expansion terminates
  // Recursive files never expand
  // Non-recursive repeats are allowed
  SmallVector<const char *, 4> Argv = {"test/test", SelfFileRef.c_str(),
                                       "-option_3"};
  BumpPtrAllocator A;
#ifdef _WIN32
  cl::TokenizerCallback Tokenizer = cl::TokenizeWindowsCommandLine;
#else
  cl::TokenizerCallback Tokenizer = cl::TokenizeGNUCommandLine;
#endif
  llvm::cl::ExpansionContext ECtx(A, Tokenizer);
  ECtx.setVFS(&FS).setCurrentDir(TestRoot);
  llvm::Error Err = ECtx.expandResponseFiles(Argv);
  ASSERT_TRUE((bool)Err);
  SmallString<128> FilePath = SelfFilePath;
  std::error_code EC = FS.makeAbsolute(FilePath);
  ASSERT_FALSE((bool)EC);
  std::string ExpectedMessage =
      std::string("recursive expansion of: '") + std::string(FilePath) + "'";
  ASSERT_TRUE(toString(std::move(Err)) == ExpectedMessage);

  EXPECT_THAT(Argv,
              testing::Pointwise(StringEquality(),
                                 {"test/test", "-option_1", "-option_x",
                                  "-option_2", "-option_x", SelfFileRef.c_str(),
                                  NestedFileRef.c_str(), SelfFileRef.c_str(),
                                  "-option_3"}));
}

TEST(CommandLineTest, ResponseFilesAtArguments) {
  vfs::InMemoryFileSystem FS;
#ifdef _WIN32
  const char *TestRoot = "C:\\";
#else
  const char *TestRoot = "/";
#endif
  FS.setCurrentWorkingDirectory(TestRoot);

  StringRef ResponseFilePath = "test.rsp";

  std::string ResponseFileContents;
  raw_string_ostream ResponseFile(ResponseFileContents);
  ResponseFile << "-foo" << "\n";
  ResponseFile << "-bar" << "\n";
  FS.addFile(ResponseFilePath, 0,
             MemoryBuffer::getMemBuffer(ResponseFile.str()));

  // Ensure we expand rsp files after lots of non-rsp arguments starting with @.
  constexpr size_t NON_RSP_AT_ARGS = 64;
  SmallVector<const char *, 4> Argv = {"test/test"};
  Argv.append(NON_RSP_AT_ARGS, "@non_rsp_at_arg");
  std::string ResponseFileRef = ("@" + ResponseFilePath).str();
  Argv.push_back(ResponseFileRef.c_str());

  BumpPtrAllocator A;
  llvm::cl::ExpansionContext ECtx(A, cl::TokenizeGNUCommandLine);
  ECtx.setVFS(&FS).setCurrentDir(TestRoot);
  ASSERT_FALSE((bool)ECtx.expandResponseFiles(Argv));

  // ASSERT instead of EXPECT to prevent potential out-of-bounds access.
  ASSERT_EQ(Argv.size(), 1 + NON_RSP_AT_ARGS + 2);
  size_t i = 0;
  EXPECT_STREQ(Argv[i++], "test/test");
  for (; i < 1 + NON_RSP_AT_ARGS; ++i)
    EXPECT_STREQ(Argv[i], "@non_rsp_at_arg");
  EXPECT_STREQ(Argv[i++], "-foo");
  EXPECT_STREQ(Argv[i++], "-bar");
}

TEST(CommandLineTest, ResponseFileRelativePath) {
  vfs::InMemoryFileSystem FS;
#ifdef _WIN32
  const char *TestRoot = "C:\\";
#else
  const char *TestRoot = "//net";
#endif
  FS.setCurrentWorkingDirectory(TestRoot);

  StringRef OuterFile = "dir/outer.rsp";
  StringRef OuterFileContents = "@inner.rsp";
  FS.addFile(OuterFile, 0, MemoryBuffer::getMemBuffer(OuterFileContents));

  StringRef InnerFile = "dir/inner.rsp";
  StringRef InnerFileContents = "-flag";
  FS.addFile(InnerFile, 0, MemoryBuffer::getMemBuffer(InnerFileContents));

  SmallVector<const char *, 2> Argv = {"test/test", "@dir/outer.rsp"};

  BumpPtrAllocator A;
  llvm::cl::ExpansionContext ECtx(A, cl::TokenizeGNUCommandLine);
  ECtx.setVFS(&FS).setCurrentDir(TestRoot).setRelativeNames(true);
  ASSERT_FALSE((bool)ECtx.expandResponseFiles(Argv));
  EXPECT_THAT(Argv,
              testing::Pointwise(StringEquality(), {"test/test", "-flag"}));
}

TEST(CommandLineTest, ResponseFileEOLs) {
  vfs::InMemoryFileSystem FS;
#ifdef _WIN32
  const char *TestRoot = "C:\\";
#else
  const char *TestRoot = "//net";
#endif
  FS.setCurrentWorkingDirectory(TestRoot);
  FS.addFile("eols.rsp", 0,
             MemoryBuffer::getMemBuffer("-Xclang -Wno-whatever\n input.cpp"));
  SmallVector<const char *, 2> Argv = {"clang", "@eols.rsp"};
  BumpPtrAllocator A;
  llvm::cl::ExpansionContext ECtx(A, cl::TokenizeWindowsCommandLine);
  ECtx.setVFS(&FS).setCurrentDir(TestRoot).setMarkEOLs(true).setRelativeNames(
      true);
  ASSERT_FALSE((bool)ECtx.expandResponseFiles(Argv));
  const char *Expected[] = {"clang", "-Xclang", "-Wno-whatever", nullptr,
                            "input.cpp"};
  ASSERT_EQ(std::size(Expected), Argv.size());
  for (size_t I = 0, E = std::size(Expected); I < E; ++I) {
    if (Expected[I] == nullptr) {
      ASSERT_EQ(Argv[I], nullptr);
    } else {
      ASSERT_STREQ(Expected[I], Argv[I]);
    }
  }
}

TEST(CommandLineTest, BadResponseFile) {
  BumpPtrAllocator A;
  StringSaver Saver(A);
  TempDir ADir("dir", /*Unique*/ true);
  SmallString<128> AFilePath = ADir.path();
  llvm::sys::path::append(AFilePath, "file.rsp");
  std::string AFileExp = std::string("@") + std::string(AFilePath.str());
  SmallVector<const char *, 2> Argv = {"clang", AFileExp.c_str()};

  bool Res = cl::ExpandResponseFiles(Saver, cl::TokenizeGNUCommandLine, Argv);
  ASSERT_TRUE(Res);
  ASSERT_EQ(2U, Argv.size());
  ASSERT_STREQ(Argv[0], "clang");
  ASSERT_STREQ(Argv[1], AFileExp.c_str());

#if !defined(_AIX) && !defined(__MVS__)
  std::string ADirExp = std::string("@") + std::string(ADir.path());
  Argv = {"clang", ADirExp.c_str()};
  Res = cl::ExpandResponseFiles(Saver, cl::TokenizeGNUCommandLine, Argv);
  ASSERT_FALSE(Res);
  ASSERT_EQ(2U, Argv.size());
  ASSERT_STREQ(Argv[0], "clang");
  ASSERT_STREQ(Argv[1], ADirExp.c_str());
#endif
}

TEST(CommandLineTest, SetDefaultValue) {
  cl::ResetCommandLineParser();

  StackOption<std::string> Opt1("opt1", cl::init("true"));
  StackOption<bool> Opt2("opt2", cl::init(true));
  cl::alias Alias("alias", llvm::cl::aliasopt(Opt2));
  StackOption<int> Opt3("opt3", cl::init(3));

  llvm::SmallVector<int, 3> IntVals = {1, 2, 3};
  llvm::SmallVector<std::string, 3> StrVals = {"foo", "bar", "baz"};

  StackOption<int, cl::list<int>> List1(
      "list1", cl::list_init<int>(llvm::ArrayRef<int>(IntVals)),
      cl::CommaSeparated);
  StackOption<std::string, cl::list<std::string>> List2(
      "list2", cl::list_init<std::string>(llvm::ArrayRef<std::string>(StrVals)),
      cl::CommaSeparated);
  cl::alias ListAlias("list-alias", llvm::cl::aliasopt(List2));

  const char *args[] = {"prog",   "-opt1=false", "-list1", "4",
                        "-list1", "5,6",         "-opt2",  "-opt3"};

  EXPECT_TRUE(
      cl::ParseCommandLineOptions(7, args, StringRef(), &llvm::nulls()));

  EXPECT_EQ(Opt1, "false");
  EXPECT_TRUE(Opt2);
  EXPECT_EQ(Opt3, 3);

  for (size_t I = 0, E = IntVals.size(); I < E; ++I) {
    EXPECT_EQ(IntVals[I] + 3, List1[I]);
    EXPECT_EQ(StrVals[I], List2[I]);
  }

  Opt2 = false;
  Opt3 = 1;

  cl::ResetAllOptionOccurrences();

  for (auto &OM : cl::getRegisteredOptions(cl::SubCommand::getTopLevel())) {
    cl::Option *O = OM.second;
    if (O->ArgStr == "opt2") {
      continue;
    }
    O->setDefault();
  }

  EXPECT_EQ(Opt1, "true");
  EXPECT_TRUE(Opt2);
  EXPECT_EQ(Opt3, 3);
  for (size_t I = 0, E = IntVals.size(); I < E; ++I) {
    EXPECT_EQ(IntVals[I], List1[I]);
    EXPECT_EQ(StrVals[I], List2[I]);
  }

  Alias.removeArgument();
  ListAlias.removeArgument();
}

TEST(CommandLineTest, ReadConfigFile) {
  llvm::SmallVector<const char *, 1> Argv;

  TempDir TestDir("unittest", /*Unique*/ true);
  TempDir TestSubDir(TestDir.path("subdir"), /*Unique*/ false);

  llvm::SmallString<128> TestCfg = TestDir.path("foo");
  TempFile ConfigFile(TestCfg, "",
                      "# Comment\n"
                      "-option_1\n"
                      "-option_2=<CFGDIR>/dir1\n"
                      "-option_3=<CFGDIR>\n"
                      "-option_4 <CFGDIR>\n"
                      "-option_5=<CFG\\\n"
                      "DIR>\n"
                      "-option_6=<CFGDIR>/dir1,<CFGDIR>/dir2\n"
                      "@subconfig\n"
                      "-option_11=abcd\n"
                      "-option_12=\\\n"
                      "cdef\n");

  llvm::SmallString<128> TestCfg2 = TestDir.path("subconfig");
  TempFile ConfigFile2(TestCfg2, "",
                       "-option_7\n"
                       "-option_8=<CFGDIR>/dir2\n"
                       "@subdir/subfoo\n"
                       "\n"
                       "   # comment\n");

  llvm::SmallString<128> TestCfg3 = TestSubDir.path("subfoo");
  TempFile ConfigFile3(TestCfg3, "",
                       "-option_9=<CFGDIR>/dir3\n"
                       "@<CFGDIR>/subfoo2\n");

  llvm::SmallString<128> TestCfg4 = TestSubDir.path("subfoo2");
  TempFile ConfigFile4(TestCfg4, "", "-option_10\n");

  // Make sure the current directory is not the directory where config files
  // resides. In this case the code that expands response files will not find
  // 'subconfig' unless it resolves nested inclusions relative to the including
  // file.
  llvm::SmallString<128> CurrDir;
  std::error_code EC = llvm::sys::fs::current_path(CurrDir);
  EXPECT_TRUE(!EC);
  EXPECT_NE(CurrDir.str(), TestDir.path());

  llvm::BumpPtrAllocator A;
  llvm::cl::ExpansionContext ECtx(A, cl::tokenizeConfigFile);
  llvm::Error Result = ECtx.readConfigFile(ConfigFile.path(), Argv);

  EXPECT_FALSE((bool)Result);
  EXPECT_EQ(Argv.size(), 13U);
  EXPECT_STREQ(Argv[0], "-option_1");
  EXPECT_STREQ(Argv[1],
               ("-option_2=" + TestDir.path() + "/dir1").str().c_str());
  EXPECT_STREQ(Argv[2], ("-option_3=" + TestDir.path()).str().c_str());
  EXPECT_STREQ(Argv[3], "-option_4");
  EXPECT_STREQ(Argv[4], TestDir.path().str().c_str());
  EXPECT_STREQ(Argv[5], ("-option_5=" + TestDir.path()).str().c_str());
  EXPECT_STREQ(Argv[6], ("-option_6=" + TestDir.path() + "/dir1," +
                         TestDir.path() + "/dir2")
                            .str()
                            .c_str());
  EXPECT_STREQ(Argv[7], "-option_7");
  EXPECT_STREQ(Argv[8],
               ("-option_8=" + TestDir.path() + "/dir2").str().c_str());
  EXPECT_STREQ(Argv[9],
               ("-option_9=" + TestSubDir.path() + "/dir3").str().c_str());
  EXPECT_STREQ(Argv[10], "-option_10");
  EXPECT_STREQ(Argv[11], "-option_11=abcd");
  EXPECT_STREQ(Argv[12], "-option_12=cdef");
}

TEST(CommandLineTest, PositionalEatArgsError) {
  cl::ResetCommandLineParser();

  StackOption<std::string, cl::list<std::string>> PosEatArgs(
      "positional-eat-args", cl::Positional, cl::desc("<arguments>..."),
      cl::PositionalEatsArgs);
  StackOption<std::string, cl::list<std::string>> PosEatArgs2(
      "positional-eat-args2", cl::Positional, cl::desc("Some strings"),
      cl::PositionalEatsArgs);

  const char *args[] = {"prog", "-positional-eat-args=XXXX"};
  const char *args2[] = {"prog", "-positional-eat-args=XXXX", "-foo"};
  const char *args3[] = {"prog", "-positional-eat-args", "-foo"};
  const char *args4[] = {"prog", "-positional-eat-args",
                         "-foo", "-positional-eat-args2",
                         "-bar", "foo"};

  std::string Errs;
  raw_string_ostream OS(Errs);
  EXPECT_FALSE(cl::ParseCommandLineOptions(2, args, StringRef(), &OS)); OS.flush();
  EXPECT_FALSE(Errs.empty()); Errs.clear();
  EXPECT_FALSE(cl::ParseCommandLineOptions(3, args2, StringRef(), &OS)); OS.flush();
  EXPECT_FALSE(Errs.empty()); Errs.clear();
  EXPECT_TRUE(cl::ParseCommandLineOptions(3, args3, StringRef(), &OS)); OS.flush();
  EXPECT_TRUE(Errs.empty()); Errs.clear();

  cl::ResetAllOptionOccurrences();
  EXPECT_TRUE(cl::ParseCommandLineOptions(6, args4, StringRef(), &OS)); OS.flush();
  EXPECT_EQ(PosEatArgs.size(), 1u);
  EXPECT_EQ(PosEatArgs2.size(), 2u);
  EXPECT_TRUE(Errs.empty());
}

#ifdef _WIN32
void checkSeparators(StringRef Path) {
  char UndesiredSeparator = sys::path::get_separator()[0] == '/' ? '\\' : '/';
  ASSERT_EQ(Path.find(UndesiredSeparator), StringRef::npos);
}

TEST(CommandLineTest, GetCommandLineArguments) {
  int argc = __argc;
  char **argv = __argv;

  // GetCommandLineArguments is called in InitLLVM.
  llvm::InitLLVM X(argc, argv);

  EXPECT_EQ(llvm::sys::path::is_absolute(argv[0]),
            llvm::sys::path::is_absolute(__argv[0]));
  checkSeparators(argv[0]);

  EXPECT_TRUE(
      llvm::sys::path::filename(argv[0]).equals_insensitive("supporttests.exe"))
      << "Filename of test executable is "
      << llvm::sys::path::filename(argv[0]);
}
#endif

class OutputRedirector {
public:
  OutputRedirector(int RedirectFD)
      : RedirectFD(RedirectFD), OldFD(dup(RedirectFD)) {
    if (OldFD == -1 ||
        sys::fs::createTemporaryFile("unittest-redirect", "", NewFD,
                                     FilePath) ||
        dup2(NewFD, RedirectFD) == -1)
      Valid = false;
  }

  ~OutputRedirector() {
    dup2(OldFD, RedirectFD);
    close(OldFD);
    close(NewFD);
  }

  SmallVector<char, 128> FilePath;
  bool Valid = true;

private:
  int RedirectFD;
  int OldFD;
  int NewFD;
};

struct AutoDeleteFile {
  SmallVector<char, 128> FilePath;
  ~AutoDeleteFile() {
    if (!FilePath.empty())
      sys::fs::remove(std::string(FilePath.data(), FilePath.size()));
  }
};

static std::string interceptStdout(std::function<void()> F) {
  outs().flush(); // flush any output from previous tests
  AutoDeleteFile File;
  {
    OutputRedirector Stdout(fileno(stdout));
    if (!Stdout.Valid)
      return "";
    File.FilePath = Stdout.FilePath;
    F();
    outs().flush();
  }
  auto Buffer = MemoryBuffer::getFile(File.FilePath);
  if (!Buffer)
    return "";
  return Buffer->get()->getBuffer().str();
}

template <void (*Func)(const cl::Option &)>
class PrintOptionTestBase : public ::testing::Test {
public:
  // Return std::string because the output of a failing EXPECT check is
  // unreadable for StringRef. It also avoids any lifetime issues.
  template <typename... Ts> std::string runTest(Ts... OptionAttributes) {
    StackOption<OptionValue> TestOption(Opt, cl::desc(HelpText),
                                        OptionAttributes...);
    return interceptStdout([&]() { Func(TestOption); });
  }

  enum class OptionValue { Val };
  const StringRef Opt = "some-option";
  const StringRef HelpText = "some help";
};

  // This is a workaround for cl::Option sub-classes having their
  // printOptionInfo functions private.
void printOptionInfo(const cl::Option &O) {
  O.printOptionInfo(/*GlobalWidth=*/26);
}

using PrintOptionInfoTest = PrintOptionTestBase<printOptionInfo>;

TEST_F(PrintOptionInfoTest, PrintOptionInfoValueOptionalWithoutSentinel) {
  std::string Output =
      runTest(cl::ValueOptional,
              cl::values(clEnumValN(OptionValue::Val, "v1", "desc1")));

  // clang-format off
  EXPECT_EQ(Output, ("  --" + Opt + "=<value> - " + HelpText + "\n"
                     "    =v1                 -   desc1\n")
                        .str());
  // clang-format on
}

TEST_F(PrintOptionInfoTest, PrintOptionInfoValueOptionalWithSentinel) {
  std::string Output = runTest(
      cl::ValueOptional, cl::values(clEnumValN(OptionValue::Val, "v1", "desc1"),
                                    clEnumValN(OptionValue::Val, "", "")));

  // clang-format off
  EXPECT_EQ(Output,
            ("  --" + Opt + "         - " + HelpText + "\n"
             "  --" + Opt + "=<value> - " + HelpText + "\n"
             "    =v1                 -   desc1\n")
                .str());
  // clang-format on
}

TEST_F(PrintOptionInfoTest, PrintOptionInfoValueOptionalWithSentinelWithHelp) {
  std::string Output = runTest(
      cl::ValueOptional, cl::values(clEnumValN(OptionValue::Val, "v1", "desc1"),
                                    clEnumValN(OptionValue::Val, "", "desc2")));

  // clang-format off
  EXPECT_EQ(Output, ("  --" + Opt + "         - " + HelpText + "\n"
                     "  --" + Opt + "=<value> - " + HelpText + "\n"
                     "    =v1                 -   desc1\n"
                     "    =<empty>            -   desc2\n")
                        .str());
  // clang-format on
}

TEST_F(PrintOptionInfoTest, PrintOptionInfoValueRequiredWithEmptyValueName) {
  std::string Output = runTest(
      cl::ValueRequired, cl::values(clEnumValN(OptionValue::Val, "v1", "desc1"),
                                    clEnumValN(OptionValue::Val, "", "")));

  // clang-format off
  EXPECT_EQ(Output, ("  --" + Opt + "=<value> - " + HelpText + "\n"
                     "    =v1                 -   desc1\n"
                     "    =<empty>\n")
                        .str());
  // clang-format on
}

TEST_F(PrintOptionInfoTest, PrintOptionInfoEmptyValueDescription) {
  std::string Output = runTest(
      cl::ValueRequired, cl::values(clEnumValN(OptionValue::Val, "v1", "")));

  // clang-format off
  EXPECT_EQ(Output,
            ("  --" + Opt + "=<value> - " + HelpText + "\n"
             "    =v1\n").str());
  // clang-format on
}

TEST_F(PrintOptionInfoTest, PrintOptionInfoMultilineValueDescription) {
  std::string Output =
      runTest(cl::ValueRequired,
              cl::values(clEnumValN(OptionValue::Val, "v1",
                                    "This is the first enum value\n"
                                    "which has a really long description\n"
                                    "thus it is multi-line."),
                         clEnumValN(OptionValue::Val, "",
                                    "This is an unnamed enum value\n"
                                    "Should be indented as well")));

  // clang-format off
  EXPECT_EQ(Output,
            ("  --" + Opt + "=<value> - " + HelpText + "\n"
             "    =v1                 -   This is the first enum value\n"
             "                            which has a really long description\n"
             "                            thus it is multi-line.\n"
             "    =<empty>            -   This is an unnamed enum value\n"
             "                            Should be indented as well\n").str());
  // clang-format on
}

void printOptionValue(const cl::Option &O) {
  O.printOptionValue(/*GlobalWidth=*/12, /*Force=*/true);
}

using PrintOptionValueTest = PrintOptionTestBase<printOptionValue>;

TEST_F(PrintOptionValueTest, PrintOptionDefaultValue) {
  std::string Output =
      runTest(cl::init(OptionValue::Val),
              cl::values(clEnumValN(OptionValue::Val, "v1", "desc1")));

  EXPECT_EQ(Output, ("    --" + Opt + " = v1       (default: v1)\n").str());
}

TEST_F(PrintOptionValueTest, PrintOptionNoDefaultValue) {
  std::string Output =
      runTest(cl::values(clEnumValN(OptionValue::Val, "v1", "desc1")));

  // Note: the option still has a (zero-initialized) value, but the default
  // is invalid and doesn't match any value.
  EXPECT_EQ(Output, ("    --" + Opt + " = v1       (default: )\n").str());
}

TEST_F(PrintOptionValueTest, PrintOptionUnknownValue) {
  std::string Output = runTest(cl::init(OptionValue::Val));

  EXPECT_EQ(Output, ("    --" + Opt + " = *unknown option value*\n").str());
}

class GetOptionWidthTest : public ::testing::Test {
public:
  enum class OptionValue { Val };

  template <typename... Ts>
  size_t runTest(StringRef ArgName, Ts... OptionAttributes) {
    StackOption<OptionValue> TestOption(ArgName, cl::desc("some help"),
                                        OptionAttributes...);
    return getOptionWidth(TestOption);
  }

private:
  // This is a workaround for cl::Option sub-classes having their
  // printOptionInfo
  // functions private.
  size_t getOptionWidth(const cl::Option &O) { return O.getOptionWidth(); }
};

TEST_F(GetOptionWidthTest, GetOptionWidthArgNameLonger) {
  StringRef ArgName("a-long-argument-name");
  size_t ExpectedStrSize = ("  --" + ArgName + "=<value> - ").str().size();
  EXPECT_EQ(
      runTest(ArgName, cl::values(clEnumValN(OptionValue::Val, "v", "help"))),
      ExpectedStrSize);
}

TEST_F(GetOptionWidthTest, GetOptionWidthFirstOptionNameLonger) {
  StringRef OptName("a-long-option-name");
  size_t ExpectedStrSize = ("    =" + OptName + " - ").str().size();
  EXPECT_EQ(
      runTest("a", cl::values(clEnumValN(OptionValue::Val, OptName, "help"),
                              clEnumValN(OptionValue::Val, "b", "help"))),
      ExpectedStrSize);
}

TEST_F(GetOptionWidthTest, GetOptionWidthSecondOptionNameLonger) {
  StringRef OptName("a-long-option-name");
  size_t ExpectedStrSize = ("    =" + OptName + " - ").str().size();
  EXPECT_EQ(
      runTest("a", cl::values(clEnumValN(OptionValue::Val, "b", "help"),
                              clEnumValN(OptionValue::Val, OptName, "help"))),
      ExpectedStrSize);
}

TEST_F(GetOptionWidthTest, GetOptionWidthEmptyOptionNameLonger) {
  size_t ExpectedStrSize = StringRef("    =<empty> - ").size();
  // The length of a=<value> (including indentation) is actually the same as the
  // =<empty> string, so it is impossible to distinguish via testing the case
  // where the empty string is picked from where the option name is picked.
  EXPECT_EQ(runTest("a", cl::values(clEnumValN(OptionValue::Val, "b", "help"),
                                    clEnumValN(OptionValue::Val, "", "help"))),
            ExpectedStrSize);
}

TEST_F(GetOptionWidthTest,
       GetOptionWidthValueOptionalEmptyOptionWithNoDescription) {
  StringRef ArgName("a");
  // The length of a=<value> (including indentation) is actually the same as the
  // =<empty> string, so it is impossible to distinguish via testing the case
  // where the empty string is ignored from where it is not ignored.
  // The dash will not actually be printed, but the space it would take up is
  // included to ensure a consistent column width.
  size_t ExpectedStrSize = ("  -" + ArgName + "=<value> - ").str().size();
  EXPECT_EQ(runTest(ArgName, cl::ValueOptional,
                    cl::values(clEnumValN(OptionValue::Val, "value", "help"),
                               clEnumValN(OptionValue::Val, "", ""))),
            ExpectedStrSize);
}

TEST_F(GetOptionWidthTest,
       GetOptionWidthValueRequiredEmptyOptionWithNoDescription) {
  // The length of a=<value> (including indentation) is actually the same as the
  // =<empty> string, so it is impossible to distinguish via testing the case
  // where the empty string is picked from where the option name is picked
  size_t ExpectedStrSize = StringRef("    =<empty> - ").size();
  EXPECT_EQ(runTest("a", cl::ValueRequired,
                    cl::values(clEnumValN(OptionValue::Val, "value", "help"),
                               clEnumValN(OptionValue::Val, "", ""))),
            ExpectedStrSize);
}

TEST(CommandLineTest, PrefixOptions) {
  cl::ResetCommandLineParser();

  StackOption<std::string, cl::list<std::string>> IncludeDirs(
      "I", cl::Prefix, cl::desc("Declare an include directory"));

  // Test non-prefixed variant works with cl::Prefix options.
  EXPECT_TRUE(IncludeDirs.empty());
  const char *args[] = {"prog", "-I=/usr/include"};
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args, StringRef(), &llvm::nulls()));
  EXPECT_EQ(IncludeDirs.size(), 1u);
  EXPECT_EQ(IncludeDirs.front().compare("/usr/include"), 0);

  IncludeDirs.erase(IncludeDirs.begin());
  cl::ResetAllOptionOccurrences();

  // Test non-prefixed variant works with cl::Prefix options when value is
  // passed in following argument.
  EXPECT_TRUE(IncludeDirs.empty());
  const char *args2[] = {"prog", "-I", "/usr/include"};
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(3, args2, StringRef(), &llvm::nulls()));
  EXPECT_EQ(IncludeDirs.size(), 1u);
  EXPECT_EQ(IncludeDirs.front().compare("/usr/include"), 0);

  IncludeDirs.erase(IncludeDirs.begin());
  cl::ResetAllOptionOccurrences();

  // Test prefixed variant works with cl::Prefix options.
  EXPECT_TRUE(IncludeDirs.empty());
  const char *args3[] = {"prog", "-I/usr/include"};
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args3, StringRef(), &llvm::nulls()));
  EXPECT_EQ(IncludeDirs.size(), 1u);
  EXPECT_EQ(IncludeDirs.front().compare("/usr/include"), 0);

  StackOption<std::string, cl::list<std::string>> MacroDefs(
      "D", cl::AlwaysPrefix, cl::desc("Define a macro"),
      cl::value_desc("MACRO[=VALUE]"));

  cl::ResetAllOptionOccurrences();

  // Test non-prefixed variant does not work with cl::AlwaysPrefix options:
  // equal sign is part of the value.
  EXPECT_TRUE(MacroDefs.empty());
  const char *args4[] = {"prog", "-D=HAVE_FOO"};
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args4, StringRef(), &llvm::nulls()));
  EXPECT_EQ(MacroDefs.size(), 1u);
  EXPECT_EQ(MacroDefs.front().compare("=HAVE_FOO"), 0);

  MacroDefs.erase(MacroDefs.begin());
  cl::ResetAllOptionOccurrences();

  // Test non-prefixed variant does not allow value to be passed in following
  // argument with cl::AlwaysPrefix options.
  EXPECT_TRUE(MacroDefs.empty());
  const char *args5[] = {"prog", "-D", "HAVE_FOO"};
  EXPECT_FALSE(
      cl::ParseCommandLineOptions(3, args5, StringRef(), &llvm::nulls()));
  EXPECT_TRUE(MacroDefs.empty());

  cl::ResetAllOptionOccurrences();

  // Test prefixed variant works with cl::AlwaysPrefix options.
  EXPECT_TRUE(MacroDefs.empty());
  const char *args6[] = {"prog", "-DHAVE_FOO"};
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args6, StringRef(), &llvm::nulls()));
  EXPECT_EQ(MacroDefs.size(), 1u);
  EXPECT_EQ(MacroDefs.front().compare("HAVE_FOO"), 0);
}

TEST(CommandLineTest, GroupingWithValue) {
  cl::ResetCommandLineParser();

  StackOption<bool> OptF("f", cl::Grouping, cl::desc("Some flag"));
  StackOption<bool> OptB("b", cl::Grouping, cl::desc("Another flag"));
  StackOption<bool> OptD("d", cl::Grouping, cl::ValueDisallowed,
                         cl::desc("ValueDisallowed option"));
  StackOption<std::string> OptV("v", cl::Grouping,
                                cl::desc("ValueRequired option"));
  StackOption<std::string> OptO("o", cl::Grouping, cl::ValueOptional,
                                cl::desc("ValueOptional option"));

  // Should be possible to use an option which requires a value
  // at the end of a group.
  const char *args1[] = {"prog", "-fv", "val1"};
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(3, args1, StringRef(), &llvm::nulls()));
  EXPECT_TRUE(OptF);
  EXPECT_STREQ("val1", OptV.c_str());
  OptV.clear();
  cl::ResetAllOptionOccurrences();

  // Should not crash if it is accidentally used elsewhere in the group.
  const char *args2[] = {"prog", "-vf", "val2"};
  EXPECT_FALSE(
      cl::ParseCommandLineOptions(3, args2, StringRef(), &llvm::nulls()));
  OptV.clear();
  cl::ResetAllOptionOccurrences();

  // Should allow the "opt=value" form at the end of the group
  const char *args3[] = {"prog", "-fv=val3"};
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args3, StringRef(), &llvm::nulls()));
  EXPECT_TRUE(OptF);
  EXPECT_STREQ("val3", OptV.c_str());
  OptV.clear();
  cl::ResetAllOptionOccurrences();

  // Should allow assigning a value for a ValueOptional option
  // at the end of the group
  const char *args4[] = {"prog", "-fo=val4"};
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args4, StringRef(), &llvm::nulls()));
  EXPECT_TRUE(OptF);
  EXPECT_STREQ("val4", OptO.c_str());
  OptO.clear();
  cl::ResetAllOptionOccurrences();

  // Should assign an empty value if a ValueOptional option is used elsewhere
  // in the group.
  const char *args5[] = {"prog", "-fob"};
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args5, StringRef(), &llvm::nulls()));
  EXPECT_TRUE(OptF);
  EXPECT_EQ(1, OptO.getNumOccurrences());
  EXPECT_EQ(1, OptB.getNumOccurrences());
  EXPECT_TRUE(OptO.empty());
  cl::ResetAllOptionOccurrences();

  // Should not allow an assignment for a ValueDisallowed option.
  const char *args6[] = {"prog", "-fd=false"};
  EXPECT_FALSE(
      cl::ParseCommandLineOptions(2, args6, StringRef(), &llvm::nulls()));
}

TEST(CommandLineTest, GroupingAndPrefix) {
  cl::ResetCommandLineParser();

  StackOption<bool> OptF("f", cl::Grouping, cl::desc("Some flag"));
  StackOption<bool> OptB("b", cl::Grouping, cl::desc("Another flag"));
  StackOption<std::string> OptP("p", cl::Prefix, cl::Grouping,
                                cl::desc("Prefix and Grouping"));
  StackOption<std::string> OptA("a", cl::AlwaysPrefix, cl::Grouping,
                                cl::desc("AlwaysPrefix and Grouping"));

  // Should be possible to use a cl::Prefix option without grouping.
  const char *args1[] = {"prog", "-pval1"};
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args1, StringRef(), &llvm::nulls()));
  EXPECT_STREQ("val1", OptP.c_str());
  OptP.clear();
  cl::ResetAllOptionOccurrences();

  // Should be possible to pass a value in a separate argument.
  const char *args2[] = {"prog", "-p", "val2"};
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(3, args2, StringRef(), &llvm::nulls()));
  EXPECT_STREQ("val2", OptP.c_str());
  OptP.clear();
  cl::ResetAllOptionOccurrences();

  // The "-opt=value" form should work, too.
  const char *args3[] = {"prog", "-p=val3"};
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args3, StringRef(), &llvm::nulls()));
  EXPECT_STREQ("val3", OptP.c_str());
  OptP.clear();
  cl::ResetAllOptionOccurrences();

  // All three previous cases should work the same way if an option with both
  // cl::Prefix and cl::Grouping modifiers is used at the end of a group.
  const char *args4[] = {"prog", "-fpval4"};
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args4, StringRef(), &llvm::nulls()));
  EXPECT_TRUE(OptF);
  EXPECT_STREQ("val4", OptP.c_str());
  OptP.clear();
  cl::ResetAllOptionOccurrences();

  const char *args5[] = {"prog", "-fp", "val5"};
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(3, args5, StringRef(), &llvm::nulls()));
  EXPECT_TRUE(OptF);
  EXPECT_STREQ("val5", OptP.c_str());
  OptP.clear();
  cl::ResetAllOptionOccurrences();

  const char *args6[] = {"prog", "-fp=val6"};
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args6, StringRef(), &llvm::nulls()));
  EXPECT_TRUE(OptF);
  EXPECT_STREQ("val6", OptP.c_str());
  OptP.clear();
  cl::ResetAllOptionOccurrences();

  // Should assign a value even if the part after a cl::Prefix option is equal
  // to the name of another option.
  const char *args7[] = {"prog", "-fpb"};
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args7, StringRef(), &llvm::nulls()));
  EXPECT_TRUE(OptF);
  EXPECT_STREQ("b", OptP.c_str());
  EXPECT_FALSE(OptB);
  OptP.clear();
  cl::ResetAllOptionOccurrences();

  // Should be possible to use a cl::AlwaysPrefix option without grouping.
  const char *args8[] = {"prog", "-aval8"};
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args8, StringRef(), &llvm::nulls()));
  EXPECT_STREQ("val8", OptA.c_str());
  OptA.clear();
  cl::ResetAllOptionOccurrences();

  // Should not be possible to pass a value in a separate argument.
  const char *args9[] = {"prog", "-a", "val9"};
  EXPECT_FALSE(
      cl::ParseCommandLineOptions(3, args9, StringRef(), &llvm::nulls()));
  cl::ResetAllOptionOccurrences();

  // With the "-opt=value" form, the "=" symbol should be preserved.
  const char *args10[] = {"prog", "-a=val10"};
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args10, StringRef(), &llvm::nulls()));
  EXPECT_STREQ("=val10", OptA.c_str());
  OptA.clear();
  cl::ResetAllOptionOccurrences();

  // All three previous cases should work the same way if an option with both
  // cl::AlwaysPrefix and cl::Grouping modifiers is used at the end of a group.
  const char *args11[] = {"prog", "-faval11"};
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args11, StringRef(), &llvm::nulls()));
  EXPECT_TRUE(OptF);
  EXPECT_STREQ("val11", OptA.c_str());
  OptA.clear();
  cl::ResetAllOptionOccurrences();

  const char *args12[] = {"prog", "-fa", "val12"};
  EXPECT_FALSE(
      cl::ParseCommandLineOptions(3, args12, StringRef(), &llvm::nulls()));
  cl::ResetAllOptionOccurrences();

  const char *args13[] = {"prog", "-fa=val13"};
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args13, StringRef(), &llvm::nulls()));
  EXPECT_TRUE(OptF);
  EXPECT_STREQ("=val13", OptA.c_str());
  OptA.clear();
  cl::ResetAllOptionOccurrences();

  // Should assign a value even if the part after a cl::AlwaysPrefix option
  // is equal to the name of another option.
  const char *args14[] = {"prog", "-fab"};
  EXPECT_TRUE(
      cl::ParseCommandLineOptions(2, args14, StringRef(), &llvm::nulls()));
  EXPECT_TRUE(OptF);
  EXPECT_STREQ("b", OptA.c_str());
  EXPECT_FALSE(OptB);
  OptA.clear();
  cl::ResetAllOptionOccurrences();
}

TEST(CommandLineTest, LongOptions) {
  cl::ResetCommandLineParser();

  StackOption<bool> OptA("a", cl::desc("Some flag"));
  StackOption<bool> OptBLong("long-flag", cl::desc("Some long flag"));
  StackOption<bool, cl::alias> OptB("b", cl::desc("Alias to --long-flag"),
                                    cl::aliasopt(OptBLong));
  StackOption<std::string> OptAB("ab", cl::desc("Another long option"));

  std::string Errs;
  raw_string_ostream OS(Errs);

  const char *args1[] = {"prog", "-a", "-ab", "val1"};
  const char *args2[] = {"prog", "-a", "--ab", "val1"};
  const char *args3[] = {"prog", "-ab", "--ab", "val1"};

  //
  // The following tests treat `-` and `--` the same, and always match the
  // longest string.
  //

  EXPECT_TRUE(
      cl::ParseCommandLineOptions(4, args1, StringRef(), &OS)); OS.flush();
  EXPECT_TRUE(OptA);
  EXPECT_FALSE(OptBLong);
  EXPECT_STREQ("val1", OptAB.c_str());
  EXPECT_TRUE(Errs.empty()); Errs.clear();
  cl::ResetAllOptionOccurrences();

  EXPECT_TRUE(
      cl::ParseCommandLineOptions(4, args2, StringRef(), &OS)); OS.flush();
  EXPECT_TRUE(OptA);
  EXPECT_FALSE(OptBLong);
  EXPECT_STREQ("val1", OptAB.c_str());
  EXPECT_TRUE(Errs.empty()); Errs.clear();
  cl::ResetAllOptionOccurrences();

  // Fails because `-ab` and `--ab` are treated the same and appear more than
  // once.  Also, `val1` is unexpected.
  EXPECT_FALSE(
      cl::ParseCommandLineOptions(4, args3, StringRef(), &OS)); OS.flush();
  outs()<< Errs << "\n";
  EXPECT_FALSE(Errs.empty()); Errs.clear();
  cl::ResetAllOptionOccurrences();

  //
  // The following tests treat `-` and `--` differently, with `-` for short, and
  // `--` for long options.
  //

  // Fails because `-ab` is treated as `-a -b`, so `-a` is seen twice, and
  // `val1` is unexpected.
  EXPECT_FALSE(cl::ParseCommandLineOptions(4, args1, StringRef(),
                                           &OS, nullptr, true)); OS.flush();
  EXPECT_FALSE(Errs.empty()); Errs.clear();
  cl::ResetAllOptionOccurrences();

  // Works because `-a` is treated differently than `--ab`.
  EXPECT_TRUE(cl::ParseCommandLineOptions(4, args2, StringRef(),
                                           &OS, nullptr, true)); OS.flush();
  EXPECT_TRUE(Errs.empty()); Errs.clear();
  cl::ResetAllOptionOccurrences();

  // Works because `-ab` is treated as `-a -b`, and `--ab` is a long option.
  EXPECT_TRUE(cl::ParseCommandLineOptions(4, args3, StringRef(),
                                           &OS, nullptr, true));
  EXPECT_TRUE(OptA);
  EXPECT_TRUE(OptBLong);
  EXPECT_STREQ("val1", OptAB.c_str());
  OS.flush();
  EXPECT_TRUE(Errs.empty()); Errs.clear();
  cl::ResetAllOptionOccurrences();
}

TEST(CommandLineTest, OptionErrorMessage) {
  // When there is an error, we expect some error message like:
  //   prog: for the -a option: [...]
  //
  // Test whether the "for the -a option"-part is correctly formatted.
  cl::ResetCommandLineParser();

  StackOption<bool> OptA("a", cl::desc("Some option"));
  StackOption<bool> OptLong("long", cl::desc("Some long option"));

  std::string Errs;
  raw_string_ostream OS(Errs);

  OptA.error("custom error", OS);
  OS.flush();
  EXPECT_NE(Errs.find("for the -a option:"), std::string::npos);
  Errs.clear();

  OptLong.error("custom error", OS);
  OS.flush();
  EXPECT_NE(Errs.find("for the --long option:"), std::string::npos);
  Errs.clear();

  cl::ResetAllOptionOccurrences();
}

TEST(CommandLineTest, OptionErrorMessageSuggest) {
  // When there is an error, and the edit-distance is not very large,
  // we expect some error message like:
  //   prog: did you mean '--option'?
  //
  // Test whether this message is well-formatted.
  cl::ResetCommandLineParser();

  StackOption<bool> OptLong("aluminium", cl::desc("Some long option"));

  const char *args[] = {"prog", "--aluminum"};

  std::string Errs;
  raw_string_ostream OS(Errs);

  EXPECT_FALSE(cl::ParseCommandLineOptions(2, args, StringRef(), &OS));
  OS.flush();
  EXPECT_NE(Errs.find("prog: Did you mean '--aluminium'?\n"),
            std::string::npos);
  Errs.clear();

  cl::ResetAllOptionOccurrences();
}

TEST(CommandLineTest, OptionErrorMessageSuggestNoHidden) {
  // We expect that 'really hidden' option do not show up in option
  // suggestions.
  cl::ResetCommandLineParser();

  StackOption<bool> OptLong("aluminium", cl::desc("Some long option"));
  StackOption<bool> OptLong2("aluminum", cl::desc("Bad option"),
                             cl::ReallyHidden);

  const char *args[] = {"prog", "--alumnum"};

  std::string Errs;
  raw_string_ostream OS(Errs);

  EXPECT_FALSE(cl::ParseCommandLineOptions(2, args, StringRef(), &OS));
  OS.flush();
  EXPECT_NE(Errs.find("prog: Did you mean '--aluminium'?\n"),
            std::string::npos);
  Errs.clear();

  cl::ResetAllOptionOccurrences();
}

TEST(CommandLineTest, Callback) {
  cl::ResetCommandLineParser();

  StackOption<bool> OptA("a", cl::desc("option a"));
  StackOption<bool> OptB(
      "b", cl::desc("option b -- This option turns on option a"),
      cl::callback([&](const bool &) { OptA = true; }));
  StackOption<bool> OptC(
      "c", cl::desc("option c -- This option turns on options a and b"),
      cl::callback([&](const bool &) { OptB = true; }));
  StackOption<std::string, cl::list<std::string>> List(
      "list",
      cl::desc("option list -- This option turns on options a, b, and c when "
               "'foo' is included in list"),
      cl::CommaSeparated,
      cl::callback([&](const std::string &Str) {
        if (Str == "foo")
          OptC = true;
      }));

  const char *args1[] = {"prog", "-a"};
  EXPECT_TRUE(cl::ParseCommandLineOptions(2, args1));
  EXPECT_TRUE(OptA);
  EXPECT_FALSE(OptB);
  EXPECT_FALSE(OptC);
  EXPECT_EQ(List.size(), 0u);
  cl::ResetAllOptionOccurrences();

  const char *args2[] = {"prog", "-b"};
  EXPECT_TRUE(cl::ParseCommandLineOptions(2, args2));
  EXPECT_TRUE(OptA);
  EXPECT_TRUE(OptB);
  EXPECT_FALSE(OptC);
  EXPECT_EQ(List.size(), 0u);
  cl::ResetAllOptionOccurrences();

  const char *args3[] = {"prog", "-c"};
  EXPECT_TRUE(cl::ParseCommandLineOptions(2, args3));
  EXPECT_TRUE(OptA);
  EXPECT_TRUE(OptB);
  EXPECT_TRUE(OptC);
  EXPECT_EQ(List.size(), 0u);
  cl::ResetAllOptionOccurrences();

  const char *args4[] = {"prog", "--list=foo,bar"};
  EXPECT_TRUE(cl::ParseCommandLineOptions(2, args4));
  EXPECT_TRUE(OptA);
  EXPECT_TRUE(OptB);
  EXPECT_TRUE(OptC);
  EXPECT_EQ(List.size(), 2u);
  cl::ResetAllOptionOccurrences();

  const char *args5[] = {"prog", "--list=bar"};
  EXPECT_TRUE(cl::ParseCommandLineOptions(2, args5));
  EXPECT_FALSE(OptA);
  EXPECT_FALSE(OptB);
  EXPECT_FALSE(OptC);
  EXPECT_EQ(List.size(), 1u);

  cl::ResetAllOptionOccurrences();
}

enum Enum { Val1, Val2 };
static cl::bits<Enum> ExampleBits(
    cl::desc("An example cl::bits to ensure it compiles"),
    cl::values(
      clEnumValN(Val1, "bits-val1", "The Val1 value"),
      clEnumValN(Val1, "bits-val2", "The Val2 value")));

TEST(CommandLineTest, ConsumeAfterOnePositional) {
  cl::ResetCommandLineParser();

  // input [args]
  StackOption<std::string, cl::opt<std::string>> Input(cl::Positional,
                                                       cl::Required);
  StackOption<std::string, cl::list<std::string>> ExtraArgs(cl::ConsumeAfter);

  const char *Args[] = {"prog", "input", "arg1", "arg2"};

  std::string Errs;
  raw_string_ostream OS(Errs);
  EXPECT_TRUE(cl::ParseCommandLineOptions(4, Args, StringRef(), &OS));
  OS.flush();
  EXPECT_EQ("input", Input);
  EXPECT_EQ(ExtraArgs.size(), 2u);
  EXPECT_EQ(ExtraArgs[0], "arg1");
  EXPECT_EQ(ExtraArgs[1], "arg2");
  EXPECT_TRUE(Errs.empty());
}

TEST(CommandLineTest, ConsumeAfterTwoPositionals) {
  cl::ResetCommandLineParser();

  // input1 input2 [args]
  StackOption<std::string, cl::opt<std::string>> Input1(cl::Positional,
                                                        cl::Required);
  StackOption<std::string, cl::opt<std::string>> Input2(cl::Positional,
                                                        cl::Required);
  StackOption<std::string, cl::list<std::string>> ExtraArgs(cl::ConsumeAfter);

  const char *Args[] = {"prog", "input1", "input2", "arg1", "arg2"};

  std::string Errs;
  raw_string_ostream OS(Errs);
  EXPECT_TRUE(cl::ParseCommandLineOptions(5, Args, StringRef(), &OS));
  OS.flush();
  EXPECT_EQ("input1", Input1);
  EXPECT_EQ("input2", Input2);
  EXPECT_EQ(ExtraArgs.size(), 2u);
  EXPECT_EQ(ExtraArgs[0], "arg1");
  EXPECT_EQ(ExtraArgs[1], "arg2");
  EXPECT_TRUE(Errs.empty());
}

TEST(CommandLineTest, ResetAllOptionOccurrences) {
  cl::ResetCommandLineParser();

  // -option -str -enableA -enableC [sink] input [args]
  StackOption<bool> Option("option");
  StackOption<std::string> Str("str");
  enum Vals { ValA, ValB, ValC };
  StackOption<Vals, cl::bits<Vals>> Bits(
      cl::values(clEnumValN(ValA, "enableA", "Enable A"),
                 clEnumValN(ValB, "enableB", "Enable B"),
                 clEnumValN(ValC, "enableC", "Enable C")));
  StackOption<std::string, cl::list<std::string>> Sink(cl::Sink);
  StackOption<std::string> Input(cl::Positional);
  StackOption<std::string, cl::list<std::string>> ExtraArgs(cl::ConsumeAfter);

  const char *Args[] = {"prog",     "-option",  "-str=STR", "-enableA",
                        "-enableC", "-unknown", "input",    "-arg"};

  std::string Errs;
  raw_string_ostream OS(Errs);
  EXPECT_TRUE(cl::ParseCommandLineOptions(8, Args, StringRef(), &OS));
  EXPECT_TRUE(OS.str().empty());

  EXPECT_TRUE(Option);
  EXPECT_EQ("STR", Str);
  EXPECT_EQ((1u << ValA) | (1u << ValC), Bits.getBits());
  EXPECT_EQ(1u, Sink.size());
  EXPECT_EQ("-unknown", Sink[0]);
  EXPECT_EQ("input", Input);
  EXPECT_EQ(1u, ExtraArgs.size());
  EXPECT_EQ("-arg", ExtraArgs[0]);

  cl::ResetAllOptionOccurrences();
  EXPECT_FALSE(Option);
  EXPECT_EQ("", Str);
  EXPECT_EQ(0u, Bits.getBits());
  EXPECT_EQ(0u, Sink.size());
  EXPECT_EQ(0, Input.getNumOccurrences());
  EXPECT_EQ(0u, ExtraArgs.size());
}

TEST(CommandLineTest, DefaultValue) {
  cl::ResetCommandLineParser();

  StackOption<bool> BoolOption("bool-option");
  StackOption<std::string> StrOption("str-option");
  StackOption<bool> BoolInitOption("bool-init-option", cl::init(true));
  StackOption<std::string> StrInitOption("str-init-option",
                                         cl::init("str-default-value"));

  const char *Args[] = {"prog"}; // no options

  std::string Errs;
  raw_string_ostream OS(Errs);
  EXPECT_TRUE(cl::ParseCommandLineOptions(1, Args, StringRef(), &OS));
  EXPECT_TRUE(OS.str().empty());

  EXPECT_TRUE(!BoolOption);
  EXPECT_FALSE(BoolOption.Default.hasValue());
  EXPECT_EQ(0, BoolOption.getNumOccurrences());

  EXPECT_EQ("", StrOption);
  EXPECT_FALSE(StrOption.Default.hasValue());
  EXPECT_EQ(0, StrOption.getNumOccurrences());

  EXPECT_TRUE(BoolInitOption);
  EXPECT_TRUE(BoolInitOption.Default.hasValue());
  EXPECT_EQ(0, BoolInitOption.getNumOccurrences());

  EXPECT_EQ("str-default-value", StrInitOption);
  EXPECT_TRUE(StrInitOption.Default.hasValue());
  EXPECT_EQ(0, StrInitOption.getNumOccurrences());

  const char *Args2[] = {"prog", "-bool-option", "-str-option=str-value",
                         "-bool-init-option=0",
                         "-str-init-option=str-init-value"};

  EXPECT_TRUE(cl::ParseCommandLineOptions(5, Args2, StringRef(), &OS));
  EXPECT_TRUE(OS.str().empty());

  EXPECT_TRUE(BoolOption);
  EXPECT_FALSE(BoolOption.Default.hasValue());
  EXPECT_EQ(1, BoolOption.getNumOccurrences());

  EXPECT_EQ("str-value", StrOption);
  EXPECT_FALSE(StrOption.Default.hasValue());
  EXPECT_EQ(1, StrOption.getNumOccurrences());

  EXPECT_FALSE(BoolInitOption);
  EXPECT_TRUE(BoolInitOption.Default.hasValue());
  EXPECT_EQ(1, BoolInitOption.getNumOccurrences());

  EXPECT_EQ("str-init-value", StrInitOption);
  EXPECT_TRUE(StrInitOption.Default.hasValue());
  EXPECT_EQ(1, StrInitOption.getNumOccurrences());
}

TEST(CommandLineTest, HelpWithoutSubcommands) {
  // Check that the help message does not contain the "[subcommand]" placeholder
  // and the "SUBCOMMANDS" section if there are no subcommands.
  cl::ResetCommandLineParser();
  StackOption<bool> Opt("opt", cl::init(false));
  const char *args[] = {"prog"};
  EXPECT_TRUE(cl::ParseCommandLineOptions(std::size(args), args, StringRef(),
                                          &llvm::nulls()));
  auto Output = interceptStdout([]() { cl::PrintHelpMessage(); });
  EXPECT_NE(std::string::npos, Output.find("USAGE: prog [options]")) << Output;
  EXPECT_EQ(std::string::npos, Output.find("SUBCOMMANDS:")) << Output;
  cl::ResetCommandLineParser();
}

TEST(CommandLineTest, HelpWithSubcommands) {
  // Check that the help message contains the "[subcommand]" placeholder in the
  // "USAGE" line and describes subcommands.
  cl::ResetCommandLineParser();
  StackSubCommand SC1("sc1", "First Subcommand");
  StackSubCommand SC2("sc2", "Second Subcommand");
  StackOption<bool> SC1Opt("sc1", cl::sub(SC1), cl::init(false));
  StackOption<bool> SC2Opt("sc2", cl::sub(SC2), cl::init(false));
  const char *args[] = {"prog"};
  EXPECT_TRUE(cl::ParseCommandLineOptions(std::size(args), args, StringRef(),
                                          &llvm::nulls()));
  auto Output = interceptStdout([]() { cl::PrintHelpMessage(); });
  EXPECT_NE(std::string::npos,
            Output.find("USAGE: prog [subcommand] [options]"))
      << Output;
  EXPECT_NE(std::string::npos, Output.find("SUBCOMMANDS:")) << Output;
  EXPECT_NE(std::string::npos, Output.find("sc1 - First Subcommand")) << Output;
  EXPECT_NE(std::string::npos, Output.find("sc2 - Second Subcommand"))
      << Output;
  cl::ResetCommandLineParser();
}

TEST(CommandLineTest, UnknownCommands) {
  cl::ResetCommandLineParser();

  StackSubCommand SC1("foo", "Foo subcommand");
  StackSubCommand SC2("bar", "Bar subcommand");
  StackOption<bool> SC1Opt("put", cl::sub(SC1));
  StackOption<bool> SC2Opt("get", cl::sub(SC2));
  StackOption<bool> TopOpt1("peek");
  StackOption<bool> TopOpt2("set");

  std::string Errs;
  raw_string_ostream OS(Errs);

  const char *Args1[] = {"prog", "baz", "--get"};
  EXPECT_FALSE(
      cl::ParseCommandLineOptions(std::size(Args1), Args1, StringRef(), &OS));
  EXPECT_EQ(Errs,
            "prog: Unknown subcommand 'baz'.  Try: 'prog --help'\n"
            "prog: Did you mean 'bar'?\n"
            "prog: Unknown command line argument '--get'.  Try: 'prog --help'\n"
            "prog: Did you mean '--set'?\n");

  // Do not show a suggestion if the subcommand is not similar to any known.
  Errs.clear();
  const char *Args2[] = {"prog", "faz"};
  EXPECT_FALSE(
      cl::ParseCommandLineOptions(std::size(Args2), Args2, StringRef(), &OS));
  EXPECT_EQ(Errs, "prog: Unknown subcommand 'faz'.  Try: 'prog --help'\n");
}

TEST(CommandLineTest, SubCommandGroups) {
  // Check that options in subcommand groups are associated with expected
  // subcommands.

  cl::ResetCommandLineParser();

  StackSubCommand SC1("sc1", "SC1 subcommand");
  StackSubCommand SC2("sc2", "SC2 subcommand");
  StackSubCommand SC3("sc3", "SC3 subcommand");
  cl::SubCommandGroup Group12 = {&SC1, &SC2};

  StackOption<bool> Opt12("opt12", cl::sub(Group12), cl::init(false));
  StackOption<bool> Opt3("opt3", cl::sub(SC3), cl::init(false));

  // The "--opt12" option is expected to be added to both subcommands in the
  // group, but not to the top-level "no subcommand" pseudo-subcommand or the
  // "sc3" subcommand.
  EXPECT_EQ(1U, SC1.OptionsMap.size());
  EXPECT_TRUE(SC1.OptionsMap.contains("opt12"));

  EXPECT_EQ(1U, SC2.OptionsMap.size());
  EXPECT_TRUE(SC2.OptionsMap.contains("opt12"));

  EXPECT_FALSE(cl::SubCommand::getTopLevel().OptionsMap.contains("opt12"));
  EXPECT_FALSE(SC3.OptionsMap.contains("opt12"));
}

TEST(CommandLineTest, HelpWithEmptyCategory) {
  cl::ResetCommandLineParser();

  cl::OptionCategory Category1("First Category");
  cl::OptionCategory Category2("Second Category");
  StackOption<int> Opt1("opt1", cl::cat(Category1));
  StackOption<int> Opt2("opt2", cl::cat(Category2));
  cl::HideUnrelatedOptions(Category2);

  const char *args[] = {"prog"};
  EXPECT_TRUE(cl::ParseCommandLineOptions(std::size(args), args, StringRef(),
                                          &llvm::nulls()));
  auto Output = interceptStdout(
      []() { cl::PrintHelpMessage(/*Hidden=*/false, /*Categorized=*/true); });
  EXPECT_EQ(std::string::npos, Output.find("First Category"))
      << "An empty category should not be printed";

  Output = interceptStdout(
      []() { cl::PrintHelpMessage(/*Hidden=*/true, /*Categorized=*/true); });
  EXPECT_EQ(std::string::npos, Output.find("First Category"))
      << "An empty category should not be printed";

  cl::ResetCommandLineParser();
}

} // anonymous namespace
