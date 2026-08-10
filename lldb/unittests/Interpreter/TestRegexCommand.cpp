//===-- TestRegexCommand.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Commands/CommandObjectRegexCommand.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb;

namespace {
class TestRegexCommand : public CommandObjectRegexCommand {
public:
  using CommandObjectRegexCommand::SubstituteVariables;

  static std::string
  Substitute(llvm::StringRef input,
             const llvm::SmallVectorImpl<llvm::StringRef> &replacements) {
    llvm::Expected<std::string> str = SubstituteVariables(input, replacements);
    if (!str)
      return llvm::toString(str.takeError());
    return *str;
  }
};
} // namespace

TEST(RegexCommandTest, SubstituteVariablesSuccess) {
  const llvm::SmallVector<llvm::StringRef, 4> substitutions = {"all", "foo",
                                                               "bar", "baz"};

  EXPECT_EQ(TestRegexCommand::Substitute("%0", substitutions), "all");
  EXPECT_EQ(TestRegexCommand::Substitute("%1", substitutions), "foo");
  EXPECT_EQ(TestRegexCommand::Substitute("%2", substitutions), "bar");
  EXPECT_EQ(TestRegexCommand::Substitute("%3", substitutions), "baz");
  EXPECT_EQ(TestRegexCommand::Substitute("%1%2%3", substitutions), "foobarbaz");
  EXPECT_EQ(TestRegexCommand::Substitute("#%1#%2#%3#", substitutions),
            "#foo#bar#baz#");
}

TEST(RegexCommandTest, SubstituteVariablesFailed) {
  const llvm::SmallVector<llvm::StringRef, 4> substitutions = {"all", "foo",
                                                               "bar", "baz"};

  ASSERT_THAT_EXPECTED(
      TestRegexCommand::SubstituteVariables("%1%2%3%4", substitutions),
      llvm::Failed());
  ASSERT_THAT_EXPECTED(
      TestRegexCommand::SubstituteVariables("%5", substitutions),
      llvm::Failed());
  ASSERT_THAT_EXPECTED(
      TestRegexCommand::SubstituteVariables("%11", substitutions),
      llvm::Failed());
}

TEST(RegexCommandTest, SubstituteVariablesNoRecursion) {
  const llvm::SmallVector<llvm::StringRef, 4> substitutions = {"all", "%2",
                                                               "%3", "%4"};
  EXPECT_EQ(TestRegexCommand::Substitute("%0", substitutions), "all");
  EXPECT_EQ(TestRegexCommand::Substitute("%1", substitutions), "%2");
  EXPECT_EQ(TestRegexCommand::Substitute("%2", substitutions), "%3");
  EXPECT_EQ(TestRegexCommand::Substitute("%3", substitutions), "%4");
  EXPECT_EQ(TestRegexCommand::Substitute("%1%2%3", substitutions), "%2%3%4");
}
