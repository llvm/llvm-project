//===-- TestCommandHistoryWithContext.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/CommandHistoryWithContext.h"
#include "lldb/Utility/FileSpec.h"

#include "llvm/Support/FileSystem.h"

#include "gtest/gtest.h"

using namespace lldb_private;

static CommandExecutionContext MakeContext(std::string target, std::string file,
                                           uint32_t line,
                                           std::string function) {
  CommandExecutionContext context;
  context.target_name = std::move(target);
  context.file = std::move(file);
  context.line = line;
  context.function_name = std::move(function);
  return context;
}

TEST(CommandHistoryWithContextTest, EmptyHistory) {
  CommandHistoryWithContext history((FileSpec()));
  EXPECT_EQ(history.FindSuggestion("print", CommandExecutionContext()),
            std::nullopt);
}

TEST(CommandHistoryWithContextTest, EmptyLineHasNoSuggestion) {
  CommandHistoryWithContext history((FileSpec()));
  history.AppendCommand("print x", CommandExecutionContext());
  EXPECT_EQ(history.FindSuggestion("", CommandExecutionContext()),
            std::nullopt);
}

TEST(CommandHistoryWithContextTest, SuggestionIsRemainderAfterPrefix) {
  CommandHistoryWithContext history((FileSpec()));
  history.AppendCommand("print hello", CommandExecutionContext());
  EXPECT_EQ(history.FindSuggestion("print ", CommandExecutionContext()),
            std::optional<std::string>("hello"));
}

TEST(CommandHistoryWithContextTest, FallsBackToMostRecentPrefixMatch) {
  CommandHistoryWithContext history((FileSpec()));
  history.AppendCommand("print older", CommandExecutionContext());
  history.AppendCommand("print newer", CommandExecutionContext());
  // With no meaningful current context, the most recent prefix match wins.
  EXPECT_EQ(history.FindSuggestion("print ", CommandExecutionContext()),
            std::optional<std::string>("newer"));
}

TEST(CommandHistoryWithContextTest, PrefersSameFileAndLine) {
  CommandHistoryWithContext history((FileSpec()));
  history.AppendCommand("print at_a", MakeContext("a.out", "a.c", 10, "foo"));
  history.AppendCommand("print at_b", MakeContext("a.out", "b.c", 20, "bar"));
  history.AppendCommand("print at_c", MakeContext("a.out", "c.c", 30, "baz"));

  // Even though "print at_c" is the most recent, the command from the same
  // file and line must be preferred.
  CommandExecutionContext current = MakeContext("a.out", "a.c", 10, "bar");
  EXPECT_EQ(history.FindSuggestion("print ", current),
            std::optional<std::string>("at_a"));
}

TEST(CommandHistoryWithContextTest, PrefersMostRecentSameFileAndLine) {
  CommandHistoryWithContext history((FileSpec()));
  history.AppendCommand("print first", MakeContext("a.out", "a.c", 10, "foo"));
  history.AppendCommand("print second", MakeContext("a.out", "a.c", 10, "foo"));

  CommandExecutionContext current = MakeContext("a.out", "a.c", 10, "foo");
  EXPECT_EQ(history.FindSuggestion("print ", current),
            std::optional<std::string>("second"));
}

TEST(CommandHistoryWithContextTest, PrefersSameFunctionWhenNoFileLineMatch) {
  CommandHistoryWithContext history((FileSpec()));
  history.AppendCommand("print at_a", MakeContext("a.out", "a.c", 10, "foo"));
  history.AppendCommand("print at_b", MakeContext("a.out", "b.c", 20, "bar"));
  history.AppendCommand("print at_c", MakeContext("a.out", "c.c", 30, "baz"));

  // No entry shares the current file/line, so the same-function entry wins.
  CommandExecutionContext current = MakeContext("a.out", "z.c", 99, "bar");
  EXPECT_EQ(history.FindSuggestion("print ", current),
            std::optional<std::string>("at_b"));
}

TEST(CommandHistoryWithContextTest, PrefersSameTargetWhenNoBetterMatch) {
  CommandHistoryWithContext history((FileSpec()));
  history.AppendCommand("print at_a", MakeContext("a.out", "a.c", 10, "foo"));
  history.AppendCommand("print at_b", MakeContext("b.out", "b.c", 20, "bar"));

  // Neither file/line nor function match; the same-target entry is chosen.
  CommandExecutionContext current = MakeContext("a.out", "z.c", 99, "nope");
  EXPECT_EQ(history.FindSuggestion("print ", current),
            std::optional<std::string>("at_a"));
}

TEST(CommandHistoryWithContextTest, MergesDuplicateCommandsAndRefreshesContext) {
  CommandHistoryWithContext history((FileSpec()));
  history.AppendCommand("print x", MakeContext("a.out", "a.c", 10, "foo"));
  // Re-invoking the same command refreshes its context to the latest one and
  // makes it the most recently used entry rather than adding a duplicate.
  history.AppendCommand("print x", MakeContext("a.out", "b.c", 20, "bar"));

  // The refreshed context (b.c:20) is what gets matched now.
  CommandExecutionContext at_new = MakeContext("a.out", "b.c", 20, "bar");
  EXPECT_EQ(history.FindSuggestion("print ", at_new),
            std::optional<std::string>("x"));
}

TEST(CommandHistoryWithContextTest, EvictsLeastFrequentlyInvoked) {
  CommandHistoryWithContext history((FileSpec()));
  const size_t max = CommandHistoryWithContext::GetMaxEntries();
  // Fill the history with distinct commands, each invoked once. The trailing
  // ';' keeps commands from being prefixes of one another.
  for (size_t i = 0; i < max; ++i)
    history.AppendCommand("c" + std::to_string(i) + ";",
                          CommandExecutionContext());

  // Invoke "c0;" more often so it is the most frequently used command.
  history.AppendCommand("c0;", CommandExecutionContext());
  history.AppendCommand("c0;", CommandExecutionContext());

  // Adding a brand new command overflows the history and forces one eviction.
  history.AppendCommand("newcmd;", CommandExecutionContext());

  // The frequently invoked command and the newest command survive.
  EXPECT_EQ(history.FindSuggestion("c0;", CommandExecutionContext()),
            std::optional<std::string>(""));
  EXPECT_EQ(history.FindSuggestion("newcmd;", CommandExecutionContext()),
            std::optional<std::string>(""));
  // The oldest of the least-invoked (count 1) commands, "c1;", is evicted.
  EXPECT_EQ(history.FindSuggestion("c1;", CommandExecutionContext()),
            std::nullopt);
  // The next-oldest least-invoked command is kept.
  EXPECT_EQ(history.FindSuggestion("c2;", CommandExecutionContext()),
            std::optional<std::string>(""));
}

TEST(CommandHistoryWithContextTest, CapsAtMaxEntries) {
  CommandHistoryWithContext history((FileSpec()));
  const size_t max = CommandHistoryWithContext::GetMaxEntries();
  // Append more than the cap; each command is distinct.
  for (size_t i = 0; i < max + 50; ++i)
    history.AppendCommand("cmd" + std::to_string(i), CommandExecutionContext());

  // The oldest entries were dropped, so "cmd0" is gone but the newest remains.
  EXPECT_EQ(history.FindSuggestion("cmd0 ", CommandExecutionContext()),
            std::nullopt);
  EXPECT_EQ(history.FindSuggestion("cmd" + std::to_string(max + 49),
                                   CommandExecutionContext()),
            std::optional<std::string>(""));
}

TEST(CommandHistoryWithContextTest, PersistsAcrossInstances) {
  llvm::SmallString<128> temp_path;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile("lldb-history", "json",
                                                  temp_path));
  FileSpec history_file(temp_path);

  {
    CommandHistoryWithContext history(history_file);
    history.AppendCommand("print saved",
                          MakeContext("a.out", "a.c", 42, "foo"));
  }

  // A fresh instance pointed at the same file should load the saved command
  // together with its context.
  CommandHistoryWithContext reloaded(history_file);
  reloaded.Load();
  CommandExecutionContext current = MakeContext("a.out", "a.c", 42, "foo");
  EXPECT_EQ(reloaded.FindSuggestion("print ", current),
            std::optional<std::string>("saved"));

  llvm::sys::fs::remove(temp_path);
}
