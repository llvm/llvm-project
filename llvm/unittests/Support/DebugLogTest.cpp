//===- llvm/unittest/Support/DebugLogTest.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/DebugLog.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/raw_ostream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <string>
using namespace llvm;
using testing::Eq;
using testing::HasSubstr;

#ifndef NDEBUG
TEST(DebugLogTest, Basic) {
  llvm::DebugFlag = true;
  static const char *DT[] = {"A", "B"};

  // Clear debug types.
  setCurrentDebugTypes(DT, 0);
  {
    std::string str;
    raw_string_ostream os(str);
    DEBUGLOG_WITH_STREAM_AND_TYPE(os, 0, nullptr) << "NoType";
    EXPECT_FALSE(StringRef(os.str()).starts_with('['));
    EXPECT_TRUE(StringRef(os.str()).ends_with("NoType\n"));
  }

  setCurrentDebugTypes(DT, 2);
  {
    std::string str;
    raw_string_ostream os(str);
    DEBUGLOG_WITH_STREAM_AND_TYPE(os, 0, "A") << "A";
    DEBUGLOG_WITH_STREAM_AND_TYPE(os, 0, "B") << "B";
    EXPECT_TRUE(StringRef(os.str()).starts_with('['));
    EXPECT_THAT(os.str(), AllOf(HasSubstr("A\n"), HasSubstr("B\n")));
  }

  setCurrentDebugType("A");
  {
    std::string str;
    raw_string_ostream os(str);
    // Just check that the macro doesn't result in dangling else.
    if (true)
      DEBUGLOG_WITH_STREAM_AND_TYPE(os, 0, "A") << "A";
    else
      DEBUGLOG_WITH_STREAM_AND_TYPE(os, 0, "A") << "B";
    DEBUGLOG_WITH_STREAM_AND_TYPE(os, 0, "B") << "B";
    EXPECT_THAT(os.str(), AllOf(HasSubstr("A\n"), Not(HasSubstr("B\n"))));

    int count = 0;
    auto inc = [&]() { return ++count; };
    EXPECT_THAT(count, Eq(0));
    DEBUGLOG_WITH_STREAM_AND_TYPE(os, 0, "A") << inc();
    EXPECT_THAT(count, Eq(1));
    DEBUGLOG_WITH_STREAM_AND_TYPE(os, 0, "B") << inc();
    EXPECT_THAT(count, Eq(1));
  }
}

TEST(DebugLogTest, BasicWithLevel) {
  llvm::DebugFlag = true;
  // We expect A to be always printed, B to be printed only when level is 1 or
  // below, and C to be printed only when level is 0 or below.
  static const char *DT[] = {"A", "B:1", "C:"};

  setCurrentDebugTypes(DT, sizeof(DT) / sizeof(DT[0]));
  std::string str;
  raw_string_ostream os(str);
  for (auto type : {"A", "B", "C", "D"})
    for (int level : llvm::seq<int>(0, 4))
      DEBUGLOG_WITH_STREAM_TYPE_FILE_AND_LINE(os, level, type, type, level)
          << level;
  EXPECT_EQ(os.str(), "[A:0] A:0 0\n[A:1] A:1 1\n[A:2] A:2 2\n[A:3] A:3 "
                      "3\n[B:0] B:0 0\n[B:1] B:1 1\n[C:0] C:0 0\n");
}

TEST(DebugLogTest, NegativeLevel) {
  llvm::DebugFlag = true;
  // Test the special behavior when all the levels are 0.
  // In this case we expect all the debug types to be printed.
  static const char *DT[] = {"A:"};

  setCurrentDebugTypes(DT, sizeof(DT) / sizeof(DT[0]));
  std::string str;
  raw_string_ostream os(str);
  for (auto type : {"A", "B"})
    for (int level : llvm::seq<int>(0, 2))
      DEBUGLOG_WITH_STREAM_TYPE_FILE_AND_LINE(os, level, type, type, level)
          << level;
  EXPECT_EQ(os.str(), "[A:0] A:0 0\n[B:0] B:0 0\n[B:1] B:1 1\n");
}

TEST(DebugLogTest, StreamPrefix) {
  llvm::DebugFlag = true;
  static const char *DT[] = {"A", "B"};
  setCurrentDebugTypes(DT, 2);

  std::string str;
  raw_string_ostream os(str);
  std::string expected = "PrefixA 1\nPrefixA 2\nPrefixA \nPrefixB "
                         "3\nPrefixB 4\nPrefixA 5";
  {
    llvm::impl::raw_ldbg_ostream ldbg_osB("PrefixB ", os);
    llvm::impl::raw_ldbg_ostream ldbg_osA("PrefixA ", os);
    ldbg_osA << "1\n2";
    ldbg_osA << "\n\n";
    ldbg_osB << "3\n4\n";
    ldbg_osA << "5";
    EXPECT_EQ(os.str(), expected);
  }
  EXPECT_EQ(os.str(), expected);
}

TEST(DebugLogTest, DestructorPrefix) {
  llvm::DebugFlag = true;
  std::string str;
  raw_string_ostream os(str);
  {
    llvm::impl::raw_ldbg_ostream ldbg_osB("PrefixB ", os);
  }
  // After destructors, nothing should have been printed.
  EXPECT_EQ(os.str(), "");
}
#else
TEST(DebugLogTest, Basic) {
  // LDBG should be compiled out in NDEBUG, so just check it compiles and has
  // no effect.
  llvm::DebugFlag = true;
  static const char *DT[] = {"A"};
  setCurrentDebugTypes(DT, 0);
  int count = 0;
  auto inc = [&]() { return ++count; };
  EXPECT_THAT(count, Eq(0));
  LDBG() << inc();
  EXPECT_THAT(count, Eq(0));
}
#endif
