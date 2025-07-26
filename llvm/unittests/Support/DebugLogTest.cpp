//===- llvm/unittest/Support/DebugLogTest.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/DebugLog.h"
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
    DEBUGLOG_WITH_STREAM_AND_TYPE(os, nullptr) << "NoType";
    EXPECT_FALSE(StringRef(os.str()).starts_with('['));
    EXPECT_TRUE(StringRef(os.str()).ends_with("NoType\n"));
  }

  setCurrentDebugTypes(DT, 2);
  {
    std::string str;
    raw_string_ostream os(str);
    DEBUGLOG_WITH_STREAM_AND_TYPE(os, "A") << "A";
    DEBUGLOG_WITH_STREAM_AND_TYPE(os, "B") << "B";
    EXPECT_TRUE(StringRef(os.str()).starts_with('['));
    EXPECT_THAT(os.str(), AllOf(HasSubstr("A\n"), HasSubstr("B\n")));
  }

  setCurrentDebugType("A");
  {
    std::string str;
    raw_string_ostream os(str);
    // Just check that the macro doesn't result in dangling else.
    if (true)
      DEBUGLOG_WITH_STREAM_AND_TYPE(os, "A") << "A";
    else
      DEBUGLOG_WITH_STREAM_AND_TYPE(os, "A") << "B";
    DEBUGLOG_WITH_STREAM_AND_TYPE(os, "B") << "B";
    EXPECT_THAT(os.str(), AllOf(HasSubstr("A\n"), Not(HasSubstr("B\n"))));

    int count = 0;
    auto inc = [&]() { return ++count; };
    EXPECT_THAT(count, Eq(0));
    DEBUGLOG_WITH_STREAM_AND_TYPE(os, "A") << inc();
    EXPECT_THAT(count, Eq(1));
    DEBUGLOG_WITH_STREAM_AND_TYPE(os, "B") << inc();
    EXPECT_THAT(count, Eq(1));
  }
}

TEST(DebugLogTest, StreamPrefix) {
  llvm::DebugFlag = true;
  static const char *DT[] = {"A", "B"};
  setCurrentDebugTypes(DT, 2);

  std::string str;
  raw_string_ostream os(str);
  std::string expected = "[Prefix] A:1 1\n[Prefix] A:1 2\n[Prefix] B:1 "
                         "3\n[Prefix] B:1 4\n[Prefix] A:1 5";
  {
    llvm::impl::LogWithNewline ldbg_osB("Prefix", "B", 1, os);
    llvm::impl::LogWithNewline ldbg_osA("Prefix", "A", 1, os);
    ldbg_osA.stream() << "1\n2\n";
    ldbg_osB.stream() << "3\n4\n";
    ldbg_osA.stream() << "5";
    EXPECT_EQ(os.str(), expected);
  }
  // After destructors, there was a pending newline for stream B.
  EXPECT_EQ(os.str(), expected + "\n[Prefix] B:1 \n");
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
