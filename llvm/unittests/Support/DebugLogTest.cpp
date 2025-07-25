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
    EXPECT_TRUE(StringRef(os.str()).starts_with('['));
    EXPECT_TRUE(StringRef(os.str()).ends_with("NoType\n"));
  }

  setCurrentDebugTypes(DT, 2);
  {
    std::string str;
    raw_string_ostream os(str);
    DEBUGLOG_WITH_STREAM_AND_TYPE(os, "A") << "A";
    DEBUGLOG_WITH_STREAM_AND_TYPE(os, "B") << "B";
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
