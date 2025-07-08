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
using testing::HasSubstr;

#ifndef NDEBUG
TEST(DebugLogTest, Basic) {
  std::string s1, s2;
  raw_string_ostream os1(s1), os2(s2);
  static const char *DT[] = {"A", "B"};

  llvm::DebugFlag = true;
  setCurrentDebugTypes(DT, 2);
  DEBUGLOG_WITH_STREAM_AND_TYPE(os1, "A") << "A";
  DEBUGLOG_WITH_STREAM_AND_TYPE(os1, "B") << "B";
  EXPECT_THAT(os1.str(), AllOf(HasSubstr("A\n"), HasSubstr("B\n")));

  setCurrentDebugType("A");
  volatile int x = 0;
  if (x == 0)
    DEBUGLOG_WITH_STREAM_AND_TYPE(os2, "A") << "A";
  else
    DEBUGLOG_WITH_STREAM_AND_TYPE(os2, "A") << "B";
  DEBUGLOG_WITH_STREAM_AND_TYPE(os2, "B") << "B";
  EXPECT_THAT(os2.str(), AllOf(HasSubstr("A\n"), Not(HasSubstr("B\n"))));
}
#endif
