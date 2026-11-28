//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/LLDB.h"

#include "SBTestingSupport/SBTestUtilities.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

class SBTypeSummaryTest : public SBDebuggerFixture {};

TEST_F(SBTypeSummaryTest, InequalityOfInvalidObjects) {
  SBTypeSummary summary = SBTypeSummary::CreateWithSummaryString("text");
  SBTypeSummary invalid;
  SBTypeSummary other_invalid;
  EXPECT_FALSE(invalid != other_invalid);
  EXPECT_TRUE(invalid != summary);
}
