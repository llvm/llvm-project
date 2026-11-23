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

class SBBreakpointNameTest : public SBDebuggerFixture {
protected:
  void SetUp() override {
    SBDebuggerFixture::SetUp();
    m_target = CreateEmptyTarget();
  }

  SBTarget m_target;
};

TEST_F(SBBreakpointNameTest, CompareInvalid) {
  SBBreakpointName name(m_target, "name");
  SBBreakpointName invalid;
  SBBreakpointName other_invalid;

  EXPECT_TRUE(invalid == other_invalid);
  EXPECT_FALSE(invalid != other_invalid);
  EXPECT_FALSE(invalid == name);
  EXPECT_TRUE(invalid != name);
  EXPECT_FALSE(name == invalid);
  EXPECT_TRUE(name != invalid);
}
