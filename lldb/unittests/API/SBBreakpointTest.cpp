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

class SBBreakpointTest : public SBDebuggerFixture {
protected:
  void SetUp() override {
    SBDebuggerFixture::SetUp();
    m_target = CreateEmptyTarget();
  }

  SBTarget m_target;
};

TEST_F(SBBreakpointTest, ListAppendIfUnique) {
  SBBreakpoint breakpoint = m_target.BreakpointCreateByAddress(0x1000);
  SBBreakpointList list(m_target);
  EXPECT_TRUE(list.AppendIfUnique(breakpoint));
  EXPECT_FALSE(list.AppendIfUnique(breakpoint));
  EXPECT_FALSE(list.AppendIfUnique(SBBreakpoint()));
  ASSERT_EQ(list.GetSize(), 1u);
  EXPECT_EQ(list.GetBreakpointAtIndex(0).GetID(), breakpoint.GetID());
}
