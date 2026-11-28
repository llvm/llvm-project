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

class SBEventTest : public SBDebuggerFixture {};

TEST_F(SBEventTest, BroadcasterClassBeforeBroadcast) {
  SBEvent event(1, "data", 4);
  EXPECT_STREQ(event.GetBroadcasterClass(), "unknown class");
  EXPECT_FALSE(SBCommandInterpreter::EventIsCommandInterpreterEvent(event));
}
