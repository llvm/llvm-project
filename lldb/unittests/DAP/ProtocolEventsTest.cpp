//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Protocol/ProtocolEvents.h"
#include "TestingSupport/TestUtilities.h"
#include "llvm/Testing/Support/Error.h"
#include <gtest/gtest.h>

using namespace llvm;
using namespace lldb_dap::protocol;
using lldb_private::PrettyPrint;
using llvm::json::parse;
using llvm::json::Value;

TEST(ProtocolEventsTest, StoppedEventBody) {
  StoppedEventBody body;
  body.reason = lldb_dap::protocol::eStoppedReasonBreakpoint;
  Expected<Value> expected_body = parse(R"({
    "reason": "breakpoint"
  })");
  ASSERT_THAT_EXPECTED(expected_body, llvm::Succeeded());
  EXPECT_EQ(PrettyPrint(*expected_body), PrettyPrint(body));

  body.reason = eStoppedReasonBreakpoint;
  body.description = "desc";
  body.text = "text";
  body.preserveFocusHint = true;
  body.allThreadsStopped = true;
  body.hitBreakpointIds = {1, 2, 3};
  expected_body = parse(R"({
    "reason": "breakpoint",
    "allThreadsStopped": true,
    "description": "desc",
    "text": "text",
    "preserveFocusHint": true,
    "hitBreakpointIds": [1, 2, 3]
  })");
  ASSERT_THAT_EXPECTED(expected_body, llvm::Succeeded());
  EXPECT_EQ(PrettyPrint(*expected_body), PrettyPrint(body));
}
