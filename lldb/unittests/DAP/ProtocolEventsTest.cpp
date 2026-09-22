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

TEST(ProtocolEventsTest, ProgressStartEventBody) {
  ProgressStartEventBody body;
  body.progressId = "1";
  body.title = "Parsing symbols";
  Expected<Value> expected_body = parse(R"({
    "progressId": "1",
    "title": "Parsing symbols"
  })");
  ASSERT_THAT_EXPECTED(expected_body, llvm::Succeeded());
  EXPECT_EQ(PrettyPrint(*expected_body), PrettyPrint(body));

  body.message = "foo.cpp";
  body.requestId = 30;
  body.percentage = 25;
  body.cancellable = true;
  expected_body = parse(R"({
    "progressId": "1",
    "title": "Parsing symbols",
    "message": "foo.cpp",
    "requestId": 30,
    "percentage": 25,
    "cancellable": true
  })");
  ASSERT_THAT_EXPECTED(expected_body, llvm::Succeeded());
  EXPECT_EQ(PrettyPrint(*expected_body), PrettyPrint(body));
}

TEST(ProtocolEventsTest, ProgressUpdateEventBody) {
  ProgressUpdateEventBody body;
  body.progressId = "1";
  Expected<Value> expected_body = parse(R"({
    "progressId": "1"
  })");
  ASSERT_THAT_EXPECTED(expected_body, llvm::Succeeded());
  EXPECT_EQ(PrettyPrint(*expected_body), PrettyPrint(body));

  body.message = "bar.cpp";
  body.percentage = 75;
  expected_body = parse(R"({
    "progressId": "1",
    "message": "bar.cpp",
    "percentage": 75
  })");
  ASSERT_THAT_EXPECTED(expected_body, llvm::Succeeded());
  EXPECT_EQ(PrettyPrint(*expected_body), PrettyPrint(body));
}

TEST(ProtocolEventsTest, ProgressEndEventBody) {
  ProgressEndEventBody body;
  body.progressId = "1";
  Expected<Value> expected_body = parse(R"({
    "progressId": "1"
  })");
  ASSERT_THAT_EXPECTED(expected_body, llvm::Succeeded());
  EXPECT_EQ(PrettyPrint(*expected_body), PrettyPrint(body));

  body.message = "done.";
  expected_body = parse(R"({
    "progressId": "1",
    "message": "done."
  })");
  ASSERT_THAT_EXPECTED(expected_body, llvm::Succeeded());
  EXPECT_EQ(PrettyPrint(*expected_body), PrettyPrint(body));
}
