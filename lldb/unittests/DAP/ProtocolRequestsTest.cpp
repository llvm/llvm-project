//===-- ProtocolRequestsTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Protocol/ProtocolRequests.h"
#include "Protocol/ProtocolTypes.h"
#include "TestingSupport/TestUtilities.h"
#include "llvm/Testing/Support/Error.h"
#include <gtest/gtest.h>

using namespace llvm;
using namespace lldb_dap::protocol;
using lldb_private::PrettyPrint;
using llvm::json::parse;

TEST(ProtocolRequestsTest, ExceptionInfoArguments) {
  llvm::Expected<ExceptionInfoArguments> expected =
      parse<ExceptionInfoArguments>(R"({
        "threadId": 3434
        })");
  ASSERT_THAT_EXPECTED(expected, llvm::Succeeded());
  EXPECT_EQ(expected->threadId, 3434U);

  // Check required keys;
  EXPECT_THAT_EXPECTED(parse<ExceptionInfoArguments>(R"({})"),
                       FailedWithMessage("missing value at (root).threadId"));

  EXPECT_THAT_EXPECTED(parse<ExceptionInfoArguments>(R"({"id": 10})"),
                       FailedWithMessage("missing value at (root).threadId"));
}

TEST(ProtocolRequestsTest, ExceptionInfoResponseBody) {
  ExceptionInfoResponseBody body;
  body.exceptionId = "signal";
  body.breakMode = eExceptionBreakModeAlways;

  // Check required keys.
  Expected<json::Value> expected = parse(
      R"({
    "exceptionId": "signal",
    "breakMode": "always"
    })");

  ASSERT_THAT_EXPECTED(expected, llvm::Succeeded());
  EXPECT_EQ(PrettyPrint(*expected), PrettyPrint(body));

  // Check optional keys.
  body.description = "SIGNAL SIGWINCH";
  body.breakMode = eExceptionBreakModeNever;
  body.details = ExceptionDetails{};
  body.details->message = "some message";

  Expected<json::Value> expected_opt = parse(
      R"({
    "exceptionId": "signal",
    "description": "SIGNAL SIGWINCH",
    "breakMode": "never",
    "details": {
      "message": "some message"
    }
  })");

  ASSERT_THAT_EXPECTED(expected_opt, llvm::Succeeded());
  EXPECT_EQ(PrettyPrint(*expected_opt), PrettyPrint(body));
}
