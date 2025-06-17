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
#include "llvm/ADT/StringRef.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <optional>

using namespace llvm;
using namespace lldb;
using namespace lldb_dap;
using namespace lldb_dap::protocol;
using lldb_private::pp;
using llvm::json::parse;

TEST(ProtocolRequestsTest, ThreadResponseBody) {
  const ThreadsResponseBody body{{{1, "thr1"}, {2, "thr2"}}};
  const StringRef json = R"({
  "threads": [
    {
      "id": 1,
      "name": "thr1"
    },
    {
      "id": 2,
      "name": "thr2"
    }
  ]
})";
  // Validate toJSON
  EXPECT_EQ(json, pp(body));
}

TEST(ProtocolRequestsTest, SetExceptionBreakpointsArguments) {
  EXPECT_THAT_EXPECTED(
      parse<SetExceptionBreakpointsArguments>(R"({"filters":[]})"),
      HasValue(testing::FieldsAre(/*filters=*/testing::IsEmpty(),
                                  /*filterOptions=*/testing::IsEmpty())));
  EXPECT_THAT_EXPECTED(
      parse<SetExceptionBreakpointsArguments>(R"({"filters":["abc"]})"),
      HasValue(testing::FieldsAre(/*filters=*/std::vector<std::string>{"abc"},
                                  /*filterOptions=*/testing::IsEmpty())));
  EXPECT_THAT_EXPECTED(
      parse<SetExceptionBreakpointsArguments>(
          R"({"filters":[],"filterOptions":[{"filterId":"abc"}]})"),
      HasValue(testing::FieldsAre(
          /*filters=*/testing::IsEmpty(),
          /*filterOptions=*/testing::Contains(testing::FieldsAre(
              /*filterId=*/"abc", /*condition=*/"", /*mode=*/"")))));

  // Validate parse errors
  EXPECT_THAT_EXPECTED(parse<SetExceptionBreakpointsArguments>(R"({})"),
                       FailedWithMessage("missing value at (root).filters"));
  EXPECT_THAT_EXPECTED(
      parse<SetExceptionBreakpointsArguments>(R"({"filters":false})"),
      FailedWithMessage("expected array at (root).filters"));
}

TEST(ProtocolRequestsTest, SetExceptionBreakpointsResponseBody) {
  SetExceptionBreakpointsResponseBody body;
  Breakpoint bp;
  bp.id = 12, bp.verified = true;
  body.breakpoints = {bp};
  EXPECT_EQ(R"({
  "breakpoints": [
    {
      "id": 12,
      "verified": true
    }
  ]
})",
            pp(body));
}
